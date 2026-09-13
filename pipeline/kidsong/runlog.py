"""
kidsong.runlog — per-run logging owned by Python, not by the shell.

Why this exists: a kidsong run is 40+ minutes of unattended GPU work. Before
this module the only capture was shell redirection (``*>> $log`` in
output/batch_queue.ps1), which silently detached — a whole run's stdout and its
fatal traceback went nowhere, leaving eight good renders on disk and no way to
tell what killed the run.

Logging is therefore owned by the process doing the work:

* Every run writes ``output/<base>.log`` — the same ``<base>`` as the run's wav
  and ``-shots.json``, so a run's artifacts group together in one listing.
* ``<base>`` is only known after the lyrics come back, so a run starts logging
  immediately to ``output/_kidsong-startup-<stamp>-<pid>.log`` and *renames*
  that file once the base exists. A crash during config load / lyrics / singing
  still leaves a readable log.
* Every crash is appended to ``output/kidsong_errors.log`` — one stable,
  append-only place to see failures across all runs.

Console output is unchanged: a StreamHandler mirrors everything to stdout, so
the human-readable ``->`` progress lines still appear exactly as before.
"""
import logging
import os
import platform
import sys
import time
import traceback

#: Filename (inside the output dir) of the cross-run, append-only error log.
ERROR_LOG_NAME = "kidsong_errors.log"

#: Prefix for the pre-``<base>`` log file. Leading underscore keeps these
#: sorted away from the dated run artifacts in a directory listing.
STARTUP_PREFIX = "_kidsong-startup-"

_FILE_FORMAT = "%(asctime)s %(levelname)-7s %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class RunLog:
    """A run-scoped logger writing to stdout *and* a per-run file.

    Created before ``<base>`` is known::

        rl = RunLog(out_dir)              # -> output/_kidsong-startup-...log
        rl.rebind("20260720-085055-kidsong-splish-splash")
                                          # -> renamed to output/<base>.log

    The logger is instance-private (unique name, ``propagate=False``) so
    several runs — or several tests — never cross-write each other's files.
    """

    def __init__(self, out_dir, base=None, level=logging.INFO, stream=True):
        self.out_dir = str(out_dir)
        os.makedirs(self.out_dir, exist_ok=True)
        self.base = base
        self._closed = False
        self._reported = set()  # id(exc) already written to the error log

        self.logger = logging.getLogger(f"kidsong.run.{os.getpid()}.{id(self)}")
        self.logger.setLevel(level)
        self.logger.propagate = False
        self.logger.handlers.clear()

        if stream:
            sh = logging.StreamHandler(stream=sys.stdout)
            sh.setLevel(level)
            # The console keeps the bare human-readable line ("  -> Singing…"):
            # timestamps and levels belong in the file, not in the terminal the
            # operator is watching.
            sh.setFormatter(logging.Formatter("%(message)s"))
            self.logger.addHandler(sh)

        self._file_handler = None
        self.path = self._default_path(base)
        self._attach_file(self.path)

    # ------------------------------------------------------------- plumbing ---
    def _default_path(self, base):
        if base:
            return os.path.join(self.out_dir, f"{base}.log")
        stamp = time.strftime("%Y%m%d-%H%M%S")
        return os.path.join(self.out_dir, f"{STARTUP_PREFIX}{stamp}-{os.getpid()}.log")

    def _attach_file(self, path):
        """Point the file handler at `path` (append mode), replacing any old one."""
        self._detach_file()
        handler = logging.FileHandler(path, mode="a", encoding="utf-8", delay=False)
        handler.setLevel(self.logger.level)
        handler.setFormatter(logging.Formatter(_FILE_FORMAT, datefmt=_DATE_FORMAT))
        self.logger.addHandler(handler)
        self._file_handler = handler

    def _detach_file(self):
        if self._file_handler is not None:
            self.logger.removeHandler(self._file_handler)
            # Closing matters on Windows: an open handle blocks the rename in
            # rebind() and blocks tmp_path cleanup in tests.
            self._file_handler.close()
            self._file_handler = None

    def rebind(self, base):
        """Adopt `base` as the run identity and move the log to <base>.log.

        Renames the startup log so the early-phase records (config, lyrics,
        singing) are preserved rather than orphaned in a separate file. If the
        rename cannot happen (target exists — e.g. a resumed run — or the file
        is locked) logging continues without loss: the existing records are
        appended into the destination where possible, else the current file is
        simply kept.
        """
        if self._closed or not base:
            return self.path
        target = os.path.join(self.out_dir, f"{base}.log")
        self.base = base
        if os.path.abspath(target) == os.path.abspath(self.path):
            return self.path

        old = self.path
        self._detach_file()
        try:
            if os.path.exists(target):
                # Resumed run: fold the startup records into the existing log
                # rather than clobbering the earlier attempt's history.
                with open(old, "r", encoding="utf-8") as src:
                    carried = src.read()
                if carried:
                    with open(target, "a", encoding="utf-8") as dst:
                        dst.write(carried)
                os.remove(old)
            else:
                os.replace(old, target)
            self.path = target
        except OSError:
            self.path = old  # keep logging where we already are
        self._attach_file(self.path)
        self.logger.info("Run log bound to base %s (%s)", base, self.path)
        return self.path

    def close(self):
        self._detach_file()
        for h in list(self.logger.handlers):
            self.logger.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass
        self._closed = True

    # ---------------------------------------------------------------- write ---
    def info(self, msg, *a):
        self.logger.info(msg, *a)

    def warning(self, msg, *a):
        self.logger.warning(msg, *a)

    def error(self, msg, *a):
        self.logger.error(msg, *a)

    def debug(self, msg, *a):
        self.logger.debug(msg, *a)

    def stage(self, name, **fields):
        """Record a stage transition — the backbone of a diagnosable run."""
        extra = " ".join(f"{k}={v}" for k, v in fields.items() if v is not None)
        self.logger.info("STAGE %s%s", name, f" {extra}" if extra else "")

    def exception(self, msg, exc=None):
        """Log `msg` with a full traceback into the run log AND the error log.

        The same exception often passes several nested guards on its way out
        (the render loop, generate_kidsong, the CLI, the excepthook); it is
        recorded once so kidsong_errors.log stays one line per real failure.
        """
        exc = exc or sys.exc_info()[1]
        if exc is not None:
            if id(exc) in self._reported:
                return None
            self._reported.add(id(exc))
        tb = _format_exc(exc)
        self.logger.error("%s\n%s", msg, tb)
        append_error_log(self.out_dir, self.base, msg, tb)
        return tb

    # ----------------------------------------------------------- fingerprint ---
    def fingerprint(self, cfg=None, comfy_client=None, autostarted=None, extra=None):
        """Log the environment once at start.

        Most failures so far have been environment drift (a moved ACE venv, a
        ComfyUI that was already up with someone else's weights resident), so
        this is recorded even when the run succeeds.
        """
        ks = (cfg or {}).get("kidsong", {}) or {}
        comfy_cfg = (cfg or {}).get("comfy", {}) or {}
        lines = [
            f"python       = {sys.version.split()[0]} ({sys.executable})",
            f"platform     = {platform.platform()}",
            f"pid          = {os.getpid()}",
            f"cwd          = {os.getcwd()}",
            f"output_dir   = {self.out_dir}",
        ]
        url = getattr(comfy_client, "url", None) or comfy_cfg.get("url")
        if url:
            lines.append(f"comfy_url    = {url}")
        if autostarted is not None:
            lines.append(f"comfy_started_by_us = {bool(autostarted)}")
        if comfy_cfg:
            lines.append(
                "comfy_cfg    = "
                + _brief({k: v for k, v in comfy_cfg.items() if not k.startswith("_")})
            )
        if ks:
            lines.append(
                "kidsong_cfg  = "
                + _brief({k: v for k, v in ks.items() if not k.startswith("_")})
            )
            ace = ks.get("ace_python")
            if ace:
                lines.append(f"ace_python   = {ace} (exists={os.path.exists(str(ace))})")
        for k, v in (extra or {}).items():
            lines.append(f"{k:<12} = {v}")
        self.logger.info("Environment fingerprint:\n  " + "\n  ".join(lines))


def _brief(obj, limit=600):
    """Compact single-line repr of a config block, truncated for readability."""
    try:
        import json

        s = json.dumps(obj, default=str, sort_keys=True)
    except (TypeError, ValueError):
        s = repr(obj)
    return s if len(s) <= limit else s[:limit] + "…(truncated)"


def _format_exc(exc):
    if exc is None:
        return "(no exception object available)"
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).rstrip()


def append_error_log(out_dir, base, msg, tb):
    """Append one failure record to output/kidsong_errors.log.

    Append-only and never raises: a logging failure must not mask the original
    error we are trying to report.
    """
    path = os.path.join(str(out_dir), ERROR_LOG_NAME)
    stamp = time.strftime(_DATE_FORMAT)
    block = (
        f"\n{'=' * 78}\n"
        f"{stamp}  base={base or '(unknown)'}  pid={os.getpid()}\n"
        f"{msg}\n"
        f"{tb}\n"
    )
    try:
        os.makedirs(str(out_dir), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(block)
    except OSError:
        pass
    return path


def install_excepthook(runlog):
    """Route any otherwise-uncaught exception into `runlog` before exit.

    Belt-and-braces alongside the try/except around the run itself: this also
    catches failures raised from atexit handlers, background threads' hooks and
    the CLI plumbing outside the guarded block.
    """
    previous = sys.excepthook

    def _hook(exc_type, exc, tb):
        if not issubclass(exc_type, (KeyboardInterrupt, SystemExit)):
            try:
                runlog.exception(f"UNCAUGHT {exc_type.__name__}", exc)
            except Exception:
                pass
        previous(exc_type, exc, tb)

    sys.excepthook = _hook
    return previous
