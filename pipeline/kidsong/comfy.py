"""
kidsong.comfy — Drive a local ComfyUI instance over its HTTP API to render
LTX-2.3 video shots for the children's-song pipeline.

We keep ComfyUI as a separate process (it lives in its own venv with the heavy
torch/CUDA build) and talk to it headless over ``/prompt`` + ``/history``. This
module therefore imports *no* torch — only ``requests`` (a base dependency) and
the stdlib. The finicky part (the LTX-2.3 node graphs) lives as pre-baked
API-format JSON under ``workflows/`` at the repo root; here we only load one,
splice in a handful of per-shot values (prompt, seed, size, output name), POST
it, and pull back the finished MP4.

Graphs are patched *by node title*: every knob we expose carries an exact
``_meta.title`` (PROMPT, NEGATIVE, SEED, WIDTH, HEIGHT, FRAMES, LORA_STYLE,
FILENAME_PREFIX, INPUT_IMAGE, FIRST_IMAGE, LAST_IMAGE, GUIDE_FIRST,
GUIDE_LAST). A patch value may be a plain scalar (applied to that title's
primary input) or a ``{input_name: value}`` dict for finer control.

Config (all optional) lives under ``cfg["comfy"]``::

    url             http://127.0.0.1:8188   base URL of the ComfyUI server
    path            .../ComfyUI             cwd used when we autostart it
    python          .../ComfyUI/venv/...    interpreter used to launch main.py
    autostart       true                    start the server if it is down
    startup_timeout 180                     seconds to wait for it to come up
    render_timeout  900                     seconds to wait for one render
"""
import json
import logging
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse

import requests

log = logging.getLogger("kidsong.comfy")

# Repo root: pipeline/kidsong/comfy.py -> parents[2].
_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOWS = _ROOT / "workflows"

# Default ComfyUI connection settings; overridden by cfg["comfy"].
_DEFAULTS = {
    "url": "http://127.0.0.1:8188",
    "path": None,
    "python": None,
    "autostart": True,
    "startup_timeout": 180,
    # Measured real shot renders: 100-214s (median ~120s). 600s is ~3x the
    # slowest observed shot -- generous, but no longer the old 1800s, which
    # let a wedged server hang a whole episode for 30 minutes on one shot
    # before Contract 1's infra-retry loop existed to recover from it.
    "render_timeout": 600,
}

# How long _poll_history tolerates CONSECUTIVE connection failures before
# giving up on the server rather than polling out the full render_timeout
# (Contract 2). A reachable server that simply hasn't finished the render
# never trips this -- only a dead/wedged one that stops answering at all.
_UNREACHABLE_THRESHOLD_SECONDS = 30


class ComfyUnreachableError(RuntimeError):
    """The ComfyUI server stopped answering HTTP requests entirely.

    Raised by ``_poll_history`` when the server has not responded for
    ``_UNREACHABLE_THRESHOLD_SECONDS`` straight, instead of waiting out the
    full ``render_timeout`` against a server that is simply gone. Distinct
    from a plain ``TimeoutError`` (server alive, render just slow/stuck) so
    callers can trigger ``restart_if_hung()`` specifically on this case.
    """

# For scalar patches (``{title: value}`` where value is not a dict), which input
# on the titled node the value is written to.
_PRIMARY_INPUT = {
    "PROMPT": "text",
    "NEGATIVE": "text",
    "SEED": "noise_seed",
    "WIDTH": "value",
    "HEIGHT": "value",
    "FRAMES": "value",
    "FILENAME_PREFIX": "filename_prefix",
    "INPUT_IMAGE": "image",
    "LORA_STYLE": "strength_model",
    # ltx23_flf2v_toon (first-last-frame i2v, kidsong.transitions): the two
    # LoadImage nodes and the two LTXVAddGuide strength knobs that bracket a
    # generated shot-to-shot transition.
    "FIRST_IMAGE": "image",
    "LAST_IMAGE": "image",
    "GUIDE_FIRST": "strength",
    "GUIDE_LAST": "strength",
}

_VIDEO_EXTS = (".mp4", ".webm", ".mov", ".mkv", ".gif", ".m4v")


class ComfyClient:
    """Thin HTTP client for a local ComfyUI server."""

    def __init__(self, cfg):
        comfy = (cfg or {}).get("comfy", {}) if isinstance(cfg, dict) else {}

        def opt(key):
            return comfy.get(key, _DEFAULTS[key])

        self.url = str(opt("url")).rstrip("/")
        self.path = opt("path")
        self.python = opt("python")
        self.autostart = bool(opt("autostart"))
        self.startup_timeout = int(opt("startup_timeout"))
        self.render_timeout = int(opt("render_timeout"))

        parsed = urlparse(self.url)
        self.host = parsed.hostname or "127.0.0.1"
        self.port = parsed.port or 8188

        # Output dir for launch logs (and a fallback for local file copies).
        self._output_dir = str((cfg.get("_root", str(_ROOT)) if isinstance(cfg, dict)
                                else str(_ROOT)))
        self._object_info = None      # cached /object_info response
        self._proc = None             # server process, only if we started it
        self._staged_inputs = []      # bare filenames staged into ComfyUI/input/

    # -------------------------------------------------------------- server ---
    def _alive(self):
        try:
            r = requests.get(self.url + "/system_stats", timeout=5)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def _port_in_use(self):
        """True if something is bound to (host, port) right now, whether or not
        it answers 200. Used to tell a WEDGED pre-existing server (holds the
        port but /system_stats is not 200 -- e.g. a CUDA-wedged ComfyUI
        returning HTTP 500) apart from a genuinely cold port. Best-effort:
        any error means 'assume free' so a cold start is never blocked."""
        import socket

        host = self.host if self.host not in ("0.0.0.0", "") else "127.0.0.1"
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1.0)
                return s.connect_ex((host, int(self.port))) == 0
        except OSError:
            return False

    def ensure_up(self):
        """Return once the server answers; autostart it first if configured."""
        if self._alive():
            return
        if not self.autostart:
            raise RuntimeError(
                f"ComfyUI is not responding at {self.url} and autostart is off. "
                f"Start it manually or set cfg['comfy']['autostart'] = true."
            )
        if not self.path:
            raise RuntimeError(
                "Cannot autostart ComfyUI: cfg['comfy']['path'] is not set."
            )
        # We are not alive but something may still be holding the port -- a
        # WEDGED pre-existing server whose CUDA context died (/system_stats
        # returns HTTP 500, so _alive() is False, yet it keeps the port bound).
        # Launching over it fails with "port already in use" -> exit code 1,
        # which fails the whole job (a measured incident: a wedged CUDA context
        # took down every queued render). Clear the wedged owner first so the
        # fresh server can bind. On a genuinely cold port this is a no-op.
        if self._port_in_use():
            log.warning(
                "ensure_up: ComfyUI at %s is not answering 200 but the port is "
                "occupied -- clearing the wedged server before autostart.", self.url,
            )
            if self._proc is not None:
                self.stop()
            else:
                self._kill_port_owner()
        python = self.python or "python"
        log_dir = os.path.join(self._output_dir, "output", "comfy_logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "comfyui_server.log")
        self._log_file = open(log_path, "ab")
        cmd = [
            python, "main.py",
            "--listen", self.host,
            "--port", str(self.port),
            "--disable-auto-launch",
        ]
        self._proc = subprocess.Popen(
            cmd, cwd=self.path, stdout=self._log_file, stderr=subprocess.STDOUT,
        )

        deadline = time.time() + self.startup_timeout
        while time.time() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"ComfyUI exited during startup (code {self._proc.returncode}); "
                    f"see {log_path}."
                )
            if self._alive():
                return
            time.sleep(2)
        raise RuntimeError(
            f"ComfyUI did not come up within {self.startup_timeout}s "
            f"(launched from {self.path}; see {log_path})."
        )

    # ------------------------------------------------------------ validate ---
    def _fetch_object_info(self):
        if self._object_info is None:
            r = requests.get(self.url + "/object_info", timeout=60)
            r.raise_for_status()
            self._object_info = r.json()
        return self._object_info

    def validate(self, workflow):
        """Raise if the graph references a class_type the server doesn't know."""
        known = self._fetch_object_info()
        missing = sorted({
            n["class_type"] for n in workflow.values()
            if n.get("class_type") not in known
        })
        if missing:
            raise RuntimeError(
                "ComfyUI is missing node classes required by this workflow: "
                + ", ".join(missing)
                + ". Install the LTX-2.3 / LTXVideo custom nodes and update ComfyUI."
            )
        return True

    # -------------------------------------------------------------- render ---
    def load_workflow(self, workflow_name):
        """Load ``workflows/<name>.json`` (name may include the .json suffix)."""
        name = workflow_name
        if name.endswith(".json"):
            name = name[:-5]
        path = _WORKFLOWS / (name + ".json")
        if not path.exists():
            raise FileNotFoundError(f"Workflow not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _apply_patches(workflow, patches):
        for title, value in (patches or {}).items():
            matched = False
            for node in workflow.values():
                if node.get("_meta", {}).get("title") != title:
                    continue
                matched = True
                inputs = node.setdefault("inputs", {})
                if isinstance(value, dict):
                    for k, v in value.items():
                        inputs[k] = v
                else:
                    key = _PRIMARY_INPUT.get(title)
                    if key is None:
                        raise KeyError(
                            f"No primary input mapping for title {title!r}; "
                            f"pass a {{input_name: value}} dict instead."
                        )
                    # A style-LoRA scalar string means "swap the lora file".
                    if title == "LORA_STYLE" and isinstance(value, str):
                        key = "lora_name"
                    inputs[key] = value
            if not matched:
                raise KeyError(
                    f"No node titled {title!r} in this workflow; "
                    f"cannot apply patch."
                )
        return workflow

    def render(self, workflow_name, patches, out_path):
        """Render ``workflow_name`` with ``patches`` and write the MP4 to out_path.

        ``patches`` maps a node ``_meta.title`` to either a scalar (written to
        that title's primary input) or a ``{input_name: value}`` dict.
        Returns ``out_path``.
        """
        workflow = self.load_workflow(workflow_name)
        self.validate(workflow)
        self._apply_patches(workflow, patches)

        client_id = str(uuid.uuid4())
        r = requests.post(
            self.url + "/prompt",
            json={"prompt": workflow, "client_id": client_id},
            timeout=30,
        )
        if r.status_code != 200:
            raise RuntimeError(
                f"ComfyUI rejected the prompt (HTTP {r.status_code}): {r.text}"
            )
        prompt_id = r.json().get("prompt_id")
        if not prompt_id:
            raise RuntimeError(f"ComfyUI returned no prompt_id: {r.text}")

        history = self._poll_history(prompt_id)
        self._raise_on_node_error(history, prompt_id)
        file_info = self._find_output_file(history)
        if file_info is None:
            raise RuntimeError(
                f"Render finished but no output file was reported for prompt "
                f"{prompt_id}."
            )
        return self._retrieve(file_info, out_path)

    def _poll_history(self, prompt_id):
        deadline = time.time() + self.render_timeout
        url = self.url + "/history/" + prompt_id
        # Tracks the start of the CURRENT unbroken run of connection failures;
        # None means the last request either succeeded or hasn't been tried
        # yet. Any successful response (even a non-200 or an empty entry --
        # the server answered, it just has nothing to report yet) resets this,
        # so only a truly dead server (not "alive but still rendering") trips
        # ComfyUnreachableError.
        unreachable_since = None
        while time.time() < deadline:
            try:
                r = requests.get(url, timeout=15)
                unreachable_since = None
                if r.status_code == 200:
                    data = r.json()
                    entry = data.get(prompt_id)
                    if entry:
                        status = entry.get("status", {})
                        if status.get("completed") or entry.get("outputs"):
                            return entry
                        if status.get("status_str") == "error":
                            return entry
            except requests.RequestException:
                now = time.time()
                if unreachable_since is None:
                    unreachable_since = now
                elif now - unreachable_since > _UNREACHABLE_THRESHOLD_SECONDS:
                    raise ComfyUnreachableError(
                        f"ComfyUI at {self.url} has not answered for over "
                        f"{_UNREACHABLE_THRESHOLD_SECONDS}s (prompt {prompt_id}) -- "
                        "treating the server as dead instead of polling out the "
                        f"full {self.render_timeout}s render_timeout."
                    )
            time.sleep(2)
        raise TimeoutError(
            f"ComfyUI render exceeded {self.render_timeout}s for prompt {prompt_id}."
        )

    @staticmethod
    def _raise_on_node_error(history, prompt_id):
        status = history.get("status", {})
        messages = status.get("messages", []) or []
        errors = []
        for msg in messages:
            # messages look like ["execution_error", {..., "exception_message": ...}]
            if isinstance(msg, list) and len(msg) == 2 and "error" in str(msg[0]):
                detail = msg[1] or {}
                node = detail.get("node_type") or detail.get("node_id") or "?"
                text = detail.get("exception_message") or detail.get("exception_type") \
                    or str(detail)
                errors.append(f"[{node}] {text}")
        if status.get("status_str") == "error" and not errors:
            errors.append(json.dumps(status))
        if errors:
            raise RuntimeError(
                f"ComfyUI reported a node error for prompt {prompt_id}: "
                + " | ".join(errors)
            )

    @staticmethod
    def _find_output_file(history):
        """Pull the first video-like output entry from a history record."""
        outputs = history.get("outputs", {}) or {}
        candidates = []
        for node_out in outputs.values():
            if not isinstance(node_out, dict):
                continue
            for value in node_out.values():
                if not isinstance(value, list):
                    continue
                for item in value:
                    if isinstance(item, dict) and item.get("filename"):
                        candidates.append(item)
        if not candidates:
            return None
        for item in candidates:
            if str(item.get("filename", "")).lower().endswith(_VIDEO_EXTS):
                return item
        return candidates[0]

    def _retrieve(self, file_info, out_path):
        filename = file_info["filename"]
        subfolder = file_info.get("subfolder", "") or ""
        ftype = file_info.get("type", "output") or "output"
        out_path = os.path.abspath(out_path)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

        # Both paths publish atomically: `out_path` is the RENDERED TAKE, and a
        # truncated mp4 there goes straight to the contact sheet, the reviewer
        # and the editor, which cannot tell it from a complete one. See
        # pipeline/atomicio.py.
        from pipeline.atomicio import atomic_path, atomic_write_bytes

        # Prefer a local copy when the server shares this filesystem.
        if self.path:
            local = os.path.join(self.path, ftype, subfolder, filename)
            if os.path.isfile(local):
                with atomic_path(out_path) as tmp:
                    shutil.copyfile(local, tmp)
                return out_path

        r = requests.get(
            self.url + "/view",
            params={"filename": filename, "subfolder": subfolder, "type": ftype},
            stream=True,
            timeout=120,
        )
        r.raise_for_status()
        atomic_write_bytes(out_path, r.iter_content(chunk_size=1 << 16))
        return out_path

    # ---------------------------------------------------------------- input ---
    def stage_input_image(self, src_path):
        """Copy `src_path` into ComfyUI's input/ dir under a unique name and
        return the bare filename to use as a LoadImage.image value.

        FLF2V/i2v graphs read conditioning frames from ComfyUI's own input
        directory (not an arbitrary path), so any PNG we hand the LoadImage
        node first has to land there. Requires `cfg["comfy"]["path"]` (the
        ComfyUI install dir) to be configured — there is no HTTP upload path
        used here, only a same-filesystem copy.
        """
        if not self.path:
            raise RuntimeError(
                "Cannot stage an input image: cfg['comfy']['path'] is not set. "
                "FLF2V/i2v workflows read conditioning frames from ComfyUI's "
                "own input/ directory, so the ComfyUI install path must be "
                "configured for this to work."
            )
        input_dir = os.path.join(self.path, "input")
        os.makedirs(input_dir, exist_ok=True)
        ext = os.path.splitext(src_path)[1] or ".png"
        dest_name = f"kidsong_{uuid.uuid4().hex[:12]}{ext}"
        dest_path = os.path.join(input_dir, dest_name)
        shutil.copyfile(src_path, dest_path)
        # Track it so cleanup_staged()/free() can remove it — otherwise every
        # staged conditioning frame (i2v reference anchor, FLF2V guides) leaks a
        # copy into ComfyUI/input/ forever.
        self._staged_inputs.append(dest_name)
        return dest_name

    def cleanup_staged(self):
        """Delete every input file this client staged via ``stage_input_image``.

        Best-effort and idempotent: a missing file (or no configured ComfyUI
        path) is ignored, and the tracking list is always cleared so a second
        call is a no-op. Called from ``free()`` (and thus from the render loop's
        ``finally``), so staged conditioning frames don't accumulate across runs.
        """
        staged, self._staged_inputs = self._staged_inputs, []
        if not self.path:
            return
        input_dir = os.path.join(self.path, "input")
        for name in staged:
            try:
                os.remove(os.path.join(input_dir, name))
            except OSError:
                pass

    # ------------------------------------------------------------- cleanup ---
    def free(self):
        """Ask ComfyUI to unload models and free memory; ignore failures.

        Also cleans up any input files this client staged (see
        ``cleanup_staged``) so a run never leaks conditioning frames into
        ComfyUI/input/.
        """
        self.cleanup_staged()
        try:
            requests.post(
                self.url + "/free",
                json={"unload_models": True, "free_memory": True},
                timeout=15,
            )
        except requests.RequestException:
            pass

    def stop(self):
        """Terminate the server only if we are the ones who started it."""
        if self._proc is None:
            return
        try:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        except Exception:
            pass
        finally:
            self._proc = None
            log = getattr(self, "_log_file", None)
            if log is not None:
                try:
                    log.close()
                except Exception:
                    pass

    # ------------------------------------------------------------- self-heal ---
    def restart_if_hung(self):
        """Self-heal after a ``ComfyUnreachableError``: kill whatever process
        owns ``self.port`` and re-run the ``ensure_up`` autostart path.

        This mirrors a measured incident: a run that reused a wedged
        pre-existing server hung for 1800s on one shot, while a fresh server
        rendered the same shot in ~200s. Retrying against the SAME dead
        process just repeats the hang, so this forces a new one.

        Works whether or not this client started the server itself: if we
        did (``self._proc`` set), ``stop()`` handles it; otherwise it hunts
        down and kills whatever is bound to the port. Requires
        ``cfg["comfy"]["path"]`` (the same field ``ensure_up`` needs to
        autostart) -- without it we have no way to relaunch, so this logs a
        clear warning and returns False rather than raising.
        """
        if not self.path:
            log.warning(
                "restart_if_hung: cfg['comfy']['path'] is not set, so this "
                "client cannot manage the ComfyUI process at %s. A wedged "
                "server will keep hanging until it is restarted manually.",
                self.url,
            )
            return False

        log.warning(
            "restart_if_hung: ComfyUI at %s looks dead -- restarting it.",
            self.url,
        )
        if self._proc is not None:
            self.stop()
        else:
            self._kill_port_owner()
        self.ensure_up()
        return True

    def _kill_port_owner(self):
        """Best-effort: terminate whatever OS process is bound to self.port.

        Only needed when we did NOT start the server ourselves (a wedged
        pre-existing server, ``self._proc is None``) -- ``stop()`` handles
        the case where we did. Tries ``psutil`` first (present in the
        project venv); falls back to parsing ``netstat -ano`` on Windows when
        psutil isn't importable. Silent/best-effort: if nothing is found or
        the kill fails, ``ensure_up()`` will surface its own clear error when
        the port is still occupied.
        """
        psutil = None
        try:
            import psutil as _psutil

            psutil = _psutil
        except ImportError:
            pass

        pids = set()
        if psutil is not None:
            try:
                for conn in psutil.net_connections(kind="inet"):
                    if conn.laddr and conn.laddr.port == self.port and conn.pid:
                        pids.add(conn.pid)
            except Exception as exc:
                log.warning(
                    "restart_if_hung: psutil could not enumerate connections "
                    "(%s); falling back to netstat.", exc,
                )
                pids = self._pids_from_netstat()
        else:
            pids = self._pids_from_netstat()

        if not pids:
            log.warning(
                "restart_if_hung: no process found bound to port %s.", self.port,
            )
            return

        for pid in pids:
            log.warning(
                "restart_if_hung: terminating pid %s bound to port %s.",
                pid, self.port,
            )
            killed = False
            if psutil is not None:
                try:
                    psutil.Process(pid).terminate()
                    killed = True
                except Exception:
                    killed = False
            if not killed:
                try:
                    subprocess.run(
                        ["taskkill", "/PID", str(pid), "/F"],
                        capture_output=True, timeout=10,
                    )
                except Exception:
                    pass
        time.sleep(2)  # give the OS a moment to release the port

    def _pids_from_netstat(self):
        """Windows fallback for _kill_port_owner when psutil is unavailable."""
        pids = set()
        try:
            out = subprocess.run(
                ["netstat", "-ano"], capture_output=True, text=True, timeout=15,
            ).stdout
        except Exception:
            return pids
        needle = f":{self.port}"
        for line in out.splitlines():
            parts = line.split()
            if len(parts) >= 5 and parts[0].upper().startswith("TCP") and needle in parts[1]:
                try:
                    pids.add(int(parts[-1]))
                except ValueError:
                    pass
        return pids


# --------------------------------------------------------------------- main ---
if __name__ == "__main__":
    import argparse
    import sys

    # Allow running directly (python pipeline/kidsong/comfy.py) as well as -m.
    sys.path.insert(0, str(_ROOT))
    from pipeline.config import load_config

    parser = argparse.ArgumentParser(description="Drive local ComfyUI for LTX-2.3.")
    parser.add_argument("--smoke", action="store_true",
                        help="ensure server up, validate + render the t2v workflow.")
    parser.add_argument("--validate-only", action="store_true",
                        help="ensure server up and validate every kidsong workflow.")
    args = parser.parse_args()

    cfg = load_config()
    client = ComfyClient(cfg)

    if args.validate_only:
        client.ensure_up()
        for name in (
            "ltx23_t2v_toon", "ltx23_t2v_toon_hires", "ltx23_i2v_toon",
            "ltx23_i2v_toon_hires", "ltx23_flf2v_toon",
        ):
            try:
                client.validate(client.load_workflow(name))
            except FileNotFoundError:
                print(f"SKIP {name} (not present)")
                continue
            print(f"OK  {name}")
    elif args.smoke:
        client.ensure_up()
        client.validate(client.load_workflow("ltx23_t2v_toon"))
        out = os.path.join(cfg["_root"], "output", "kidsong_comfy_smoke", "smoke.mp4")
        patches = {
            "PROMPT": "P1x4r, 3D CGI toon style, a cheerful toddler waving hello "
                      "in a sunny green backyard with colorful balloons",
            "SEED": 42,
            "FRAMES": 49,
        }
        start = time.time()
        path = client.render("ltx23_t2v_toon", patches, out)
        print(f"Rendered {path} in {time.time() - start:.1f}s")
    else:
        parser.error("choose one of --smoke or --validate-only")
