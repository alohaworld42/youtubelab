"""pipeline.gpu_lock — a small cross-process mutex for the one shared GPU.

Two independent entry points can start GPU work at the same time on this
machine: the studio scheduler's background thread (studio/scheduler.py,
in-process with app.py) and a `python -m pipeline.kidsong.generate` CLI
invocation kicked off from a batch script (output/batch_queue.ps1). Nothing
in-process stops that collision — they're different OS processes — and it
has already cost a render (ComfyUI timed out mid-job and the video was
lost).

`GPULock` is a lockfile under the output directory holding the holder's pid,
start time, and a random per-acquire nonce. A crashed/killed holder leaves
the file behind; `try_acquire` notices the recorded pid is no longer running
(STALE) and steals the lock instead of deadlocking forever.

No third-party dependency: liveness is checked with `OpenProcess` (ctypes)
on Windows and `os.kill(pid, 0)` elsewhere.

What is actually guaranteed
---------------------------
1. *Publication is atomic.* The lockfile is never observable in a partially
   written state. The payload is written to a temp file in the same
   directory, flushed and fsync'd, and only then moved into place with a
   **no-clobber** move (`os.rename` on Windows, which raises FileExistsError
   if the destination exists; `os.link` + unlink on POSIX, where plain
   `os.rename` would silently overwrite). Both are atomic on the target
   filesystem, so a reader sees either no file or the complete ledger — never
   the zero-byte window that the old os.open()-then-write() left behind.

2. *Fresh acquire is mutually exclusive.* The no-clobber move is the single
   linearization point: of N racing processes exactly one move succeeds and
   the rest get FileExistsError.

3. *Stale steal is mutually exclusive and leaves no empty window.* Stealing
   does NOT remove the lockfile — not unconditionally (two processes reading
   the same stale ledger both removed and both created, and both believed
   they held the lock) and not by renaming it away either (that left the path
   momentarily EMPTY, and a fresh acquirer creating a lock in that window
   could then be renamed away and dropped by a straggling stealer — two
   holders again). Instead the stealer claims an *arbitration file* next to
   the lock (atomically published, so exactly one stealer exists at a time),
   re-confirms the ledger is still the stale one it judged, and then
   `os.replace`s its OWN ledger straight over it. `os.replace` is atomic and
   the path is never absent, so no fresh acquirer can slip in: a no-clobber
   create fails for as long as the stale file is there.

4. *Ownership is verified, not assumed.* Every acquire re-reads the lockfile
   and confirms it carries this instance's nonce before reporting success,
   and `release()` re-checks the nonce before removing anything — so an
   instance can never delete a lockfile that belongs to somebody else.

Windows specifics
-----------------
`os.rename` on Windows maps to MoveFileExW *without* MOVEFILE_REPLACE_EXISTING
and fails with FileExistsError if the destination exists — which is exactly
the no-clobber primitive we need, and unlike POSIX rename it cannot silently
clobber a competing holder. Windows also refuses to rename or delete a file
another process holds open; every operation here that can fail that way is
treated as "lost the race" and falls through to a retry or a clean False, so
the failure mode is *not acquiring* rather than double-acquiring.

Why the takeover is in place, not remove-then-create
----------------------------------------------------
Every design here that lets the lock path be *absent* for even an instant has
produced two simultaneous holders, and tests/test_gpu_lock.py's 8-thread and
4-subprocess stale races reproduce it within a hundred repetitions. The
sequence that killed the rename-away version: A renames the stale file to a
side path; B creates a fresh lock in the now-empty slot and rightly declares
itself the holder; C — still holding a read of the old stale ledger — renames
B's fresh lock away, cannot restore it because D has created another, and
drops it. B and D both hold the GPU.

The in-place takeover has no such window, and the arbitration file that
serialises stealers is published atomically for the same reason: an arbiter
created empty and filled in afterwards was read as "abandoned" by a second
claimant, which then removed and re-claimed it — two stealers, and two
stealers is exactly the condition the in-place takeover assumes away.

Note also that only genuine races consume retry attempts — finding a live
holder returns False immediately. Losing a steal race re-drives the loop
rather than giving up, so contention degrades into waiting, never into a
spurious "the GPU is busy" when it is in fact free.
"""
import json
import logging
import os
import sys
import time
import uuid

log = logging.getLogger("pipeline.gpu_lock")

DEFAULT_LOCK_NAME = "gpu_render.lock"

# How many times try_acquire() re-drives its create/inspect/steal loop before
# giving up. Only *races* consume attempts — finding a live holder returns
# immediately — so this bounds contention churn, not waiting. Eight racers on
# one stale lock can burn several passes before a winner settles.
_ACQUIRE_ATTEMPTS = 12


def pid_alive(pid):
    """Best-effort liveness check for a pid recorded by another process.

    Used only to decide whether a leftover lockfile is stale (holder crashed/
    was killed) or genuinely still rendering — false positives are safe (we'd
    just wait a bit longer / a re-check next tick will re-evaluate), so this
    deliberately errs toward "alive" when the check itself is inconclusive.
    """
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION, False, pid
        )
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, just owned by someone else
    except OSError:
        return False
    return True


def _move_no_clobber(src, dst):
    """Atomically move `src` -> `dst`, raising FileExistsError if `dst` exists.

    This is the primitive the whole module rests on. Windows `os.rename`
    already refuses to overwrite; POSIX `os.rename` would silently clobber a
    competing holder, so there we hardlink-then-unlink instead (`os.link`
    fails with FileExistsError if the destination exists, atomically).
    """
    if sys.platform == "win32":
        os.rename(src, dst)
        return
    os.link(src, dst)
    os.unlink(src)


def _identity(info):
    """The comparable identity of a lock ledger — used to confirm that the
    file we renamed away is the same one we inspected and judged stale."""
    if not isinstance(info, dict):
        return None
    return (info.get("pid"), info.get("started_at"), info.get("nonce"))


class GPULock:
    """Cross-process mutex backed by a lockfile at `path`.

    Not re-entrant. It exists primarily to keep two separate OS *processes*
    off the GPU at once, but the acquire path is safe against threads racing
    on the same path too (the no-clobber move and the steal-rename are the
    only decision points, and both are single-winner).
    """

    def __init__(self, path):
        self.path = path
        self._held = False
        self._nonce = None

    # ------------------------------------------------------------ internals --
    def _read(self, path=None):
        try:
            with open(path or self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    def _create_exclusive(self):
        """Publish a complete ledger at `self.path`, atomically.

        Returns True if this call created the lockfile, False if a lockfile
        was already there. Never leaves a partial file at `self.path`.
        """
        d = os.path.dirname(self.path) or "."
        os.makedirs(d, exist_ok=True)
        nonce = uuid.uuid4().hex
        payload = {
            "pid": os.getpid(),
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "nonce": nonce,
        }
        tmp = os.path.join(d, f".{os.path.basename(self.path)}.{os.getpid()}.{nonce}.tmp")
        fd = os.open(tmp, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f)
                f.flush()
                os.fsync(f.fileno())
            try:
                _move_no_clobber(tmp, self.path)
            except FileExistsError:
                return False
            except OSError:
                # Windows: destination held open by another process, or the
                # move lost to a competitor. Fail closed — treat as taken.
                return False
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass  # already moved into place, or never created
        self._nonce = nonce
        return True

    def _owns(self):
        """True only if the lockfile on disk currently carries our nonce."""
        if not self._nonce:
            return False
        info = self._read()
        return bool(info) and info.get("nonce") == self._nonce

    def _arbiter_path(self):
        return f"{self.path}.steal"

    def _claim_arbiter(self):
        """Become the single in-flight stealer, or return False.

        The arbiter is PUBLISHED ATOMICALLY (write a temp file, then no-clobber
        move), for the same reason the lock itself is. An earlier version here
        used `os.open(O_CREAT|O_EXCL)` and wrote the pid afterwards, which left
        the file momentarily EMPTY — a second claimant then read no pid, judged
        the arbiter abandoned, removed it and claimed it too. Two stealers, and
        with two stealers the in-place takeover loses its only guarantee.

        A stealer that dies mid-takeover would block every future steal
        forever, so an arbiter naming a pid that is definitively not running is
        cleared and the claim retried once. An arbiter we cannot parse is
        treated as held (fail closed) — publication is atomic, so it is never a
        half-written file.
        """
        arb = self._arbiter_path()
        d = os.path.dirname(self.path) or "."
        for attempt in (0, 1):
            tmp = os.path.join(
                d, f".{os.path.basename(arb)}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
            )
            try:
                fd = os.open(tmp, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                try:
                    os.write(fd, str(os.getpid()).encode("ascii"))
                finally:
                    os.close(fd)
                _move_no_clobber(tmp, arb)
                return True
            except FileExistsError:
                pass
            except OSError:
                return False
            finally:
                try:
                    os.remove(tmp)
                except OSError:
                    pass

            if attempt:
                return False
            holder = None
            try:
                with open(arb, "r", encoding="utf-8") as f:
                    holder = int((f.read() or "").strip())
            except (OSError, ValueError):
                return False  # unreadable/corrupt — assume a live stealer
            if pid_alive(holder):
                return False  # a live stealer is mid-takeover
            try:
                os.remove(arb)  # abandoned by a dead stealer
            except OSError:
                return False
        return False

    def _steal_stale(self, seen):
        """Take over the stale lock `seen` IN PLACE. Returns True if we now
        hold the lock (the caller does not need to re-create it).

        The lock path is never absent during a takeover, and that is the whole
        point. The previous implementation renamed the stale file away to a
        side path and let the caller re-create it; between the rename and the
        re-create the path was EMPTY, so a third contender could create a fresh
        lock there — and a fourth stealer, still holding a read of the old
        stale ledger, could then rename that fresh lock away and, failing to
        restore it, drop it. Two processes then believed they held the GPU
        (reproduced by tests/test_gpu_lock.py's 8-thread stale race, which
        failed roughly one run in three).

        Here instead: claim the arbiter, re-confirm the ledger is still the
        stale one we judged, then `os.replace` OUR ledger straight over it.
        `os.replace` is atomic and leaves no empty window, and a fresh acquirer
        cannot get in because `_create_exclusive`'s no-clobber move fails for
        as long as the stale file is there. Only one stealer exists at a time,
        so nothing can rename our fresh lock away either.
        """
        if _identity(self._read()) != _identity(seen):
            return False
        if not self._claim_arbiter():
            return False
        try:
            current = self._read()
            if _identity(current) != _identity(seen):
                return False  # another stealer already took it over
            if current and current.get("pid") and pid_alive(current.get("pid")):
                return False  # not stale after all — the holder is running

            d = os.path.dirname(self.path) or "."
            nonce = uuid.uuid4().hex
            payload = {
                "pid": os.getpid(),
                "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "nonce": nonce,
            }
            tmp = os.path.join(
                d, f".{os.path.basename(self.path)}.{os.getpid()}.{nonce}.steal"
            )
            try:
                fd = os.open(tmp, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(payload, f)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp, self.path)
            except OSError:
                # Windows can refuse the replace if another process holds the
                # target open. Fail closed: we simply did not get the lock.
                try:
                    os.remove(tmp)
                except OSError:
                    pass
                return False

            self._nonce = nonce
            log.warning(
                "gpu lock at %s was stale (pid %s not running) — took it over",
                self.path, seen.get("pid") if seen else None,
            )
            return True
        finally:
            try:
                os.remove(self._arbiter_path())
            except OSError:
                pass

    # --------------------------------------------------------------- public --
    def try_acquire(self):
        """Non-blocking. Returns True if this process now holds the lock
        (fresh acquire, or it stole a stale one from a dead pid), False if a
        live process genuinely holds it right now.
        """
        if self._held:
            return True
        for _ in range(_ACQUIRE_ATTEMPTS):
            if self._create_exclusive():
                # Confirm nobody renamed our brand-new lock away underneath us
                # before we call ourselves the holder.
                if self._owns():
                    self._held = True
                    return True
                continue
            info = self._read()
            if info and info.get("nonce") and info.get("nonce") == self._nonce:
                # Our own ledger from a previous attempt in this loop.
                self._held = True
                return True
            if info is None and not os.path.exists(self.path):
                # The holder released (or a competing stealer moved it away)
                # between our failed create and our read — just try again.
                continue
            holder = info.get("pid") if info else None
            if holder and pid_alive(holder):
                # Someone is genuinely rendering right now. This is the only
                # clean "no" — everything else is a race we should re-drive.
                return False
            # Stale (dead pid) or unparseable. Publication is atomic, so an
            # unparseable file is genuinely corrupt, not one mid-creation.
            # Losing the steal race is NOT a reason to give up: the winner may
            # release, or we may find a live holder, on the next pass.
            if self._steal_stale(info) and self._owns():
                self._held = True
                return True
        return False

    def acquire(self, timeout=None, poll_seconds=2.0, on_wait=None):
        """Blocking acquire. `timeout=None` waits forever; otherwise raises
        `TimeoutError` once `timeout` seconds have elapsed without acquiring.

        `on_wait(holder_pid)`, if given, is called each time the lock is
        found held and we're about to sleep — lets a caller print/log a
        "waiting for the GPU" message without spamming on every poll.
        """
        deadline = None if timeout is None else time.time() + timeout
        while True:
            if self.try_acquire():
                return
            if deadline is not None and time.time() >= deadline:
                info = self._read()
                holder = info.get("pid") if info else "unknown"
                raise TimeoutError(
                    f"timed out after {timeout}s waiting for the GPU lock at {self.path} "
                    f"(held by pid {holder})"
                )
            if on_wait:
                info = self._read()
                on_wait(info.get("pid") if info else None)
            time.sleep(poll_seconds)

    def release(self):
        """Safe to call unconditionally (e.g. from a `finally`), even if this
        process never actually held the lock.

        Removes the lockfile only if it still carries our nonce — releasing
        must never delete a lock some other process now legitimately holds.
        """
        if not self._held:
            return
        if self._owns():
            try:
                os.remove(self.path)
            except OSError:
                pass
        else:
            log.warning(
                "gpu lock at %s is no longer ours at release time — leaving it alone",
                self.path,
            )
        self._held = False
        self._nonce = None

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()
        return False


def get_lock(cfg):
    """Build the GPULock for this config's output directory.

    `cfg` is expected to be a loaded pipeline config (has `_root` and
    `paths.output_dir`, as returned by `pipeline.config.load_config`).
    """
    from pipeline.config import abspath

    lock_cfg = (cfg or {}).get("gpu_lock") or {}
    out_dir = abspath(cfg, cfg["paths"]["output_dir"])
    name = lock_cfg.get("lock_file", DEFAULT_LOCK_NAME)
    return GPULock(os.path.join(out_dir, name))
