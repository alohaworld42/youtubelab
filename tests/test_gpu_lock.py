"""Tests for pipeline.gpu_lock — the cross-process GPU mutex.

No GPU/network needed: everything here exercises the lockfile mechanics
directly (this process's own pid is always "alive", a made-up huge pid
never is).
"""
import json
import os
import subprocess
import sys
import threading
import time

import pytest

from pipeline import gpu_lock
from pipeline.gpu_lock import GPULock


def _lock_path(tmp_path):
    return str(tmp_path / "gpu_render.lock")


# ------------------------------------------------------------------ pid_alive --
def test_pid_alive_true_for_own_pid():
    assert gpu_lock.pid_alive(os.getpid()) is True


def test_pid_alive_false_for_bogus_pid():
    # A pid far outside any real process id range.
    assert gpu_lock.pid_alive(999_999_999) is False


def test_pid_alive_false_for_garbage_input():
    assert gpu_lock.pid_alive(None) is False
    assert gpu_lock.pid_alive("not-a-pid") is False
    assert gpu_lock.pid_alive(0) is False
    assert gpu_lock.pid_alive(-5) is False


# -------------------------------------------------------------- try_acquire --
def test_try_acquire_succeeds_when_free(tmp_path):
    lock = GPULock(_lock_path(tmp_path))
    assert lock.try_acquire() is True
    assert os.path.exists(lock.path)


def test_try_acquire_is_idempotent_for_the_holder(tmp_path):
    lock = GPULock(_lock_path(tmp_path))
    assert lock.try_acquire() is True
    assert lock.try_acquire() is True  # already held by us — no error, no re-lock dance


def test_try_acquire_excludes_a_second_holder(tmp_path):
    path = _lock_path(tmp_path)
    first = GPULock(path)
    second = GPULock(path)

    assert first.try_acquire() is True
    # A second GPULock instance pointed at the same file, held by a live pid
    # (our own — always alive) must NOT be able to acquire it.
    assert second.try_acquire() is False
    assert second._held is False


def test_release_then_second_can_acquire(tmp_path):
    path = _lock_path(tmp_path)
    first = GPULock(path)
    second = GPULock(path)

    assert first.try_acquire() is True
    first.release()
    assert not os.path.exists(path)
    assert second.try_acquire() is True


def test_release_is_safe_when_never_held(tmp_path):
    lock = GPULock(_lock_path(tmp_path))
    lock.release()  # must not raise
    assert not os.path.exists(lock.path)


def test_release_only_removes_file_if_this_instance_holds_it(tmp_path):
    """A GPULock that lost try_acquire() must never delete the winner's
    lockfile out from under it."""
    path = _lock_path(tmp_path)
    winner = GPULock(path)
    loser = GPULock(path)
    assert winner.try_acquire() is True
    assert loser.try_acquire() is False

    loser.release()  # loser never held it — must be a no-op
    assert os.path.exists(path)  # winner's lock survives


# --------------------------------------------------------- stale lock takeover --
def test_stale_lock_from_dead_pid_is_stolen(tmp_path):
    path = _lock_path(tmp_path)
    import json

    with open(path, "w", encoding="utf-8") as f:
        json.dump({"pid": 999_999_999, "started_at": "2020-01-01 00:00:00"}, f)

    lock = GPULock(path)
    assert lock.try_acquire() is True
    # The stolen lock now records OUR pid.
    with open(path, "r", encoding="utf-8") as f:
        info = json.load(f)
    assert info["pid"] == os.getpid()


def test_unreadable_lock_file_is_treated_as_stale(tmp_path):
    path = _lock_path(tmp_path)
    with open(path, "w", encoding="utf-8") as f:
        f.write("{not valid json")

    lock = GPULock(path)
    assert lock.try_acquire() is True


# -------------------------------------------------------- release on exception --
def test_lock_releases_via_finally_on_exception(tmp_path):
    path = _lock_path(tmp_path)
    lock = GPULock(path)
    assert lock.try_acquire() is True

    with pytest.raises(ValueError):
        try:
            raise ValueError("boom mid-render")
        finally:
            lock.release()

    assert not os.path.exists(path)
    # A fresh lock can now be acquired — nothing was left dangling.
    assert GPULock(path).try_acquire() is True


def test_context_manager_releases_on_exception(tmp_path):
    path = _lock_path(tmp_path)
    with pytest.raises(RuntimeError):
        with GPULock(path):
            assert os.path.exists(path)
            raise RuntimeError("boom")
    assert not os.path.exists(path)


# --------------------------------------------------------------- blocking acquire --
def test_acquire_blocks_then_succeeds_after_release(tmp_path):
    path = _lock_path(tmp_path)
    holder = GPULock(path)
    assert holder.try_acquire() is True

    waiter = GPULock(path)
    calls = []

    def _release_soon(holder_pid):
        calls.append(holder_pid)
        if len(calls) == 1:
            holder.release()

    waiter.acquire(timeout=5, poll_seconds=0.05, on_wait=_release_soon)
    assert waiter._held is True
    assert calls  # on_wait was invoked at least once while waiting


def test_acquire_times_out_with_clear_error(tmp_path):
    path = _lock_path(tmp_path)
    holder = GPULock(path)
    assert holder.try_acquire() is True

    waiter = GPULock(path)
    with pytest.raises(TimeoutError, match="GPU lock"):
        waiter.acquire(timeout=0.15, poll_seconds=0.05)


def test_acquire_waits_forever_flag_is_none_by_default(tmp_path):
    """timeout=None must not raise TimeoutError — sanity check the default
    argument shape (actually waiting forever isn't exercised here)."""
    path = _lock_path(tmp_path)
    lock = GPULock(path)
    lock.acquire(timeout=None)  # free lock, returns immediately
    assert lock._held is True


# --------------------------------------------------------------------- get_lock --
def test_get_lock_uses_configured_output_dir_and_name(tmp_path):
    cfg = {
        "_root": str(tmp_path),
        "paths": {"output_dir": "out"},
        "gpu_lock": {"lock_file": "custom.lock"},
    }
    lock = gpu_lock.get_lock(cfg)
    assert lock.path == os.path.join(str(tmp_path), "out", "custom.lock")


def test_get_lock_default_name(tmp_path):
    cfg = {"_root": str(tmp_path), "paths": {"output_dir": "out"}}
    lock = gpu_lock.get_lock(cfg)
    assert os.path.basename(lock.path) == "gpu_render.lock"


# =====================================================================
# Concurrency / contention
# =====================================================================
# These are the tests that were missing when two real races shipped:
#   BUG 1 — the stale-lock steal removed the lockfile unconditionally, so two
#           processes that read the same stale ledger both removed and both
#           created, and BOTH believed they held the lock.
#   BUG 2 — the lockfile was created empty and written afterwards, so a reader
#           could see a zero-byte file, fail to parse it, call it stale, and
#           steal a lock that was being created right then.
# Every test here repeats its race many times rather than trusting one lucky
# interleaving, and synchronises with barriers/events rather than sleeps.

DEAD_PID = 999_999_999
_STALE_LEDGER = {"pid": DEAD_PID, "started_at": "2020-01-01 00:00:00"}


def _write_stale(path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dict(_STALE_LEDGER), f)


def _race_try_acquire(path, n_threads):
    """Release `n_threads` threads onto `try_acquire(path)` simultaneously.

    Returns the list of results. A barrier — not a sleep — is what makes them
    contend, so this is fast and does not depend on scheduler luck for the
    threads to be in flight at the same time.
    """
    barrier = threading.Barrier(n_threads, timeout=30)
    results = []
    lock = threading.Lock()

    def run():
        barrier.wait()
        got = GPULock(path).try_acquire()
        with lock:
            results.append(got)

    threads = [threading.Thread(target=run) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    return results


@pytest.mark.parametrize("n_threads", [2, 8])
def test_race_on_free_path_has_exactly_one_winner(tmp_path, n_threads):
    """N threads racing an unheld lock: exactly one may win, every time."""
    reps = 100
    for i in range(reps):
        path = str(tmp_path / f"free_{n_threads}_{i}.lock")
        results = _race_try_acquire(path, n_threads)
        assert len(results) == n_threads
        assert results.count(True) == 1, (
            f"rep {i}: {results.count(True)} of {n_threads} threads believed "
            f"they held the lock (expected exactly 1)"
        )


@pytest.mark.parametrize("n_threads", [2, 8])
def test_race_on_stale_lock_has_exactly_one_winner(tmp_path, n_threads):
    """Same race, but against a lock left behind by a dead pid.

    Two things must hold: exactly one thread wins, AND the dead holder's lock
    is genuinely taken over (stale detection must not be crippled by the fix).
    """
    reps = 100
    for i in range(reps):
        path = str(tmp_path / f"stale_{n_threads}_{i}.lock")
        _write_stale(path)

        results = _race_try_acquire(path, n_threads)
        assert results.count(True) == 1, (
            f"rep {i}: {results.count(True)} of {n_threads} threads stole the "
            f"same stale lock (expected exactly 1)"
        )
        # The dead holder was genuinely replaced, not merely left in place.
        with open(path, "r", encoding="utf-8") as f:
            info = json.load(f)
        assert info["pid"] == os.getpid()
        assert info["pid"] != DEAD_PID


def test_stale_steal_is_atomic_against_a_delayed_second_stealer(tmp_path, monkeypatch):
    """Regression test for BUG 1, pinned to the exact interleaving that broke.

    Both threads read the SAME stale ledger (a barrier inside `pid_alive`
    guarantees it — that call happens immediately after the read). Then the
    first thread is allowed to finish its entire steal, and only afterwards is
    the second thread allowed to act on the now-obsolete ledger it read.

    Against the old unconditional `os.remove` steal this interleaving produced
    two winners 50 times out of 50: thread B deleted thread A's freshly
    created lockfile and created its own on top.
    """
    real_pid_alive = gpu_lock.pid_alive
    reps = 30

    for i in range(reps):
        path = str(tmp_path / f"interleave_{i}.lock")
        _write_stale(path)

        barrier = threading.Barrier(2, timeout=30)
        first_finished = threading.Event()
        roles = {}
        roles_lock = threading.Lock()
        tlocal = threading.local()

        def gated_pid_alive(pid):
            alive = real_pid_alive(pid)
            # Only gate the first "is the stale holder dead?" question each
            # thread asks; later internal checks must not re-enter the barrier.
            if alive or getattr(tlocal, "gated", False):
                return alive
            tlocal.gated = True
            barrier.wait()  # both threads have now inspected the same stale file
            with roles_lock:
                role = "A" if "A" not in roles else "B"
                roles[role] = threading.current_thread()
            if role == "B":
                first_finished.wait(timeout=30)
            return alive

        monkeypatch.setattr(gpu_lock, "pid_alive", gated_pid_alive)

        results = []
        results_lock = threading.Lock()

        def run():
            try:
                got = GPULock(path).try_acquire()
                with results_lock:
                    results.append(got)
            finally:
                if roles.get("A") is threading.current_thread():
                    first_finished.set()

        threads = [threading.Thread(target=run) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        first_finished.set()

        assert results.count(True) == 1, (
            f"rep {i}: both threads stole the same stale lock — the steal is "
            f"not atomic ({results})"
        )


def test_lockfile_is_never_observable_partially_written(tmp_path):
    """Regression test for BUG 2.

    A concurrent reader must never see the lockfile in a state it cannot
    parse. The old `_write_new` created the file with O_CREAT|O_EXCL and only
    then wrote the pid, leaving a zero-byte window; a reader hitting that
    window classified the lock as stale and stole a lock that was being
    created at that very moment.

    Here a writer hammers acquire/release while a reader continuously reads
    the raw bytes. Every observation in which the file exists must be a
    complete, parseable ledger.
    """
    path = str(tmp_path / "partial.lock")
    cycles = 400
    stop = threading.Event()
    bad = []
    observations = [0]

    def writer():
        try:
            for _ in range(cycles):
                lock = GPULock(path)
                if lock.try_acquire():
                    lock.release()
        finally:
            stop.set()

    def reader():
        while not stop.is_set():
            try:
                with open(path, "rb") as f:
                    raw = f.read()
            except OSError:
                continue  # absent (or momentarily unopenable) — fine
            observations[0] += 1
            try:
                info = json.loads(raw.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                bad.append(raw)
                continue
            if not info.get("pid"):
                bad.append(raw)

    wt, rt = threading.Thread(target=writer), threading.Thread(target=reader)
    rt.start()
    wt.start()
    wt.join(timeout=60)
    stop.set()
    rt.join(timeout=30)

    assert observations[0] > 0, "reader never managed to observe the lockfile"
    assert not bad, (
        f"{len(bad)} of {observations[0]} observations saw a partially written "
        f"lockfile, e.g. {bad[0]!r}"
    )


def test_reader_never_steals_a_lock_that_is_being_created(tmp_path):
    """The behavioural consequence of BUG 2: while one thread is creating the
    lock, another must never conclude it is stale and take it.

    Rather than patching the write path, this races many create/read cycles
    and asserts the two never hold simultaneously.
    """
    path = str(tmp_path / "concurrent_create.lock")
    cycles = 400
    stop = threading.Event()
    holders = []  # non-empty at the same time from both sides == violation
    violations = []
    guard = threading.Lock()

    def writer():
        try:
            for _ in range(cycles):
                lock = GPULock(path)
                if lock.try_acquire():
                    with guard:
                        holders.append("writer")
                        if len(holders) > 1:
                            violations.append(list(holders))
                    with guard:
                        holders.remove("writer")
                    lock.release()
        finally:
            stop.set()

    def stealer():
        while not stop.is_set():
            lock = GPULock(path)
            if lock.try_acquire():
                with guard:
                    holders.append("stealer")
                    if len(holders) > 1:
                        violations.append(list(holders))
                with guard:
                    holders.remove("stealer")
                lock.release()

    wt, st = threading.Thread(target=writer), threading.Thread(target=stealer)
    st.start()
    wt.start()
    wt.join(timeout=60)
    stop.set()
    st.join(timeout=30)

    assert not violations, f"two threads held the lock at once: {violations[:3]}"


# The winner must still HOLD the lock while its siblings are contending.
# Printing and exiting immediately (what this did before) kills the winner's
# pid while the ledger still names it — so a straggler still inside
# `try_acquire`'s retry loop correctly judges the lock stale and steals it,
# and reports WON too. That is the steal path working as designed (a crashed
# render must not deadlock the GPU forever); the test was asserting an
# invariant it had itself broken, and failed ~2 runs in 12.
#
# Each child now marks that it has finished contending, and the winner waits
# for every sibling's marker before releasing. "Exactly one winner" is then a
# real invariant over the whole contention window, not a timing accident.
_CHILD_SRC = """\
import os, sys, time
sys.path.insert(0, sys.argv[1])
from pipeline.gpu_lock import GPULock
path, start_at, n_procs = sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
while time.time() < start_at:   # spin so all children contend at once
    pass
won = GPULock(path).try_acquire()
open(path + ".done." + str(os.getpid()), "w").close()
if won:
    d = os.path.dirname(path) or "."
    marker = os.path.basename(path) + ".done."
    deadline = time.time() + 30
    while time.time() < deadline:
        done = [f for f in os.listdir(d) if f.startswith(marker)]
        if len(done) >= n_procs:
            break
        time.sleep(0.01)
print("WON" if won else "LOST")
"""


@pytest.mark.parametrize("stale", [False, True])
def test_race_across_real_subprocesses_has_exactly_one_winner(tmp_path, stale):
    """The real deployment shape: separate OS processes, not threads.

    Fewer repetitions than the threaded tests because process startup on
    Windows is expensive; the wall-clock start barrier is what makes them
    genuinely contend rather than run one after another.
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(gpu_lock.__file__)))
    child = tmp_path / "child.py"
    child.write_text(_CHILD_SRC, encoding="utf-8")

    reps, n_procs = 3, 4
    for i in range(reps):
        path = str(tmp_path / f"proc_{stale}_{i}.lock")
        if stale:
            _write_stale(path)

        start_at = time.time() + 1.5  # generous: covers interpreter startup
        procs = [
            subprocess.Popen(
                [sys.executable, str(child), repo_root, path, str(start_at), str(n_procs)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            for _ in range(n_procs)
        ]
        outs = []
        for p in procs:
            out, err = p.communicate(timeout=120)
            assert p.returncode == 0, f"child failed: {err}"
            outs.append(out.strip())

        assert outs.count("WON") == 1, (
            f"rep {i} (stale={stale}): {outs.count('WON')} of {n_procs} "
            f"processes won the lock (expected exactly 1) — {outs}"
        )
        if stale:
            with open(path, "r", encoding="utf-8") as f:
                assert json.load(f)["pid"] != DEAD_PID


# ------------------------------------------------- ownership on release --
def test_release_does_not_remove_a_lock_another_holder_now_owns(tmp_path):
    """If our lockfile is replaced while we think we hold it (the residual
    steal race), releasing must not delete the new owner's lock."""
    path = _lock_path(tmp_path)
    mine = GPULock(path)
    assert mine.try_acquire() is True

    # Simulate someone else having taken over the path underneath us.
    os.remove(path)
    other = GPULock(path)
    assert other.try_acquire() is True

    mine.release()
    assert os.path.exists(path), "release() deleted a lock we no longer owned"
    with open(path, "r", encoding="utf-8") as f:
        assert json.load(f)["nonce"] == other._nonce


# ------------------------------------------- steal arbitration (BUG 3) --
def test_the_lock_path_is_never_empty_during_a_stale_takeover(tmp_path):
    """The stale lock is REPLACED in place, never removed or renamed away.

    Every earlier design let the path go absent for an instant, and each one
    produced two holders under contention: a fresh acquirer would create a lock
    in the empty slot and a straggling stealer would then rip it away. Pin the
    invariant directly — a watcher thread polling as fast as it can must never
    observe the path missing while a takeover runs.
    """
    path = _lock_path(tmp_path)
    _write_stale(path)

    stop = threading.Event()
    saw_missing = []

    def watch():
        while not stop.is_set():
            if not os.path.exists(path):
                saw_missing.append(True)
                return

    watcher = threading.Thread(target=watch)
    watcher.start()
    try:
        assert GPULock(path).try_acquire() is True
    finally:
        stop.set()
        watcher.join(timeout=10)

    assert not saw_missing, "the lock path went missing during the takeover"


def test_an_empty_arbiter_is_not_treated_as_abandoned(tmp_path):
    """The arbitration file is published atomically, so it is never empty.

    When it WAS created empty and filled in afterwards, a second claimant read
    no pid from it, judged the stealer dead, removed the arbiter and claimed it
    too — two concurrent stealers, which is precisely the condition the
    in-place takeover assumes cannot happen. A zero-byte arbiter must now read
    as "held", not as "abandoned".
    """
    path = _lock_path(tmp_path)
    _write_stale(path)
    lock = GPULock(path)

    open(lock._arbiter_path(), "w").close()  # the window that used to exist
    assert lock._claim_arbiter() is False

    os.remove(lock._arbiter_path())
    assert lock._claim_arbiter() is True
    os.remove(lock._arbiter_path())


def test_an_arbiter_left_by_a_dead_stealer_is_cleared(tmp_path):
    """...but a genuinely abandoned arbiter must not block steals forever."""
    path = _lock_path(tmp_path)
    _write_stale(path)
    lock = GPULock(path)

    with open(lock._arbiter_path(), "w", encoding="utf-8") as f:
        f.write(str(DEAD_PID))

    assert lock.try_acquire() is True
    assert not os.path.exists(lock._arbiter_path())


def test_a_takeover_leaves_no_side_files_behind(tmp_path):
    path = _lock_path(tmp_path)
    _write_stale(path)
    assert GPULock(path).try_acquire() is True
    assert os.listdir(tmp_path) == ["gpu_render.lock"]
