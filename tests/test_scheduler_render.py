"""Tests for the two scheduler defects: an unreviewed kidsong cut auto-
approving (must never happen), and the GPU lock around the render step (must
exclude a live holder without burning a retry, and must always release).

No GPU/network/ComfyUI: `pipeline.generate.generate` (imported by name
inside `_run_pipeline`) is monkeypatched, so these only exercise scheduler
bookkeeping.
"""
import json
import os

from pipeline import gpu_lock
from studio import assets, models
from studio.scheduler import Scheduler


def _cfg(tmp_path):
    return {
        "_root": str(tmp_path),
        "paths": {"output_dir": "out"},
        "studio": {"max_attempts": 3, "tick_seconds": 1},
    }


def _make_job(ch, video_type="kidsong", auto_approve=False):
    return models.create_job(
        channel_id=ch, video_type=video_type, topic="counting to five",
        auto_approve=auto_approve,
    )


def _result(**overrides):
    base = {
        "video_path": "out/video.mp4",
        "title": "T",
        "description": "D",
        "tags": ["t"],
        "duration": 42.0,
    }
    base.update(overrides)
    return base


# ------------------------------------------------------- needs_review gate --
def test_needs_review_cut_never_becomes_approved(tmp_db, tmp_path, monkeypatch):
    ch = models.create_channel("Test")
    job_id = _make_job(ch, video_type="kidsong", auto_approve=True)
    job = models.get_job(job_id)

    monkeypatch.setattr(
        "pipeline.generate.generate",
        lambda *a, **k: _result(status="needs_review"),
    )

    Scheduler(_cfg(tmp_path))._run_pipeline(job)

    updated = models.get_job(job_id)
    assert updated["status"] == "ready"          # never "approved", despite auto_approve
    assert updated["status"] != "approved"

    events = models.events_for_job(job_id)
    assert any("needs_review" in e["message"] or "review" in e["stage"] for e in events)


def test_needs_review_cut_ready_even_without_auto_approve(tmp_db, tmp_path, monkeypatch):
    ch = models.create_channel("Test")
    job_id = _make_job(ch, video_type="kidsong", auto_approve=False)
    job = models.get_job(job_id)

    monkeypatch.setattr(
        "pipeline.generate.generate",
        lambda *a, **k: _result(status="needs_review"),
    )

    Scheduler(_cfg(tmp_path))._run_pipeline(job)

    assert models.get_job(job_id)["status"] == "ready"


def test_approved_kidsong_cut_still_auto_approves(tmp_db, tmp_path, monkeypatch):
    """Regression guard: a cut that DID pass QC must keep the old behaviour."""
    ch = models.create_channel("Test")
    job_id = _make_job(ch, video_type="kidsong", auto_approve=True)
    job = models.get_job(job_id)

    monkeypatch.setattr(
        "pipeline.generate.generate",
        lambda *a, **k: _result(status="approved"),
    )

    Scheduler(_cfg(tmp_path))._run_pipeline(job)

    assert models.get_job(job_id)["status"] == "approved"


def test_non_kidsong_video_type_unaffected_by_status_field(tmp_db, tmp_path, monkeypatch):
    """Other video types never return a "status" key — behaviour must be
    exactly as before (approved iff auto_approve)."""
    ch = models.create_channel("Test")
    monkeypatch.setattr(assets, "get_background", lambda *a, **k: (None, None))

    monkeypatch.setattr("pipeline.generate.generate", lambda *a, **k: _result())

    job_id = _make_job(ch, video_type="facts", auto_approve=True)
    Scheduler(_cfg(tmp_path))._run_pipeline(models.get_job(job_id))
    assert models.get_job(job_id)["status"] == "approved"

    job_id2 = _make_job(ch, video_type="facts", auto_approve=False)
    Scheduler(_cfg(tmp_path))._run_pipeline(models.get_job(job_id2))
    assert models.get_job(job_id2)["status"] == "ready"


# --------------------------------------------------------- background skip --
def test_kidsong_job_skips_background_asset_sourcing(tmp_db, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(assets, "get_background", lambda *a, **k: calls.append(1) or (None, None))
    monkeypatch.setattr("pipeline.generate.generate", lambda *a, **k: _result(status="approved"))

    ch = models.create_channel("Test")
    job_id = _make_job(ch, video_type="kidsong")
    Scheduler(_cfg(tmp_path))._run_pipeline(models.get_job(job_id))

    assert calls == []  # generate_kidsong renders its own footage — never sourced


def test_normal_job_still_sources_background_asset(tmp_db, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(assets, "get_background", lambda *a, **k: calls.append(1) or (None, None))
    monkeypatch.setattr("pipeline.generate.generate", lambda *a, **k: _result())

    ch = models.create_channel("Test")
    job_id = _make_job(ch, video_type="facts")
    Scheduler(_cfg(tmp_path))._run_pipeline(models.get_job(job_id))

    assert calls == [1]


# ------------------------------------------------------------------ GPU lock --
def test_live_lock_holder_skips_tick_quietly(tmp_db, tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    lock = gpu_lock.get_lock(cfg)
    os.makedirs(os.path.dirname(lock.path), exist_ok=True)
    with open(lock.path, "w", encoding="utf-8") as f:
        json.dump({"pid": os.getpid(), "started_at": "now"}, f)  # our own pid: always alive

    called = []
    monkeypatch.setattr(Scheduler, "_run_pipeline", lambda self, job: called.append(job["id"]))

    ch = models.create_channel("Test")
    job_id = _make_job(ch, video_type="kidsong")

    Scheduler(cfg)._process_one_render()

    assert called == []                                   # render never started
    job = models.get_job(job_id)
    assert job["status"] == "queued"                       # untouched
    assert job["attempts"] == 0                             # no retry burned
    assert os.path.exists(lock.path)                        # the live holder's lock survives


def test_lock_is_released_after_successful_render(tmp_db, tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(assets, "get_background", lambda *a, **k: (None, None))
    monkeypatch.setattr("pipeline.generate.generate", lambda *a, **k: _result(status="approved"))

    ch = models.create_channel("Test")
    job_id = _make_job(ch, video_type="kidsong", auto_approve=True)

    Scheduler(cfg)._process_one_render()

    assert models.get_job(job_id)["status"] == "approved"
    lock = gpu_lock.get_lock(cfg)
    assert not os.path.exists(lock.path)                    # released, not left dangling
    assert lock.try_acquire() is True                        # a fresh acquire proves it's free


def test_lock_is_released_after_render_exception(tmp_db, tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(assets, "get_background", lambda *a, **k: (None, None))

    def _boom(*a, **k):
        raise RuntimeError("ComfyUI timed out")

    monkeypatch.setattr("pipeline.generate.generate", _boom)

    ch = models.create_channel("Test")
    job_id = _make_job(ch, video_type="kidsong")

    Scheduler(cfg)._process_one_render()

    job = models.get_job(job_id)
    assert job["status"] == "queued"          # normal retry path, not "held forever"
    assert job["attempts"] == 1
    assert "ComfyUI timed out" in (job["error"] or "")

    lock = gpu_lock.get_lock(cfg)
    assert not os.path.exists(lock.path)      # lock released despite the exception
    assert lock.try_acquire() is True


# -------------------------------------------------------- resume_base (C3) --
def test_kidsong_failure_persists_resume_base_from_exception(tmp_db, tmp_path, monkeypatch):
    """A kidsong run that dies mid-render tags the exception with the base it
    was rendering (pipeline.kidsong.generate._run); the scheduler must persist
    that onto the job row's nullable resume_base column so a retry can pick it
    back up instead of starting a brand-new episode."""
    monkeypatch.setattr(assets, "get_background", lambda *a, **k: (None, None))

    def _boom(*a, **k):
        exc = RuntimeError("ComfyUI infra failure, gave up after retries")
        exc.kidsong_resume_base = "20260721-090000-kidsong-died-mid-render"
        raise exc

    monkeypatch.setattr("pipeline.generate.generate", _boom)

    ch = models.create_channel("Test")
    job_id = _make_job(ch, video_type="kidsong")

    Scheduler(_cfg(tmp_path))._run_pipeline(models.get_job(job_id))

    job = models.get_job(job_id)
    assert job["status"] == "queued"
    assert job["resume_base"] == "20260721-090000-kidsong-died-mid-render"


def test_resume_base_threaded_into_generate_on_retry(tmp_db, tmp_path, monkeypatch):
    """Once a job carries a resume_base, the NEXT _run_pipeline call must pass
    it through to generate() -- the actual resume mechanism."""
    monkeypatch.setattr(assets, "get_background", lambda *a, **k: (None, None))

    captured = {}

    def fake_generate(*a, **kw):
        captured["resume_base"] = kw.get("resume_base")
        return _result(status="approved")

    monkeypatch.setattr("pipeline.generate.generate", fake_generate)

    ch = models.create_channel("Test")
    job_id = _make_job(ch, video_type="kidsong")
    models.update_job(job_id, resume_base="20260721-090000-kidsong-died-mid-render")

    Scheduler(_cfg(tmp_path))._run_pipeline(models.get_job(job_id))

    assert captured["resume_base"] == "20260721-090000-kidsong-died-mid-render"


def test_resume_base_cleared_on_success(tmp_db, tmp_path, monkeypatch):
    """Once the resumed run finishes, the pointer must be cleared -- nothing
    left to resume, and a future unrelated failure must not inherit it."""
    monkeypatch.setattr(assets, "get_background", lambda *a, **k: (None, None))
    monkeypatch.setattr("pipeline.generate.generate", lambda *a, **k: _result(status="approved"))

    ch = models.create_channel("Test")
    job_id = _make_job(ch, video_type="kidsong", auto_approve=True)
    models.update_job(job_id, resume_base="20260721-090000-kidsong-died-mid-render")

    Scheduler(_cfg(tmp_path))._run_pipeline(models.get_job(job_id))

    job = models.get_job(job_id)
    assert job["status"] == "approved"
    assert job["resume_base"] is None


def test_unrelated_failure_does_not_clobber_existing_resume_base(tmp_db, tmp_path, monkeypatch):
    """An exception with no kidsong_resume_base attribute (e.g. a failure
    before any base existed) must not overwrite a resume_base a PREVIOUS
    attempt already recorded."""
    monkeypatch.setattr(assets, "get_background", lambda *a, **k: (None, None))
    monkeypatch.setattr(
        "pipeline.generate.generate",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("unrelated failure, no base yet")),
    )

    ch = models.create_channel("Test")
    job_id = _make_job(ch, video_type="kidsong")
    models.update_job(job_id, resume_base="20260721-090000-kidsong-died-mid-render")

    Scheduler(_cfg(tmp_path))._run_pipeline(models.get_job(job_id))

    job = models.get_job(job_id)
    assert job["resume_base"] == "20260721-090000-kidsong-died-mid-render"


def test_second_tick_proceeds_once_lock_is_free(tmp_db, tmp_path, monkeypatch):
    """End-to-end sanity: a busy tick that skips is followed by a tick that
    actually renders once the lock clears."""
    cfg = _cfg(tmp_path)
    lock = gpu_lock.get_lock(cfg)
    os.makedirs(os.path.dirname(lock.path), exist_ok=True)
    with open(lock.path, "w", encoding="utf-8") as f:
        json.dump({"pid": os.getpid(), "started_at": "now"}, f)

    monkeypatch.setattr(assets, "get_background", lambda *a, **k: (None, None))
    monkeypatch.setattr("pipeline.generate.generate", lambda *a, **k: _result(status="approved"))

    ch = models.create_channel("Test")
    job_id = _make_job(ch, video_type="kidsong", auto_approve=True)

    sched = Scheduler(cfg)
    sched._process_one_render()
    assert models.get_job(job_id)["status"] == "queued"      # first tick: busy, skipped

    os.remove(lock.path)                                      # simulate the other holder finishing
    sched._process_one_render()
    assert models.get_job(job_id)["status"] == "approved"     # second tick: rendered
