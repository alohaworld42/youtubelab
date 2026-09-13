"""Scheduler defects around a render that ALREADY SUCCEEDED, and around jobs
that quietly stop moving.

Three things are pinned here:

  * Post-success bookkeeping (idea status, asset touch, cache prune) used to run
    inside `_run_pipeline`'s own `try`, between `set_job_status(ready)` and the
    success log line. Anything raising in there was caught by the render's
    `except`, which flipped the finished job back to 'queued' and re-rendered
    it — an hour of GPU thrown away because a cache file could not be deleted.
  * An approved job on a `needs_reauth` channel was skipped by the uploader on
    every tick with no record anywhere. 'approved' appears in no view (not the
    review queue, not the failed list, not a counter), so the video simply
    vanished from the operator's point of view.
  * The daily-upload-cap warning was logged once per upload tick — ~5700 lines
    a day at the default 15s tick.

No GPU/network: `pipeline.generate.generate` and the uploader are
monkeypatched, same as tests/test_scheduler_render.py.
"""
import logging

import pytest

from studio import assets, models
from studio.scheduler import Scheduler


def _cfg(tmp_path):
    return {
        "_root": str(tmp_path),
        "paths": {"output_dir": "out"},
        "studio": {"max_attempts": 3, "tick_seconds": 1, "daily_upload_cap": 2},
    }


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


def _ok_render(monkeypatch):
    monkeypatch.setattr("pipeline.generate.generate", lambda *a, **k: _result())


# ------------------------------------------- bookkeeping cannot undo a render --
def test_cache_prune_failure_does_not_requeue_a_finished_render(
    tmp_db, tmp_path, monkeypatch
):
    ch = models.create_channel("Test")
    job_id = models.create_job(channel_id=ch, video_type="kidsong", topic="counting")
    job = models.get_job(job_id)
    _ok_render(monkeypatch)

    monkeypatch.setattr(assets, "prune_cache",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("cache file busy")))

    Scheduler(_cfg(tmp_path))._run_pipeline(job)

    updated = models.get_job(job_id)
    assert updated["status"] == "ready", (
        "a finished render was thrown away because post-success bookkeeping failed"
    )
    assert updated["attempts"] == 0
    assert models.get_video_by_job(job_id) is not None


def test_asset_touch_failure_does_not_requeue_a_finished_render(
    tmp_db, tmp_path, monkeypatch
):
    """Needs a job that actually sources a background — kidsong renders its own
    footage, so `bg_asset_id` is None there and `touch_asset` is never reached."""
    ch = models.create_channel("Test")
    job_id = models.create_job(channel_id=ch, video_type="facts", topic="space")
    job = models.get_job(job_id)
    _ok_render(monkeypatch)

    bg = tmp_path / "bg.mp4"
    bg.write_bytes(b"x")
    monkeypatch.setattr(assets, "get_background", lambda *a, **k: (str(bg), 7))
    monkeypatch.setattr(assets, "prune_cache", lambda *a, **k: None)
    monkeypatch.setattr(models, "touch_asset",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("db locked")))

    Scheduler(_cfg(tmp_path))._run_pipeline(job)

    updated = models.get_job(job_id)
    assert updated["status"] == "ready"
    assert updated["attempts"] == 0


def test_idea_bookkeeping_failure_does_not_requeue_a_finished_render(
    tmp_db, tmp_path, monkeypatch
):
    ch = models.create_channel("Test")
    from studio import ideas as ideas_mod

    idea_id = ideas_mod.add_user_idea(ch, "counting to five")
    job_id = models.create_job(channel_id=ch, idea_id=idea_id,
                               video_type="kidsong", topic="counting to five")
    job = models.get_job(job_id)
    _ok_render(monkeypatch)

    monkeypatch.setattr(models, "set_idea_status",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("db locked")))

    Scheduler(_cfg(tmp_path))._run_pipeline(job)
    assert models.get_job(job_id)["status"] == "ready"


def test_a_real_render_failure_still_requeues(tmp_db, tmp_path, monkeypatch):
    """The guard above must not swallow an actual render error."""
    ch = models.create_channel("Test")
    job_id = models.create_job(channel_id=ch, video_type="kidsong", topic="counting")
    job = models.get_job(job_id)

    monkeypatch.setattr("pipeline.generate.generate",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("ComfyUI died")))

    Scheduler(_cfg(tmp_path))._run_pipeline(job)

    updated = models.get_job(job_id)
    assert updated["status"] == "queued"
    assert updated["attempts"] == 1
    assert "ComfyUI died" in updated["error"]


# --------------------------------------------- a stalled upload is not silent --
def test_a_reauth_blocked_upload_records_why(tmp_db, tmp_path, monkeypatch):
    ch = models.create_channel("Test")
    models.update_channel(ch, status="needs_reauth")
    job_id = models.create_job(channel_id=ch, video_type="kidsong", topic="counting")
    models.create_video(job_id=job_id, channel_id=ch, path=str(tmp_path / "v.mp4"),
                        title="T", description="D", tags=["t"], duration=1.0)
    (tmp_path / "v.mp4").write_bytes(b"x")
    models.set_job_status(job_id, "approved")

    sched = Scheduler(_cfg(tmp_path))
    sched._process_uploads()

    assert models.get_job(job_id)["status"] == "approved"  # still waiting, not failed
    messages = [e["message"] for e in models.events_for_job(job_id)]
    assert any("neue Anmeldung" in m for m in messages), messages


def test_the_reauth_reason_is_recorded_once_not_every_tick(tmp_db, tmp_path):
    ch = models.create_channel("Test")
    models.update_channel(ch, status="needs_reauth")
    job_id = models.create_job(channel_id=ch, video_type="kidsong", topic="counting")
    models.create_video(job_id=job_id, channel_id=ch, path=str(tmp_path / "v.mp4"),
                        title="T", description="D", tags=["t"], duration=1.0)
    (tmp_path / "v.mp4").write_bytes(b"x")
    models.set_job_status(job_id, "approved")

    sched = Scheduler(_cfg(tmp_path))
    for _ in range(5):
        sched._process_uploads()

    blocked = [e for e in models.events_for_job(job_id) if "neue Anmeldung" in e["message"]]
    assert len(blocked) == 1, f"the uploader logged the same block {len(blocked)} times"


def test_the_daily_cap_warning_is_logged_once_a_day(tmp_db, tmp_path, monkeypatch, caplog):
    ch = models.create_channel("Test")
    job_id = models.create_job(channel_id=ch, video_type="kidsong", topic="counting")
    models.set_job_status(job_id, "approved")
    monkeypatch.setattr(models, "uploads_today", lambda: 99)

    sched = Scheduler(_cfg(tmp_path))
    with caplog.at_level(logging.WARNING, logger="studio.scheduler"):
        for _ in range(10):
            sched._process_uploads()

    hits = [r for r in caplog.records if "daily upload cap" in r.getMessage()]
    assert len(hits) == 1, f"logged the cap {len(hits)} times in 10 ticks"


# ------------------------------------------------- the count that surfaces it --
def test_waiting_upload_count_is_exposed(tmp_db, tmp_path):
    ch = models.create_channel("Test")
    approved = models.create_job(channel_id=ch, video_type="kidsong", topic="a")
    models.set_job_status(approved, "approved")

    assert models.count_jobs("approved") == 1


# ------------------------------------------------------- COPPA declaration --
def _upload_capture(monkeypatch):
    """Capture the kwargs the scheduler hands to pipeline.youtube_upload.upload."""
    seen = {}

    def fake_upload(*a, **k):
        seen.update(k)
        return {"video_id": "x", "url": "https://youtu.be/x", "privacy": "private"}

    import pipeline.youtube_upload as yt

    monkeypatch.setattr(yt, "upload", fake_upload)
    monkeypatch.setattr("studio.oauth.channel_upload_paths",
                        lambda *a, **k: ("secret.json", "token.json"))
    return seen


def _approved_job(tmp_path, ch, video_type):
    job_id = models.create_job(channel_id=ch, video_type=video_type, topic="counting")
    (tmp_path / "v.mp4").write_bytes(b"x")
    models.create_video(job_id=job_id, channel_id=ch, path=str(tmp_path / "v.mp4"),
                        title="T", description="D", tags=["t"], duration=1.0)
    models.set_job_status(job_id, "approved")
    return job_id


@pytest.mark.parametrize("video_type", ["kidsong", "kids"])
def test_a_kids_video_is_declared_made_for_kids_even_on_an_unflagged_channel(
    tmp_db, tmp_path, monkeypatch, video_type
):
    """The channel checkbox is an opt-IN, never an opt-out. A kidsong queued
    against a channel whose box is unchecked used to upload declared
    not-for-kids — a COPPA declaration this repo's own rules forbid."""
    ch = models.create_channel("Test")            # made_for_kids defaults to 0
    _approved_job(tmp_path, ch, video_type)
    seen = _upload_capture(monkeypatch)

    Scheduler(_cfg(tmp_path))._process_uploads()

    assert seen.get("made_for_kids") is True


def test_a_non_kids_video_still_follows_the_channel_flag(tmp_db, tmp_path, monkeypatch):
    ch = models.create_channel("Test")
    _approved_job(tmp_path, ch, "facts")
    seen = _upload_capture(monkeypatch)

    Scheduler(_cfg(tmp_path))._process_uploads()

    assert seen.get("made_for_kids") is False


def test_a_flagged_channel_declares_kids_for_every_type(tmp_db, tmp_path, monkeypatch):
    ch = models.create_channel("Test")
    models.update_channel(ch, made_for_kids=1)
    _approved_job(tmp_path, ch, "facts")
    seen = _upload_capture(monkeypatch)

    Scheduler(_cfg(tmp_path))._process_uploads()

    assert seen.get("made_for_kids") is True


# ------------------------------------- COPPA: the style is a kids signal too --
# YouTube's classification does not ask what a database column is called. A
# video counts as made for kids when children are the primary audience, and also
# when its characters, songs, activities or language appeal to children as part
# of a mixed audience. A "kids"-styled video is that by construction:
# `studio/svm.py` gives it a kids voice, happy music at high volume and a pink
# caption background. The declaration used to consult only the channel checkbox
# and the video type, so this whole shape uploaded declared not-for-kids.
def test_a_kids_styled_channel_declares_kids_even_for_a_non_kids_video_type(
    tmp_db, tmp_path, monkeypatch
):
    """The documented real-world shape: `studio/ideas.py` records that the
    channel KinderKrippenLernLieder "used to be a 'facts' channel". A kids
    channel still carrying video_type=facts, with the box unticked, published
    children's content declared as not children's content."""
    ch = models.create_channel("KinderKrippenLernLieder")
    models.update_channel(ch, style="kids")       # made_for_kids still 0
    _approved_job(tmp_path, ch, "facts")
    seen = _upload_capture(monkeypatch)

    Scheduler(_cfg(tmp_path))._process_uploads()

    assert seen.get("made_for_kids") is True


def test_a_kids_styled_job_declares_kids_on_an_unstyled_channel(
    tmp_db, tmp_path, monkeypatch
):
    """`jobs.style` overrides the channel's, and /quick sets it per job — so the
    job's own style has to count, not just the channel's."""
    ch = models.create_channel("Test")
    job_id = _approved_job(tmp_path, ch, "facts")
    models.update_job(job_id, style="kids")
    seen = _upload_capture(monkeypatch)

    Scheduler(_cfg(tmp_path))._process_uploads()

    assert seen.get("made_for_kids") is True


@pytest.mark.parametrize("style", ["brainrot", "clean", "", None])
def test_a_non_kids_style_does_not_invent_a_declaration(
    tmp_db, tmp_path, monkeypatch, style
):
    """The declaration is not free in the other direction: a made-for-kids video
    loses personalised ads, comments and end screens. Only real signals count."""
    ch = models.create_channel("Test")
    models.update_channel(ch, style=style or "")
    _approved_job(tmp_path, ch, "facts")
    seen = _upload_capture(monkeypatch)

    Scheduler(_cfg(tmp_path))._process_uploads()

    assert seen.get("made_for_kids") is False


def test_channel_prose_alone_never_triggers_the_declaration(
    tmp_db, tmp_path, monkeypatch
):
    """`cascade` may guess a kids channel from name/description wording, because
    guessing wrong there only mis-ranks a row. A COPPA declaration is a legal
    filing, so it follows explicit operator choices — style and video type —
    and never prose."""
    ch = models.create_channel("Kinderlieder für Kleinkinder")
    models.update_channel(ch, description="Nursery rhymes for toddlers and kids")
    _approved_job(tmp_path, ch, "facts")
    seen = _upload_capture(monkeypatch)

    Scheduler(_cfg(tmp_path))._process_uploads()

    assert seen.get("made_for_kids") is False
