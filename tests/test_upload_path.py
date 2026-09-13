"""Regression tests for "the upload never happened".

Three defects, all covered here without GPU, network or real OAuth:
  1. the cascade created a channel that could not upload instead of adopting
     the connected one;
  2. uploads starved behind the inline render on the scheduler thread;
  3. logging setup stacked a new file handler on every call.
"""
import logging
import os
import threading

import pytest

from studio import cascade, models
from studio.scheduler import Scheduler


def _cfg(tmp_path, **studio):
    base = {"max_attempts": 3, "tick_seconds": 1, "daily_upload_cap": 6}
    base.update(studio)
    return {"_root": str(tmp_path), "paths": {"output_dir": "out"}, "studio": base}


def _token(tmp_path, yt_id):
    d = tmp_path / "tokens"
    d.mkdir(exist_ok=True)
    p = d / f"{yt_id}.json"
    p.write_text("{}", encoding="utf-8")
    return str(p)


def _ready_video(tmp_path, channel_id, job_id, name="v.mp4"):
    path = tmp_path / name
    path.write_bytes(b"fake mp4")
    return models.create_video(
        job_id=job_id, channel_id=channel_id, path=str(path),
        title="T", description="D", tags=["t"], duration=1.0,
    )


def _stub_upload(monkeypatch, calls, fail=None):
    import pipeline.youtube_upload as yt

    def fake_upload(path, title, desc, tags, cfg, **kw):
        calls.append(path)
        if fail:
            raise fail
        return {"video_id": f"yt{len(calls)}", "url": f"https://y/{len(calls)}",
                "privacy": "private"}

    monkeypatch.setattr(yt, "upload", fake_upload)
    return calls


# ---------------------------------------------------- defect 1: channel pick --
def test_cascade_prefers_connected_channel_over_creating_one(tmp_db, tmp_path):
    """The real bug: a connected kids channel whose default_video_type was
    'facts' was missed, and a credential-less duplicate got created."""
    cfg = _cfg(tmp_path)
    real = models.create_channel(
        "KinderKrippenLernLieder", yt_channel_id="UCfnSZjWRt1WDES6FEky8vgQ",
        token_path=_token(tmp_path, "UCfnSZjWRt1WDES6FEky8vgQ"),
    )
    models.update_channel(real, default_video_type="facts", made_for_kids=1,
                          niche="Krippenlieder", style="kids")

    ch = cascade.ensure_kidsong_channel(cfg)

    assert ch["id"] == real
    assert len(models.list_channels()) == 1  # nothing new created
    # and the row is repaired so it can't be missed again
    assert ch["default_video_type"] == "kidsong"
    assert cascade.channel_can_upload(ch, cfg) is True


def test_cascade_prefers_connected_over_disconnected_kidsong_channel(tmp_db, tmp_path):
    cfg = _cfg(tmp_path)
    orphan = models.create_channel("Kidsong")
    models.update_channel(orphan, default_video_type="kidsong", made_for_kids=1)
    connected = models.create_channel(
        "KinderKrippenLernLieder", yt_channel_id="UC123",
        token_path=_token(tmp_path, "UC123"),
    )
    models.update_channel(connected, made_for_kids=1, style="kids")

    assert cascade.ensure_kidsong_channel(cfg)["id"] == connected


def test_channel_with_yt_id_but_missing_token_is_not_connected(tmp_db, tmp_path):
    cfg = _cfg(tmp_path)
    cid = models.create_channel("Kidsong", yt_channel_id="UCgone",
                                token_path=str(tmp_path / "tokens" / "nope.json"))
    models.update_channel(cid, made_for_kids=1)
    assert cascade.channel_can_upload(models.get_channel(cid), cfg) is False


def test_paused_channel_is_not_adopted(tmp_db, tmp_path):
    cfg = _cfg(tmp_path)
    old = models.create_channel("Kidsong (Duplikat)")
    models.update_channel(old, default_video_type="kidsong", status="paused")

    ch = cascade.ensure_kidsong_channel(cfg)
    assert ch["id"] != old
    assert ch["default_video_type"] == "kidsong"


def test_cascade_reports_clearly_when_only_disconnected_channel_exists(
        tmp_db, tmp_path, monkeypatch):
    from studio import ideas as ideas_mod

    monkeypatch.setattr(ideas_mod, "generate_ideas",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))
    summary = cascade.run_cascade(_cfg(tmp_path), count=1)

    assert summary["channel_connected"] is False
    assert "channel_warning" in summary
    assert "NICHT" in summary["channel_warning"]
    assert summary["jobs_created"] == 1  # work is still queued, just flagged


def test_cascade_summary_reports_connected_channel(tmp_db, tmp_path, monkeypatch):
    from studio import ideas as ideas_mod

    cfg = _cfg(tmp_path)
    cid = models.create_channel("KinderKrippenLernLieder", yt_channel_id="UC123",
                                token_path=_token(tmp_path, "UC123"))
    models.update_channel(cid, made_for_kids=1, default_video_type="kidsong")
    monkeypatch.setattr(ideas_mod, "generate_ideas",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))

    summary = cascade.run_cascade(cfg, count=1)
    assert summary["channel_id"] == cid
    assert summary["channel_connected"] is True
    assert "channel_warning" not in summary


# ------------------------------------------------- defect 2: upload progress --
def _approved_job(tmp_path, cfg, name="v.mp4"):
    cid = models.create_channel("KinderKrippenLernLieder", yt_channel_id="UC123",
                                token_path=_token(tmp_path, "UC123"))
    models.update_channel(cid, made_for_kids=1, default_video_type="kidsong")
    job_id = models.create_job(channel_id=cid, video_type="kidsong", topic="socks")
    _ready_video(tmp_path, cid, job_id, name)
    models.set_job_status(job_id, "approved")
    return cid, job_id


def test_tick_no_longer_uploads(tmp_db, tmp_path, monkeypatch):
    """Uploads must not be coupled to the render tick any more."""
    cfg = _cfg(tmp_path)
    _approved_job(tmp_path, cfg)
    calls = _stub_upload(monkeypatch, [])
    Scheduler(cfg).tick()
    assert calls == []  # the uploader thread owns this now


def test_approved_job_uploads_while_a_render_is_in_flight(tmp_db, tmp_path, monkeypatch):
    """The actual defect: an approved video had to wait ~1h for the render.

    The render is simulated by holding the scheduler thread inside
    _run_pipeline; the upload must complete before that unblocks.
    """
    cfg = _cfg(tmp_path, tick_seconds=1, upload_tick_seconds=1)
    cid, job_id = _approved_job(tmp_path, cfg)
    # a second, queued job that the render thread will pick up and block on
    models.create_job(channel_id=cid, video_type="kidsong", topic="long render")

    calls = _stub_upload(monkeypatch, [])
    render_started = threading.Event()
    let_render_finish = threading.Event()

    def slow_render(self, job):
        render_started.set()
        let_render_finish.wait(30)
        models.set_job_status(job["id"], "ready")

    monkeypatch.setattr(Scheduler, "_run_pipeline", slow_render)

    sched = Scheduler(cfg)
    sched.start()
    try:
        assert render_started.wait(10), "render never started"
        assert not let_render_finish.is_set()  # render still blocked
        deadline = threading.Event()
        for _ in range(100):
            if models.get_job(job_id)["status"] == "uploaded":
                break
            deadline.wait(0.1)
        assert models.get_job(job_id)["status"] == "uploaded"
        assert len(calls) == 1
        # the whole point: the upload finished while the render was STILL
        # running. Under the old inline tick() this could not happen.
        assert not let_render_finish.is_set()
    finally:
        let_render_finish.set()
        sched.stop()


def test_no_double_upload(tmp_db, tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    cid, job_id = _approved_job(tmp_path, cfg)
    calls = _stub_upload(monkeypatch, [])
    sched = Scheduler(cfg)

    sched._process_uploads()
    assert len(calls) == 1
    assert models.get_job(job_id)["status"] == "uploaded"

    # somebody flips it back to approved (stale UI action / recovery race)
    models.update_job(job_id, status="approved")
    sched._process_uploads()
    assert len(calls) == 1, "already-uploaded video must not be sent twice"
    assert models.get_job(job_id)["status"] == "uploaded"


def test_claim_is_atomic(tmp_db, tmp_path):
    cfg = _cfg(tmp_path)
    _cid, job_id = _approved_job(tmp_path, cfg)
    sched = Scheduler(cfg)
    assert sched._claim_for_upload(job_id) is True
    assert sched._claim_for_upload(job_id) is False  # no longer 'approved'


def test_concurrent_upload_passes_do_not_overlap(tmp_db, tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    _approved_job(tmp_path, cfg)
    sched = Scheduler(cfg)
    inside = []

    def slow_pass():
        inside.append(1)
        assert len(inside) == 1, "two upload passes ran at once"
        threading.Event().wait(0.3)
        inside.pop()

    monkeypatch.setattr(sched, "_process_uploads_locked", slow_pass)
    threads = [threading.Thread(target=sched._process_uploads) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert inside == []


def test_daily_upload_cap_still_holds(tmp_db, tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, daily_upload_cap=2)
    cid = models.create_channel("KinderKrippenLernLieder", yt_channel_id="UC123",
                                token_path=_token(tmp_path, "UC123"))
    models.update_channel(cid, made_for_kids=1, default_video_type="kidsong")
    job_ids = []
    for i in range(4):
        jid = models.create_job(channel_id=cid, video_type="kidsong", topic=f"t{i}")
        _ready_video(tmp_path, cid, jid, f"v{i}.mp4")
        models.set_job_status(jid, "approved")
        job_ids.append(jid)

    calls = _stub_upload(monkeypatch, [])
    Scheduler(cfg)._process_uploads()

    assert len(calls) == 2
    assert models.uploads_today() == 2
    assert sum(models.get_job(j)["status"] == "approved" for j in job_ids) == 2


def test_render_not_blocked_by_an_in_flight_upload(tmp_db, tmp_path, monkeypatch):
    """A job stuck in 'uploading' must not stall the render queue."""
    cfg = _cfg(tmp_path)
    cid, _job_id = _approved_job(tmp_path, cfg)
    models.set_job_status(_job_id, "uploading")
    queued = models.create_job(channel_id=cid, video_type="kidsong", topic="next")

    rendered = []
    monkeypatch.setattr(Scheduler, "_run_pipeline",
                        lambda self, job: rendered.append(job["id"]))
    Scheduler(cfg)._process_one_render()
    assert rendered == [queued]


# -------------------------------------- human approval of a needs_review cut --
def test_human_approved_needs_review_job_uploads(tmp_db, tmp_path, monkeypatch):
    """The QC guard holds the job at 'ready'; a human approving it in /review
    is the manual review the guard demands, so it must then upload."""
    from studio import assets

    cfg = _cfg(tmp_path)
    cid = models.create_channel("KinderKrippenLernLieder", yt_channel_id="UC123",
                                token_path=_token(tmp_path, "UC123"))
    models.update_channel(cid, made_for_kids=1, default_video_type="kidsong")
    job_id = models.create_job(channel_id=cid, video_type="kidsong", topic="socks",
                               auto_approve=True)
    video_file = tmp_path / "cut.mp4"
    video_file.write_bytes(b"fake mp4")

    monkeypatch.setattr(assets, "get_background", lambda *a, **k: (None, None))
    monkeypatch.setattr("pipeline.generate.generate", lambda *a, **k: {
        "video_path": str(video_file), "title": "T", "description": "D",
        "tags": ["t"], "duration": 1.0, "status": "needs_review",
    })

    sched = Scheduler(cfg)
    sched._run_pipeline(models.get_job(job_id))
    assert models.get_job(job_id)["status"] == "ready"  # guard intact

    calls = _stub_upload(monkeypatch, [])
    sched._process_uploads()
    assert calls == []  # nothing uploads while it is merely 'ready'

    models.set_job_status(job_id, "approved")  # what /review's approve does
    sched._process_uploads()
    assert models.get_job(job_id)["status"] == "uploaded"
    assert len(calls) == 1


def test_review_approve_endpoint_marks_job_approved(tmp_db, tmp_path, monkeypatch):
    from app import create_app

    cid = models.create_channel("KinderKrippenLernLieder", yt_channel_id="UC123",
                                token_path=_token(tmp_path, "UC123"))
    job_id = models.create_job(channel_id=cid, video_type="kidsong", topic="socks")
    vid = _ready_video(tmp_path, cid, job_id)
    models.set_job_status(job_id, "ready")

    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        res = c.post(f"/api/videos/{vid}/approve", json={})
    assert res.status_code == 200
    assert models.get_job(job_id)["status"] == "approved"


# ------------------------------------------------------ defect 3: log handlers --
def test_setup_logging_is_idempotent(tmp_path):
    import app as app_mod

    root_logger = logging.getLogger()
    saved = list(root_logger.handlers)
    try:
        for h in saved:
            root_logger.removeHandler(h)
        for _ in range(5):
            app_mod._setup_logging(str(tmp_path))
        tagged = [h for h in root_logger.handlers
                  if getattr(h, app_mod._LOG_HANDLER_TAG, None)]
        files = [h for h in tagged if isinstance(h, logging.FileHandler)]
        assert len(files) == 1, f"{len(files)} file handlers attached"
        assert len(tagged) == 2  # one file, one console
    finally:
        for h in list(root_logger.handlers):
            root_logger.removeHandler(h)
        for h in saved:
            root_logger.addHandler(h)


def test_setup_logging_writes_each_record_once(tmp_path):
    import app as app_mod

    root_logger = logging.getLogger()
    saved = list(root_logger.handlers)
    try:
        for h in saved:
            root_logger.removeHandler(h)
        for _ in range(3):
            app_mod._setup_logging(str(tmp_path))
        logging.getLogger("studio.test").warning("job 1 needs manual review")
        for h in root_logger.handlers:
            h.flush()
        text = (tmp_path / "logs" / "studio.log").read_text(encoding="utf-8")
        assert text.count("job 1 needs manual review") == 1
    finally:
        for h in list(root_logger.handlers):
            try:
                h.close()
            except Exception:
                pass
            root_logger.removeHandler(h)
        for h in saved:
            root_logger.addHandler(h)


# ----------------------------------------------- portable token paths (shared db) --
def test_a_relative_token_path_resolves_under_the_project_root(tmp_path):
    """`studio.db` is committed and shared between machines, so a channel row
    must not pin an absolute path from whichever machine ran the OAuth flow.

    Before this, `channel_upload_paths` returned `token_path` verbatim: a row
    written on one checkout carried an absolute `C:/Users/<someone>/.../tokens/UC....json`, which does
    not exist anywhere else, so `get_service` raised "Token ... is missing or
    invalid" and the uploader flipped the channel to needs_reauth even with a
    valid token sitting in `tokens/`.
    """
    from studio.oauth import channel_upload_paths

    cfg = {"_root": str(tmp_path)}
    secret, token = channel_upload_paths({"token_path": "tokens/UCabc.json"}, cfg)

    assert token == os.path.join(str(tmp_path), "tokens/UCabc.json")
    assert secret == os.path.join(str(tmp_path), "client_secret.json")


def test_an_absolute_token_path_is_still_honoured(tmp_path):
    """Existing rows and deliberate out-of-tree tokens keep working."""
    from studio.oauth import channel_upload_paths

    elsewhere = str(tmp_path / "elsewhere" / "tok.json")
    _secret, token = channel_upload_paths({"token_path": elsewhere}, {"_root": str(tmp_path)})

    assert token == elsewhere


def test_the_shipped_channel_rows_carry_no_absolute_paths():
    """Guard the committed database itself: an absolute path in studio.db is a
    machine-specific value that breaks every other checkout."""
    import sqlite3

    db = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "studio.db")
    if not os.path.exists(db):
        pytest.skip("studio.db not present in this checkout")
    con = sqlite3.connect(db)
    try:
        rows = con.execute("SELECT id, token_path, secret_path FROM channels").fetchall()
    finally:
        con.close()
    offenders = [
        (cid, p) for cid, tok, sec in rows for p in (tok, sec) if p and os.path.isabs(p)
    ]
    assert offenders == [], f"absolute paths in committed studio.db: {offenders}"
