"""The re-render path must not smuggle an unsuitable topic back in.

Every other path that creates a kidsong job gates the topic, but /reject
with regenerate=true CLONES the rejected job's topic. That is precisely the
moment someone is most likely to retry a bad subject — a rejected episode.
"""
import os
import tempfile

import pytest


@pytest.fixture()
def app_client(monkeypatch):
    tmp = tempfile.mkdtemp()
    monkeypatch.setenv("STUDIO_DB_PATH", os.path.join(tmp, "studio.db"))
    monkeypatch.setenv("STUDIO_NO_WORKER", "1")
    monkeypatch.setenv("STUDIO_NO_BROWSER", "1")

    from studio import db

    db.reset_for_tests()
    db.init_db()

    import app as app_module

    application = app_module.create_app()
    application.config["TESTING"] = True

    # The requeue path wakes the scheduler; there is none in tests.
    from studio.views import review as review_view

    monkeypatch.setattr(review_view.scheduler, "wake", lambda *a, **k: None)

    with application.test_client() as client:
        yield client
    db.reset_for_tests()


def _rejected_job(topic, video_type="kidsong"):
    """A kidsong job sitting in `ready` with a video, ready to be rejected."""
    from studio import models

    channel_id = models.create_channel(name="Kidsong")
    models.update_channel(channel_id, default_video_type=video_type, status="active")
    job_id = models.create_job(
        channel_id=channel_id, idea_id=None, video_type=video_type, topic=topic
    )
    models.set_job_status(job_id, "ready")
    video_id = models.create_video(
        job_id=job_id, channel_id=channel_id, path="x.mp4",
        title="t", description="d", tags=[], duration=60.0,
    )
    return job_id, video_id


UNSUITABLE = [
    "Germany's scariest folkloric creatures come to life",
    "People share the weirdest things they thought were normal growing up in Germany",
    "Bizarre German festivals that will leave you scratching your head",
]


@pytest.mark.parametrize("topic", UNSUITABLE)
def test_regenerate_refuses_an_unsuitable_topic(app_client, topic):
    from studio import models

    job_id, video_id = _rejected_job(topic)
    r = app_client.post(f"/api/videos/{video_id}/reject", json={"regenerate": True})

    assert r.status_code == 400, r.get_data(as_text=True)
    assert r.get_json().get("error")
    # The rejection itself must still have taken effect...
    assert models.get_job(job_id)["status"] == "rejected"
    # ...but no replacement job may exist.
    assert models.count_jobs(status="queued") == 0


def test_regenerate_allows_a_wholesome_topic(app_client):
    from studio import models

    job_id, video_id = _rejected_job("Washing Little Hands")
    r = app_client.post(f"/api/videos/{video_id}/reject", json={"regenerate": True})

    assert r.status_code == 200, r.get_data(as_text=True)
    assert r.get_json()["new_job_id"]
    assert models.count_jobs(status="queued") == 1


def test_reject_without_regenerate_is_unaffected(app_client):
    """A plain reject of a bad topic must still work — nothing new is created."""
    from studio import models

    job_id, video_id = _rejected_job(UNSUITABLE[0])
    r = app_client.post(f"/api/videos/{video_id}/reject", json={})

    assert r.status_code == 200
    assert models.get_job(job_id)["status"] == "rejected"
    assert models.count_jobs(status="queued") == 0


def test_non_kidsong_type_is_not_gated(app_client):
    """A facts channel may legitimately regenerate a facts topic."""
    from studio import models

    job_id, video_id = _rejected_job(UNSUITABLE[0], video_type="facts")
    r = app_client.post(f"/api/videos/{video_id}/reject", json={"regenerate": True})

    assert r.status_code == 200, r.get_data(as_text=True)
    assert models.count_jobs(status="queued") == 1
