from studio import models
from studio.scheduler import Scheduler

CFG = {"studio": {"max_attempts": 3, "tick_seconds": 1}}


def test_recover_requeues_transient_jobs(tmp_db):
    ch = models.create_channel("Test")
    job_id = models.create_job(channel_id=ch, video_type="facts")
    models.set_job_status(job_id, "rendering")

    Scheduler(CFG)._recover()

    job = models.get_job(job_id)
    assert job["status"] == "queued"
    assert job["attempts"] == 1


def test_recover_fails_job_after_max_attempts(tmp_db):
    ch = models.create_channel("Test")
    idea_id = models.add_idea(ch, "topic x", dedupe_hash="h")
    models.set_idea_status(idea_id, "in_progress")
    job_id = models.create_job(channel_id=ch, idea_id=idea_id, video_type="facts")
    models.update_job(job_id, status="rendering", attempts=2)

    Scheduler(CFG)._recover()

    assert models.get_job(job_id)["status"] == "failed"
    # idea released back to the queue
    assert models.next_pending_idea(ch)["id"] == idea_id


def test_recover_moves_uploading_back_to_approved(tmp_db):
    ch = models.create_channel("Test")
    job_id = models.create_job(channel_id=ch, video_type="facts")
    models.set_job_status(job_id, "uploading")

    Scheduler(CFG)._recover()

    assert models.get_job(job_id)["status"] == "approved"


def test_review_queue_shows_only_ready_jobs(tmp_db):
    ch = models.create_channel("Test")
    ready_job = models.create_job(channel_id=ch, video_type="facts")
    models.create_video(ready_job, ch, "out/a.mp4", "A", "", ["t"], 30)
    models.set_job_status(ready_job, "ready")

    done_job = models.create_job(channel_id=ch, video_type="facts")
    models.create_video(done_job, ch, "out/b.mp4", "B", "", ["t"], 30)
    models.set_job_status(done_job, "uploaded")

    videos = models.videos_for_review()
    assert len(videos) == 1
    assert videos[0]["title"] == "A"
