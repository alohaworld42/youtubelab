import os

from flask import Blueprint, jsonify, render_template, request

from studio import models, scheduler

bp = Blueprint("review", __name__)


@bp.route("/review")
def index():
    videos = models.videos_for_review()
    for v in videos:
        v["filename"] = os.path.basename(v["path"])
        thumb = v.get("thumb_path")
        v["thumb_filename"] = (
            os.path.basename(thumb) if thumb and os.path.exists(thumb) else None
        )
    return render_template("review.html", videos=videos)


@bp.route("/history")
def history():
    return render_template(
        "history.html",
        uploads=models.list_uploads(),
        failed=models.list_jobs(status="failed", limit=50),
    )


@bp.route("/api/videos/<int:video_id>/approve", methods=["POST"])
def approve(video_id):
    video = models.get_video(video_id)
    if not video:
        return jsonify({"error": "unbekanntes Video"}), 404
    job = models.get_job(video["job_id"])
    if not job or job["status"] != "ready":
        return jsonify({"error": "Job ist nicht im Review-Status"}), 409

    data = request.get_json(silent=True) or {}
    fields = {}
    if (data.get("title") or "").strip():
        fields["title"] = data["title"].strip()
    if "description" in data:
        fields["description"] = (data.get("description") or "").strip()
    if isinstance(data.get("tags"), list):
        fields["tags"] = [t.strip() for t in data["tags"] if t.strip()]
    if fields:
        models.update_video(video_id, **fields)

    models.set_job_status(job["id"], "approved")
    scheduler.wake()
    return jsonify({"ok": True})


@bp.route("/api/videos/<int:video_id>/reject", methods=["POST"])
def reject(video_id):
    video = models.get_video(video_id)
    if not video:
        return jsonify({"error": "unbekanntes Video"}), 404
    job = models.get_job(video["job_id"])
    if not job or job["status"] != "ready":
        return jsonify({"error": "Job ist nicht im Review-Status"}), 409

    data = request.get_json(silent=True) or {}
    models.set_job_status(job["id"], "rejected")

    new_job_id = None
    if data.get("regenerate"):
        # Re-queueing clones the old job's topic, so a topic that should never
        # have become a kids video can re-enter here even though every other
        # job-creating path now gates it. A rejected episode is exactly when
        # someone is most likely to hit "regenerate" on a bad subject.
        from studio.ideas import kidsong_topic_rejection

        if str(job["video_type"]).lower().startswith("kidsong"):
            reason = kidsong_topic_rejection(job["topic"])
            if reason:
                return jsonify({"error": reason}), 400

        new_job_id = models.create_job(
            channel_id=job["channel_id"],
            idea_id=job["idea_id"],
            video_type=job["video_type"],
            topic=job["topic"],
        )
        scheduler.wake()
    return jsonify({"ok": True, "new_job_id": new_job_id})
