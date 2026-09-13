import os

from flask import Blueprint, current_app, jsonify, request, send_from_directory

from pipeline.config import abspath
from studio import cascade, models, scheduler
from studio import ideas as ideas_mod
from studio.views import STYLES, VIDEO_TYPE_IDS

bp = Blueprint("api", __name__)

RUNNING = set(models.TRANSIENT_JOB_STATES)


# ------------------------------------------------- legacy quick-generate API --
@bp.route("/generate", methods=["POST"])
def start_generate():
    data = request.get_json(force=True)
    video_type = data.get("video_type")
    if video_type not in VIDEO_TYPE_IDS:
        return jsonify({"error": "unknown video type"}), 400
    topic = (data.get("topic") or "").strip() or None
    channel_id = data.get("channel_id") or None
    if channel_id and not models.get_channel(channel_id):
        return jsonify({"error": "unbekannter Kanal"}), 400
    auto_approve = bool(data.get("upload")) and channel_id is not None
    style = data.get("style") or None
    if style not in STYLES:
        style = None
    if not style and video_type == "kids":
        style = "kids"  # kids scripts always get the kids look

    # Children's-content gate: a hand-typed topic reaches the kidsong renderer
    # unchanged, so it gets the same check as an auto-generated idea.
    if ideas_mod.is_kidsong_type(video_type):
        reason = ideas_mod.kidsong_topic_rejection(topic)
        if reason:
            return jsonify({
                "error": "Thema ist für ein Kinderlied nicht geeignet (%s). "
                         "Bitte ein einfaches Alltagsthema wählen "
                         "(Tiere, Zählen, Farben, Wetter, Teilen, Familie)." % reason
            }), 400

    job_id = models.create_job(
        channel_id=channel_id, video_type=video_type, topic=topic,
        auto_approve=auto_approve, style=style,
    )
    scheduler.wake()
    return jsonify({"job_id": job_id})


@bp.route("/status/<int:job_id>")
def status(job_id):
    job = models.get_job(job_id)
    if not job:
        return jsonify({"error": "unknown job"}), 404
    progress = [e["message"] for e in models.events_for_job(job_id)]

    if job["status"] == "queued":
        status_str = "queued"
        if not progress:
            progress = ["In Warteschlange — Worker startet gleich…"]
    elif job["status"] in RUNNING:
        status_str = "running"
    elif job["status"] in ("ready", "approved", "uploading", "uploaded"):
        status_str = "done"
    else:  # failed / rejected
        status_str = "error"

    payload = {"status": status_str, "progress": progress, "result": None,
               "error": job["error"]}
    if status_str == "done":
        video = models.get_video_by_job(job_id)
        if video:
            payload["result"] = {
                "filename": os.path.basename(video["path"]),
                "title": video["title"],
                "description": video["description"],
                "tags": video["tags"],
            }
    return jsonify(payload)


@bp.route("/video/<path:filename>")
def video(filename):
    cfg = current_app.config["STUDIO_CFG"]
    return send_from_directory(abspath(cfg, cfg["paths"]["output_dir"]), filename)


# ----------------------------------------------------------------- studio API --
@bp.route("/api/status")
def studio_status():
    current = models.current_job()
    current_payload = None
    if current:
        channel = models.get_channel(current["channel_id"]) if current["channel_id"] else None
        events = models.events_for_job(current["id"])
        current_payload = {
            "id": current["id"],
            "status": current["status"],
            "topic": current["topic"],
            "channel": channel["name"] if channel else None,
            "last_message": events[-1]["message"] if events else None,
        }
    return jsonify({
        "current_job": current_payload,
        "counts": {
            "queued": models.count_jobs("queued"),
            "ready": models.count_jobs("ready"),
            "failed": models.count_jobs("failed"),
            "uploaded": models.count_jobs("uploaded"),
            "uploads_today": models.uploads_today(),
            # 'approved' means "waiting for the uploader thread". It appeared in
            # no view at all, so a job blocked on a needs_reauth channel or on
            # the daily upload cap looked like it had simply vanished.
            "waiting_upload": models.count_jobs("approved") + models.count_jobs("uploading"),
        },
    })


@bp.route("/api/cascade", methods=["POST"])
def cascade_start():
    """One click: ensure the kidsong channel, top up ideas, queue N jobs.

    Synchronous on purpose — the only slow part is a single LLM completion
    (same as /api/ideas/generate, and the dashboard button disables itself
    while it runs). Rendering itself is NOT done here: the jobs land in the
    queue and the scheduler thread picks them up one at a time.
    """
    data = request.get_json(force=True, silent=True) or {}
    cfg = current_app.config["STUDIO_CFG"]
    try:
        summary = cascade.run_cascade(
            cfg, count=data.get("count", 3), channel_id=data.get("channel_id") or None
        )
    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500
    return jsonify(summary)


@bp.route("/api/jobs/<int:job_id>/retry", methods=["POST"])
def retry(job_id):
    job = models.get_job(job_id)
    if not job:
        return jsonify({"error": "unbekannter Job"}), 404
    if job["status"] != "failed":
        return jsonify({"error": "nur fehlgeschlagene Jobs"}), 409
    models.update_job(job_id, status="queued", attempts=0, error=None)
    scheduler.wake()
    return jsonify({"ok": True})
