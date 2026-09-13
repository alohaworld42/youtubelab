from flask import Blueprint, current_app, jsonify, render_template, request

from studio import ideas as ideas_mod
from studio import models, scheduler
from studio.views import VIDEO_TYPES, VIDEO_TYPE_IDS

bp = Blueprint("ideas", __name__)


@bp.route("/ideas")
def index():
    channels = models.list_channels()
    per_channel = {
        c["id"]: models.list_ideas(channel_id=c["id"], limit=100) for c in channels
    }
    return render_template(
        "ideas.html", channels=channels, per_channel=per_channel, types=VIDEO_TYPES
    )


@bp.route("/api/ideas", methods=["POST"])
def add():
    data = request.get_json(force=True)
    channel_id = data.get("channel_id")
    topic = (data.get("topic") or "").strip()
    if not channel_id or not models.get_channel(channel_id):
        return jsonify({"error": "Kanal wählen"}), 400
    if not topic:
        return jsonify({"error": "Thema fehlt"}), 400
    vt = data.get("video_type")
    if vt not in VIDEO_TYPE_IDS:
        vt = None
    idea_id = ideas_mod.add_user_idea(channel_id, topic, video_type=vt)
    if not idea_id:
        return jsonify({"error": "Duplikat — sehr ähnliche Idee existiert schon"}), 409
    scheduler.wake()
    return jsonify({"idea_id": idea_id})


@bp.route("/api/ideas/<int:idea_id>/delete", methods=["POST"])
def delete(idea_id):
    models.delete_idea(idea_id)
    return jsonify({"ok": True})


@bp.route("/api/ideas/generate", methods=["POST"])
def generate():
    data = request.get_json(force=True)
    channel = models.get_channel(data.get("channel_id"))
    if not channel:
        return jsonify({"error": "unbekannter Kanal"}), 404
    cfg = current_app.config["STUDIO_CFG"]
    try:
        n = ideas_mod.generate_ideas(channel, cfg)
    except Exception as e:
        return jsonify({"error": str(e)}), 502
    return jsonify({"inserted": n})
