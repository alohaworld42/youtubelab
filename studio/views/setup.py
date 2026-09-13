from flask import Blueprint, current_app, jsonify, render_template, request

from pipeline.config import load_config
from studio import setup as setup_mod

bp = Blueprint("setup", __name__)


def _cfg():
    return current_app.config["STUDIO_CFG"]


def _reload_cfg():
    current_app.config["STUDIO_CFG"] = load_config()
    return current_app.config["STUDIO_CFG"]


@bp.route("/setup")
def index():
    return render_template("setup.html", status=setup_mod.setup_status(_cfg()))


@bp.route("/api/setup/status")
def status():
    return jsonify(setup_mod.setup_status(_cfg()))


@bp.route("/api/setup/llm", methods=["POST"])
def set_llm():
    data = request.get_json(force=True)
    backend = (data.get("backend") or "").strip()
    api_key = (data.get("api_key") or "").strip() or None
    try:
        setup_mod.set_llm_backend(_cfg(), backend, api_key=api_key)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    cfg = _reload_cfg()
    return jsonify({"ok": True, "llm": setup_mod._check_llm(cfg)})


@bp.route("/api/setup/test-llm", methods=["POST"])
def test_llm():
    ok, detail = setup_mod.test_llm(_cfg())
    return jsonify({"ok": ok, "detail": detail})


@bp.route("/api/setup/keys", methods=["POST"])
def save_keys():
    data = request.get_json(force=True)
    updates = {}
    if (data.get("pexels") or "").strip():
        updates["PEXELS_API_KEY"] = data["pexels"].strip()
    if (data.get("pixabay") or "").strip():
        updates["PIXABAY_API_KEY"] = data["pixabay"].strip()
    if not updates:
        return jsonify({"error": "Kein Key übergeben"}), 400
    setup_mod.write_env(_cfg()["_root"], updates)
    return jsonify({"ok": True})


@bp.route("/api/setup/client-secret", methods=["POST"])
def upload_client_secret():
    f = request.files.get("file")
    if not f:
        return jsonify({"error": "Keine Datei hochgeladen"}), 400
    try:
        setup_mod.save_client_secret(_cfg()["_root"], f.read())
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"ok": True})


@bp.route("/api/setup/gameplay", methods=["POST"])
def download_gameplay():
    data = request.get_json(silent=True) or {}
    urls = None
    custom = (data.get("url") or "").strip()
    if custom:
        urls = [custom]
    started = setup_mod.start_gameplay_download(_cfg(), urls=urls)
    return jsonify({"started": started, **setup_mod.gameplay_download_status()})


@bp.route("/api/setup/gameplay/status")
def gameplay_status():
    return jsonify(setup_mod.gameplay_download_status())
