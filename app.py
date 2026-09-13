"""
app.py — Brainrot Studio web UI + background worker.

Run:   python app.py
Then open http://127.0.0.1:5000 in your browser.

Pages: Dashboard, Kanäle (YouTube connect), Ideen, Review, Verlauf, Quick Generate.
The scheduler thread renders queued jobs and uploads approved videos.
"""
import json
import logging
import os
from logging.handlers import RotatingFileHandler

from flask import Flask

from pipeline.config import load_config
from studio.db import init_db
from studio.views import register_blueprints


_LOG_HANDLER_TAG = "_brainrot_studio_handler"

# Kept in sync with pipeline.kidsong.review_queue's suffixes — the review page
# writes exactly the files that scanner reports as pending.
_REQUEST_SUFFIX = ".request.json"
_RESPONSE_SUFFIX = ".response.json"
REVIEW_PAGE_SIZE = 25


def _setup_logging(root):
    """Attach exactly one file (+ console) handler to the root logger.

    Idempotent on purpose: create_app() runs at import AND again per test
    client / Flask reloader, and the old unconditional addHandler() meant
    every call stacked another RotatingFileHandler — one log record showed up
    5-7 times in logs/studio.log, which made a single warning look like a
    repeating failure while diagnosing.
    """
    logs_dir = os.path.join(root, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # drop handlers we installed on a previous call before installing fresh
    # ones (also handles a changed root dir); anything else stays untouched.
    for h in list(root_logger.handlers):
        if getattr(h, _LOG_HANDLER_TAG, None):
            root_logger.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass

    file_handler = RotatingFileHandler(
        os.path.join(logs_dir, "studio.log"),
        maxBytes=2 * 1024 * 1024, backupCount=3, encoding="utf-8",
    )
    file_handler.setFormatter(fmt)
    setattr(file_handler, _LOG_HANDLER_TAG, "file")
    root_logger.addHandler(file_handler)

    has_console = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in root_logger.handlers
    )
    if not has_console:
        console = logging.StreamHandler()
        console.setFormatter(fmt)
        setattr(console, _LOG_HANDLER_TAG, "console")
        root_logger.addHandler(console)


def _register_production_routes(app, cfg):
    """Kids-song production dashboard: browse finished videos and in-progress
    takes straight from the output folder (no DB involved). Also hosts the
    local human review queue (/kidsong-review), which replaces the
    token-expensive "Claude reviews every take via chat" loop with
    click-to-Accept/Reject — it only writes the same response.json files
    that pipeline.kidsong.review.poll_response() already polls for, so the
    orchestrator/review-gate contract is untouched.

    Note: this is deliberately NOT "/review" — studio/views/review.py
    already owns that path (the pre-existing "approve video before YouTube
    upload" page, registered via register_blueprints() before this
    function runs). Reusing it would create two GET handlers for the same
    URL rule."""
    from flask import redirect, render_template, request, send_from_directory, url_for

    from pipeline.config import abspath
    from pipeline.kidsong.review_queue import list_pending
    from pipeline.kidsong.status import scan_output

    output_dir = abspath(cfg, cfg["paths"]["output_dir"])
    os.makedirs(output_dir, exist_ok=True)

    def _resolve_in_output(relpath):
        """Resolve `relpath` under output_dir with a realpath containment
        check; returns the absolute path, or None if it escapes output_dir
        (e.g. "../../config.json") or is empty."""
        if not relpath:
            return None
        requested = os.path.realpath(os.path.join(output_dir, relpath))
        real_out = os.path.realpath(output_dir)
        try:
            if os.path.commonpath([requested, real_out]) != real_out:
                return None
        except ValueError:
            # Different drives on Windows (e.g. an absolute path escaping output_dir).
            return None
        return requested

    @app.route("/dashboard")
    def production_dashboard():
        return render_template(
            "production.html", data=scan_output(cfg), pending_count=len(list_pending(cfg))
        )

    @app.route("/artifact/<path:relpath>")
    def artifact(relpath):
        # Serve any file under output_dir (contact sheets, takes, wavs, finals)
        # for the dashboard. send_from_directory already blocks ".." segments,
        # but we double-check with a realpath containment test since this route
        # is reachable with an arbitrary path.
        if _resolve_in_output(relpath) is None:
            return "forbidden", 403
        return send_from_directory(output_dir, relpath)

    @app.route("/kidsong-review")
    def review_queue_view():
        pending = list_pending(cfg)
        # A backlog of several hundred takes rendered into one page means a
        # multi-megabyte document full of <img>/<video> that a phone cannot
        # scroll — and the reviewer only ever works the oldest few anyway.
        # Page it, oldest first (list_pending's own order).
        try:
            page_size = max(1, min(200, int(request.args.get("per_page", REVIEW_PAGE_SIZE))))
        except ValueError:
            page_size = REVIEW_PAGE_SIZE
        try:
            page = max(1, int(request.args.get("page", 1)))
        except ValueError:
            page = 1
        total = len(pending)
        pages = max(1, (total + page_size - 1) // page_size)
        page = min(page, pages)
        start = (page - 1) * page_size
        return render_template(
            "review_queue.html",
            pending=pending[start:start + page_size],
            total=total,
            page=page,
            pages=pages,
            per_page=page_size,
            notice=request.args.get("notice") or None,
        )

    @app.route("/api/kidsong-review/count")
    def review_queue_count():
        """Pending-take counter for the nav badge on every page."""
        from flask import jsonify

        try:
            return jsonify({"pending": len(list_pending(cfg))})
        except Exception:
            # A counter must never be the thing that breaks a page.
            return jsonify({"pending": 0})

    @app.route("/kidsong-review/submit", methods=["POST"])
    def review_submit():
        response_path = request.form.get("response_path", "")
        target = _resolve_in_output(response_path)
        if target is None or not target.endswith(_RESPONSE_SUFFIX):
            return "forbidden", 403
        # A verdict only ever belongs next to the request that asked for it.
        # Requiring the sibling request file keeps this route from being a
        # write-anything-under-output/ primitive, and rejects a stale tab whose
        # request has already been cleaned up.
        request_file = target[: -len(_RESPONSE_SUFFIX)] + _REQUEST_SUFFIX
        if not os.path.exists(request_file):
            return "forbidden", 403
        if os.path.exists(target):
            # Already answered — by another tab, by --watch, or by a double
            # submit from the auto-refresh. Never silently overwrite a verdict
            # the render loop may already have consumed.
            return redirect(url_for("review_queue_view", notice="already-answered"))

        accept = request.form.get("accept") == "true"

        reasons = [r for r in request.form.getlist("reasons") if r]
        # The checkbox group is labelled "reject reasons", so ticking one and
        # hitting Accept is a contradiction. It used to resolve as ACCEPT with
        # the reject reasons attached — the take shipped and the reasons became
        # invisible noise. The reasons win.
        if reasons and accept:
            accept = False
        other_notes = (request.form.get("other_notes") or "").strip()
        if other_notes:
            reasons.append(other_notes)

        score_raw = (request.form.get("score") or "").strip()
        try:
            score = float(score_raw) if score_raw else (1.0 if accept else 0.4)
        except ValueError:
            score = 1.0 if accept else 0.4
        # The input is type=number min=0 max=1, but nothing stops a hand-rolled
        # POST (or a browser that ignores the range) from sending 42 — and a
        # score outside 0-1 skews every downstream average that reads it.
        score = min(1.0, max(0.0, score))

        payload = {
            "accept": accept,
            "score": score,
            "reasons": reasons,
            "retry_hints": {
                "seed_bump": request.form.get("seed_bump") == "true",
                "simplify_action": request.form.get("simplify_action") == "true",
                "force_i2v": request.form.get("force_i2v") == "true",
            },
        }

        with open(target, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        return redirect(url_for(
            "review_queue_view",
            page=request.form.get("page") or None,
            notice="accepted" if accept else "rejected",
        ))


def create_app():
    cfg = load_config()
    _setup_logging(cfg["_root"])
    init_db()

    app = Flask(__name__)
    app.config["STUDIO_CFG"] = cfg
    from studio.humanize import install_filters

    install_filters(app)
    register_blueprints(app)
    _register_production_routes(app, cfg)
    # Password gate LAST so it can whitelist every registered endpoint. No-op
    # unless STUDIO_PASSWORD is set — needed once the studio is exposed through
    # a tunnel, since the pages themselves have no login.
    from studio.auth import install_auth

    install_auth(app)
    return app


app = create_app()


def _lan_addresses():
    """Best-effort list of this machine's LAN IPv4 addresses, for the banner.

    Uses a UDP connect to a public address to learn which local interface the
    OS would route through — no packet is actually sent, and it works without
    DNS. Falls back to hostname resolution, then to nothing: a missing banner
    line must never stop the server from starting.
    """
    import socket

    found = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            found.append(s.getsockname()[0])
        finally:
            s.close()
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith(("127.", "169.254.")) and ip not in found:
                found.append(ip)
    except OSError:
        pass
    return found


if __name__ == "__main__":
    if os.environ.get("STUDIO_NO_WORKER") != "1":
        from studio.scheduler import start_scheduler

        start_scheduler(app.config["STUDIO_CFG"])
    if os.environ.get("STUDIO_NO_BROWSER") != "1":
        import threading
        import webbrowser

        threading.Timer(1.5, lambda: webbrowser.open("http://127.0.0.1:5000")).start()

    # Bind on all interfaces by default so the review queue is reachable from a
    # phone on the same WLAN — reviewing takes on a small screen is the whole
    # point of the browser gate. Set STUDIO_PASSWORD (in .env) to require a login
    # before exposing this beyond the LAN (e.g. through a tunnel); set
    # STUDIO_HOST=127.0.0.1 to keep it machine-only.
    host = os.environ.get("STUDIO_HOST", "0.0.0.0")
    port = int(os.environ.get("STUDIO_PORT", "5000"))
    print(f"\n  Brainrot Studio running at  http://127.0.0.1:{port}")
    if host == "0.0.0.0":
        for _ip in _lan_addresses():
            print(f"  On this network (phone/tablet)  http://{_ip}:{port}")
    if (os.environ.get("STUDIO_PASSWORD") or "").strip():
        print("  Login: ON (password required)")
    else:
        print("  Login: OFF (no STUDIO_PASSWORD set — LAN-trusted, do NOT tunnel)")
    print()
    app.run(host=host, port=port, debug=False, use_reloader=False)
