"""Optional single-password gate for the whole studio.

Why this exists: the pages have no per-user accounts — the app was built to
trust the local network (see app.py's bind-on-0.0.0.0 note). The moment it is
reachable from outside the LAN (a Cloudflare quick tunnel, ngrok, …) that trust
is gone: anyone with the URL could trigger renders or upload to YouTube with the
stored OAuth tokens. This adds one shared password in front of *every* route so
a tunnel can be exposed safely.

Activation is opt-in and controlled entirely by the STUDIO_PASSWORD env var
(read from .env via pipeline.config.load_dotenv, which runs inside
load_config()):

  * STUDIO_PASSWORD empty/unset  -> gate is a no-op. Local LAN use and the test
    suite behave exactly as before this file existed.
  * STUDIO_PASSWORD set          -> every request needs a valid session cookie
    or it is redirected to /login. One password, shared, "is this me?".

The Flask session secret comes from STUDIO_SECRET_KEY; if that is missing we
generate one once and persist it to .env so logins survive a server restart.
"""
import hmac
import os

from flask import (
    Blueprint,
    redirect,
    render_template_string,
    request,
    session,
    url_for,
)

bp = Blueprint("auth", __name__)

# Endpoints reachable WITHOUT a session. Everything else is gated.
_PUBLIC_ENDPOINTS = {"auth.login", "auth.logout", "static"}


def _password():
    return (os.environ.get("STUDIO_PASSWORD") or "").strip()


def _ensure_secret_key(root):
    """Return a stable Flask session secret.

    Prefers STUDIO_SECRET_KEY from the environment. If absent, mint one and
    append it to .env so it is reused on the next start (otherwise every restart
    would silently invalidate all logins). Falls back to an in-memory key if the
    file cannot be written — logins then reset on restart, which is annoying but
    never fatal.
    """
    existing = (os.environ.get("STUDIO_SECRET_KEY") or "").strip()
    if existing:
        return existing

    import secrets

    key = secrets.token_hex(32)
    os.environ["STUDIO_SECRET_KEY"] = key
    try:
        env_path = os.path.join(root, ".env")
        with open(env_path, "a", encoding="utf-8") as f:
            f.write(f"\nSTUDIO_SECRET_KEY={key}\n")
    except OSError:
        pass
    return key


def _safe_next(target):
    """Only allow same-site relative redirect targets (blocks open-redirect via
    ?next=https://evil.example). Must start with a single '/'."""
    if target and target.startswith("/") and not target.startswith("//"):
        return target
    return url_for("production_dashboard") if _has_endpoint("production_dashboard") else "/"


def _has_endpoint(name):
    from flask import current_app

    return name in current_app.view_functions


_LOGIN_HTML = """<!DOCTYPE html>
<html lang="de"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Login — Brainrot Studio</title>
<style>
  :root{--bg:#0d0d11;--surface:#16161d;--surface2:#1d1d27;--line:#2a2a36;
    --text:#f5f5f7;--muted:#8b8b97;--violet:#7c3aed;--violet-bright:#9d5cff;
    --acid:#ffe600;--err:#ff7a7a;}
  *{box-sizing:border-box}
  body{margin:0;min-height:100vh;display:flex;align-items:center;justify-content:center;
    background:var(--bg);color:var(--text);
    font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;padding:20px}
  .card{background:var(--surface);border:1px solid var(--line);border-radius:16px;
    padding:28px 24px;width:100%;max-width:360px}
  .logo{font-size:22px;text-transform:uppercase;letter-spacing:-.02em;margin:0 0 4px;
    font-family:"Arial Black",Impact,system-ui,sans-serif}
  .logo b{color:var(--acid)}
  p.sub{color:var(--muted);font-size:13px;margin:0 0 20px}
  label{display:block;font-size:12px;text-transform:uppercase;letter-spacing:.1em;
    color:var(--muted);margin:0 0 6px}
  input[type=password]{width:100%;background:var(--surface2);border:2px solid var(--line);
    color:var(--text);border-radius:10px;padding:12px;font-size:16px;font-family:inherit}
  input:focus{outline:none;border-color:var(--violet)}
  button{width:100%;margin-top:16px;border:none;border-radius:10px;cursor:pointer;
    font-weight:700;font-size:15px;padding:12px;background:var(--violet);color:#fff}
  button:hover{background:var(--violet-bright)}
  .err{color:var(--err);font-size:13px;margin:12px 0 0}
</style></head>
<body>
  <form class="card" method="post" action="{{ url_for('auth.login') }}">
    <h1 class="logo">Brainrot <b>Studio</b></h1>
    <p class="sub">Bitte Passwort eingeben.</p>
    <input type="hidden" name="next" value="{{ next_target }}">
    <label for="pw">Passwort</label>
    <input id="pw" type="password" name="password" autofocus autocomplete="current-password">
    {% if error %}<p class="err">{{ error }}</p>{% endif %}
    <button type="submit">Anmelden</button>
  </form>
</body></html>"""


@bp.route("/login", methods=["GET", "POST"])
def login():
    pw = _password()
    if not pw:
        # Gate disabled — nothing to log into. Send them home.
        return redirect(_safe_next(request.args.get("next")))

    error = None
    if request.method == "POST":
        supplied = (request.form.get("password") or "")
        # constant-time compare so a wrong password's length/prefix does not leak
        if hmac.compare_digest(supplied, pw):
            session["authed"] = True
            session.permanent = True
            return redirect(_safe_next(request.form.get("next")))
        error = "Falsches Passwort."

    next_target = request.values.get("next") or ""
    return render_template_string(_LOGIN_HTML, error=error, next_target=next_target)


@bp.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("auth.login"))


def install_auth(app):
    """Register the login/logout routes and, when a password is configured,
    the before_request gate. Call AFTER all other blueprints/routes so the
    gate can whitelist by endpoint name and _safe_next() can resolve the
    dashboard endpoint."""
    from datetime import timedelta

    app.register_blueprint(bp)

    if not _password():
        return  # opt-in: no password -> no gate (local/LAN/tests unchanged)

    root = app.config.get("STUDIO_CFG", {}).get("_root") or os.getcwd()
    app.secret_key = _ensure_secret_key(root)
    app.permanent_session_lifetime = timedelta(days=30)
    app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax")

    @app.before_request
    def _gate():
        # This used to open for any app with app.config["TESTING"] set, so that
        # web-layer tests building the real create_app() would not be redirected
        # when the developer's .env happened to carry a STUDIO_PASSWORD. That is
        # a test concern solved in production security code, and it cost two
        # things:
        #
        #   1. a configured password was VOID on any app with TESTING set —
        #      "production never does" is a convention, not a guarantee, and the
        #      whole point of this gate is to make a tunnel safe;
        #   2. the gate became untestable. No test could check the real app,
        #      because using the test client turned the gate off — which is
        #      exactly why tests/test_auth_gate.py had to assert against a stub
        #      app with two fake routes instead.
        #
        # The gate is opt-in already (install_auth returns early with no
        # password), so a test that does not want it simply does not set one.
        # tests/conftest.py clears STUDIO_PASSWORD for the suite, which also
        # makes those tests independent of the developer's environment.
        if session.get("authed"):
            return None
        if request.endpoint in _PUBLIC_ENDPOINTS:
            return None
        # Static files (favicon etc.) also carry no endpoint sometimes.
        if request.endpoint is None:
            return None
        return redirect(url_for("auth.login", next=request.full_path))
