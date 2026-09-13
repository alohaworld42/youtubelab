"""The optional password gate (studio.auth.install_auth)."""
import pytest
from flask import Flask

from studio.auth import install_auth


def _app(monkeypatch, tmp_path, password):
    """Bare app with one protected route, wired the way create_app() wires it."""
    monkeypatch.setenv("STUDIO_SECRET_KEY", "test-secret-not-random")
    if password is None:
        monkeypatch.delenv("STUDIO_PASSWORD", raising=False)
    else:
        monkeypatch.setenv("STUDIO_PASSWORD", password)

    app = Flask(__name__)
    app.config["STUDIO_CFG"] = {"_root": str(tmp_path)}

    @app.route("/")
    def production_dashboard():  # name matters: _safe_next() default target
        return "home"

    @app.route("/secret")
    def secret():
        return "top secret"

    install_auth(app)
    return app


def test_no_password_means_no_gate(monkeypatch, tmp_path):
    app = _app(monkeypatch, tmp_path, password=None)
    client = app.test_client()
    assert client.get("/secret").status_code == 200


def test_unauthed_request_redirects_to_login(monkeypatch, tmp_path):
    app = _app(monkeypatch, tmp_path, password="hunter2")
    resp = app.test_client().get("/secret")
    assert resp.status_code == 302
    assert "/login" in resp.headers["Location"]


def test_wrong_password_is_rejected(monkeypatch, tmp_path):
    app = _app(monkeypatch, tmp_path, password="hunter2")
    resp = app.test_client().post("/login", data={"password": "nope"})
    assert resp.status_code == 200
    assert "Falsches Passwort" in resp.get_data(as_text=True)


def test_correct_password_grants_access(monkeypatch, tmp_path):
    app = _app(monkeypatch, tmp_path, password="hunter2")
    client = app.test_client()
    login = client.post("/login", data={"password": "hunter2", "next": "/secret"})
    assert login.status_code == 302
    assert login.headers["Location"].endswith("/secret")
    assert client.get("/secret").get_data(as_text=True) == "top secret"


def test_next_open_redirect_is_blocked(monkeypatch, tmp_path):
    app = _app(monkeypatch, tmp_path, password="hunter2")
    resp = app.test_client().post(
        "/login", data={"password": "hunter2", "next": "https://evil.example/x"}
    )
    # must fall back to a local target, never the external URL
    assert resp.status_code == 302
    assert "evil.example" not in resp.headers["Location"]


def test_login_page_renders(monkeypatch, tmp_path):
    app = _app(monkeypatch, tmp_path, password="hunter2")
    body = app.test_client().get("/login").get_data(as_text=True)
    assert "Passwort" in body
