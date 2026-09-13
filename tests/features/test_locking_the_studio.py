"""FEATURE: the operator puts a password on the studio and it actually holds.

`tests/test_auth_gate.py` covers this — against a bare Flask app with two stub
routes. That proves `install_auth` works on a toy; it cannot prove the real
studio is covered, because `create_app()` registers its blueprints FIRST and
installs the gate LAST so the gate can whitelist every registered endpoint.
A route added to a blueprint after that ordering was written would be a hole
no unit test can see.

So these walk the real route table: whatever the app actually serves must be
behind the password, and the login must actually open it.
"""
import pytest


@pytest.fixture()
def locked(studio, monkeypatch):
    """The same real app, rebuilt with a password set."""
    monkeypatch.setenv("STUDIO_PASSWORD", "hunter2")
    import app as app_module

    rebuilt = app_module.create_app()
    rebuilt.config["TESTING"] = True
    return rebuilt


def _protected_get_routes(flask_app):
    """Every GET route the app really serves, minus the ones a locked-out
    visitor must still reach (the login form itself, and static files)."""
    skip = {"auth.login", "auth.logout", "static"}
    out = []
    for rule in flask_app.url_map.iter_rules():
        if rule.endpoint in skip or "GET" not in rule.methods:
            continue
        if any(c in rule.rule for c in "<>"):
            continue  # needs params; covered by the artifact test below
        out.append(rule.rule)
    return sorted(out)


# =========================================== no password means no interruption ==
def test_without_a_password_every_page_is_open(operator, studio):
    for path in _protected_get_routes(studio):
        assert operator.get(path).status_code != 302 or \
            "/login" not in operator.get(path).headers.get("Location", ""), path


# ================================================= with a password, everything ==
def test_every_page_the_app_serves_is_behind_the_password(locked):
    """Enumerated from the live route table, so a blueprint route added later
    is covered the day it is added."""
    client = locked.test_client()
    unprotected = []
    for path in _protected_get_routes(locked):
        resp = client.get(path)
        if resp.status_code != 302 or "/login" not in resp.headers.get("Location", ""):
            unprotected.append((path, resp.status_code))
    assert not unprotected, f"reachable without logging in: {unprotected}"


def test_the_media_route_is_behind_the_password_too(locked, world):
    """`/artifact/<path>` serves every rendered frame and take. It takes a URL
    parameter, so the route-table sweep above skips it — check it by hand."""
    base = world.episode()
    world.rendered_take(base, "s00")

    resp = locked.test_client().get(f"/artifact/{base}-shots/s00_a0.mp4")
    assert resp.status_code == 302
    assert "/login" in resp.headers["Location"]


def test_the_json_endpoints_are_behind_the_password(locked):
    """A counter or status endpoint leaking whether work exists is a smaller
    leak than a page, but it is still a leak."""
    client = locked.test_client()
    for path in ("/api/kidsong-review/count", "/api/setup/status", "/api/status"):
        resp = client.get(path)
        assert resp.status_code == 302, f"{path} answered without a login"


# ======================================================= logging in and out ====
def test_the_right_password_opens_the_studio(locked):
    client = locked.test_client()
    resp = client.post("/login", data={"password": "hunter2"}, follow_redirects=False)
    assert resp.status_code == 302, "a correct password did not log the operator in"

    assert client.get("/dashboard").status_code == 200


def test_the_wrong_password_does_not(locked):
    client = locked.test_client()
    client.post("/login", data={"password": "nope"})
    resp = client.get("/dashboard")
    assert resp.status_code == 302 and "/login" in resp.headers["Location"]


def test_logging_out_closes_it_again(locked):
    client = locked.test_client()
    client.post("/login", data={"password": "hunter2"})
    assert client.get("/dashboard").status_code == 200

    client.get("/logout")
    resp = client.get("/dashboard")
    assert resp.status_code == 302 and "/login" in resp.headers["Location"]


def test_a_logged_in_operator_can_answer_a_gate(locked, world):
    """The gate must not lock the operator out of the one job they came for."""
    base = world.episode()
    req = world.shot_gate(base, "s00")

    client = locked.test_client()
    client.post("/login", data={"password": "hunter2"})
    client.post("/kidsong-review/submit", data={
        "response_path": str(req).replace(".request.json", ".response.json"),
        "accept": "true",
    })

    assert world.answered(req) is not None


def test_a_logged_out_visitor_cannot_answer_a_gate(locked, world):
    base = world.episode()
    req = world.shot_gate(base, "s00")

    locked.test_client().post("/kidsong-review/submit", data={
        "response_path": str(req).replace(".request.json", ".response.json"),
        "accept": "true",
    })

    assert world.answered(req) is None, "an anonymous POST approved a take"
