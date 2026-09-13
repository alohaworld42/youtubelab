"""The browser take-review gate (/kidsong-review + /kidsong-review/submit).

This route is the only writer of the `*.response.json` files that every
external review gate in `pipeline.kidsong` blocks on, so a wrong verdict here
ships a bad take (or stalls a render). These tests pin the parts that used to
be wrong or unguarded:

  * a ticked reject reason must not resolve as ACCEPT,
  * a score outside 0-1 must not reach the response file,
  * the path in the form must not be a write-anywhere-under-output primitive,
  * an already-answered gate must not be silently overwritten.

No network, no GPU: the app is built against a throwaway output dir.
"""
import json
import os

import pytest


@pytest.fixture()
def review_app(tmp_path, monkeypatch, tmp_db):
    """A studio app whose output_dir is an empty tmp dir."""
    import app as app_mod
    from pipeline.config import load_config

    out_dir = tmp_path / "output"
    out_dir.mkdir()

    real = load_config()
    cfg = dict(real)
    cfg["paths"] = dict(real["paths"], output_dir=str(out_dir))
    monkeypatch.setattr(app_mod, "load_config", lambda: cfg)

    flask_app = app_mod.create_app()
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as client:
        yield client, out_dir


def _write_request(out_dir, name="s01_a0", base="20260101-000000-kidsong-demo"):
    """Create a pending per-shot gate the way review.ExternalReviewer does."""
    review_dir = out_dir / f"{base}-shots" / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    request_path = review_dir / f"{name}.request.json"
    request_path.write_text(json.dumps({
        "shot_id": name.split("_")[0],
        "take": name,
        "shot": {"action": "waters a sunflower", "characters": ["Zuri"],
                 "shot_type": "medium", "camera": "static", "setting": "garden"},
        "heuristic": {"accept": True, "score": 0.9, "reasons": []},
    }), encoding="utf-8")
    response_rel = f"{base}-shots/review/{name}.response.json"
    return response_rel, review_dir / f"{name}.response.json"


# ---------------------------------------------------------------- verdicts ---
def test_accept_writes_an_accept_verdict(review_app):
    client, out_dir = review_app
    response_rel, response_path = _write_request(out_dir)

    res = client.post("/kidsong-review/submit",
                      data={"response_path": response_rel, "accept": "true"})
    assert res.status_code == 302

    body = json.loads(response_path.read_text(encoding="utf-8"))
    assert body["accept"] is True
    assert body["score"] == 1.0
    assert body["retry_hints"] == {"seed_bump": False, "simplify_action": False,
                                   "force_i2v": False}


def test_a_ticked_reject_reason_overrides_the_accept_button(review_app):
    """The checkbox group says "ein Haken = Reject". It used to be decoration:
    ticking "cast integrity" and hitting Accept shipped the take with the
    reject reason attached as invisible noise."""
    client, out_dir = review_app
    response_rel, response_path = _write_request(out_dir)

    client.post("/kidsong-review/submit", data={
        "response_path": response_rel, "accept": "true",
        "reasons": ["cast integrity"],
    })

    body = json.loads(response_path.read_text(encoding="utf-8"))
    assert body["accept"] is False
    assert "cast integrity" in body["reasons"]


def test_free_text_notes_do_not_block_an_accept(review_app):
    """Only the reason CHECKBOXES imply reject — a note stays a note."""
    client, out_dir = review_app
    response_rel, response_path = _write_request(out_dir)

    client.post("/kidsong-review/submit", data={
        "response_path": response_rel, "accept": "true",
        "other_notes": "slightly dark but fine",
    })

    body = json.loads(response_path.read_text(encoding="utf-8"))
    assert body["accept"] is True
    assert body["reasons"] == ["slightly dark but fine"]


def test_retry_hints_round_trip(review_app):
    client, out_dir = review_app
    response_rel, response_path = _write_request(out_dir)

    client.post("/kidsong-review/submit", data={
        "response_path": response_rel, "accept": "false",
        "reasons": ["action fidelity"], "seed_bump": "true", "force_i2v": "true",
    })

    body = json.loads(response_path.read_text(encoding="utf-8"))
    assert body["retry_hints"] == {"seed_bump": True, "simplify_action": False,
                                   "force_i2v": True}


# ------------------------------------------------------------------- score ---
@pytest.mark.parametrize("raw,expected", [
    ("0.75", 0.75), ("42", 1.0), ("-3", 0.0), ("", 1.0), ("not a number", 1.0),
    # float() accepts these happily; the clamp has to as well.
    ("nan", 0.0), ("inf", 1.0), ("-inf", 0.0),
])
def test_score_is_always_inside_zero_to_one(review_app, raw, expected):
    """The input is type=number min=0 max=1, but a hand-rolled POST is not.
    A score of 42 skews every downstream average that reads these files."""
    client, out_dir = review_app
    response_rel, response_path = _write_request(out_dir)

    client.post("/kidsong-review/submit",
                data={"response_path": response_rel, "accept": "true", "score": raw})

    body = json.loads(response_path.read_text(encoding="utf-8"))
    assert body["score"] == expected


# -------------------------------------------------------------- path guard ---
@pytest.mark.parametrize("bad_path", [
    "../config.json",                      # escapes output_dir
    "",                                    # empty
    "20260101-000000-kidsong-demo-shots/review/s01_a0.request.json",  # not a response
    "20260101-000000-kidsong-demo-shots/review/nope.response.json",   # no sibling request
])
def test_only_a_real_pending_gate_can_be_answered(review_app, bad_path):
    client, out_dir = review_app
    _write_request(out_dir)

    res = client.post("/kidsong-review/submit",
                      data={"response_path": bad_path, "accept": "true"})
    assert res.status_code == 403
    assert not (out_dir.parent / "config.json").exists()


def test_an_already_answered_gate_is_never_overwritten(review_app):
    """Two tabs, or a stale tab plus `auto_review --watch`: the second submit
    must not flip a verdict the render loop may already have consumed."""
    client, out_dir = review_app
    response_rel, response_path = _write_request(out_dir)

    client.post("/kidsong-review/submit",
                data={"response_path": response_rel, "accept": "true"})
    res = client.post("/kidsong-review/submit",
                      data={"response_path": response_rel, "accept": "false",
                            "reasons": ["wardrobe"]})

    assert res.status_code == 302
    assert "already-answered" in res.headers["Location"]
    assert json.loads(response_path.read_text(encoding="utf-8"))["accept"] is True


# -------------------------------------------------------------- queue page ---
def test_queue_page_lists_the_pending_gate(review_app):
    client, out_dir = review_app
    _write_request(out_dir)

    html = client.get("/kidsong-review").get_data(as_text=True)
    assert "s01_a0.response.json" in html
    assert "20260101-000000-kidsong-demo" in html
    # The destructive whole-page meta refresh is what wiped a half-filled
    # review form every 8 seconds; it must stay gone.
    assert "<meta http-equiv=\"refresh\"" not in html


def test_queue_page_pages_a_large_backlog(review_app):
    client, out_dir = review_app
    for i in range(30):
        _write_request(out_dir, name=f"s{i:02d}_a0")

    first = client.get("/kidsong-review").get_data(as_text=True)
    assert first.count("name=\"response_path\"") == 25
    second = client.get("/kidsong-review?page=2").get_data(as_text=True)
    assert second.count("name=\"response_path\"") == 5


def test_review_count_endpoint_feeds_the_nav_badge(review_app):
    client, out_dir = review_app
    assert client.get("/api/kidsong-review/count").get_json() == {"pending": 0}
    _write_request(out_dir)
    assert client.get("/api/kidsong-review/count").get_json() == {"pending": 1}


def test_answering_a_gate_removes_it_from_the_queue(review_app):
    client, out_dir = review_app
    response_rel, _ = _write_request(out_dir)

    client.post("/kidsong-review/submit",
                data={"response_path": response_rel, "accept": "true"})
    assert client.get("/api/kidsong-review/count").get_json()["pending"] == 0


def test_artifact_route_still_refuses_to_escape_output(review_app):
    client, out_dir = review_app
    assert client.get("/artifact/../config.json").status_code in (403, 404)
