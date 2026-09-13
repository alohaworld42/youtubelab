"""FEATURE: an operator answers a blocking gate and the pipeline moves on.

This is the app's reason to exist. The render pipeline writes a
`*.request.json` and then *blocks*, polling for a `*.response.json` beside it.
Every episode in the corpus is stopped on one. The studio's job is to let a
person answer it.

Every step below is exercised through the real app: the queue page a person
opens, the form that page renders, the route that form posts to, and the file
the pipeline is waiting on. Nothing is stubbed — if any link in that chain
breaks, these fail, and unit tests of the individual links would not.
"""
import json


def _pending_count(operator):
    return operator.get("/api/kidsong-review/count").get_json()["pending"]


# ============================================================ the happy path ===
def test_a_waiting_gate_shows_up_in_the_queue(operator, world):
    base = world.episode()
    world.shot_gate(base, "s00")

    page = operator.get("/kidsong-review").get_data(as_text=True)

    assert base in page, "the queue does not name the episode it is blocking"
    assert "Accept" in page and "Reject" in page
    assert _pending_count(operator) == 1


def test_accepting_writes_the_file_the_pipeline_is_waiting_for(operator, world):
    """The whole feature in one test: a click unblocks a render."""
    base = world.episode()
    req = world.shot_gate(base, "s00")
    assert world.answered(req) is None, "precondition: nothing answered yet"

    resp = operator.post("/kidsong-review/submit", data={
        "response_path": str(req).replace(".request.json", ".response.json"),
        "accept": "true",
    }, follow_redirects=True)

    assert resp.status_code == 200
    verdict = world.answered(req)
    assert verdict is not None, "Accept did not write the response file"
    assert verdict["accept"] is True
    assert _pending_count(operator) == 0, "the answered gate is still pending"


def test_an_answered_gate_leaves_the_queue(operator, world):
    base = world.episode()
    req = world.shot_gate(base, "s00")
    operator.post("/kidsong-review/submit", data={
        "response_path": str(req).replace(".request.json", ".response.json"),
        "accept": "true",
    })

    page = operator.get("/kidsong-review").get_data(as_text=True)
    assert "Nichts offen" in page, "the queue still lists a gate that was answered"


# ================================================================= rejecting ===
def test_rejecting_records_the_reasons_the_renderer_retries_on(operator, world):
    base = world.episode()
    req = world.shot_gate(base, "s00")

    operator.post("/kidsong-review/submit", data={
        "response_path": str(req).replace(".request.json", ".response.json"),
        "accept": "false",
        "reasons": ["cast integrity", "anatomy/skin-tone"],
        "other_notes": "the third child has four arms",
    })

    verdict = world.answered(req)
    assert verdict["accept"] is False
    assert "cast integrity" in verdict["reasons"]
    assert "anatomy/skin-tone" in verdict["reasons"]
    assert any("four arms" in str(r) for r in verdict["reasons"]) or \
        "four arms" in json.dumps(verdict), "the free-text note was dropped"


def test_ticking_a_reason_and_pressing_accept_resolves_as_reject(operator, world):
    """The checkbox group is labelled "Reject-Gründe". Ticking one and hitting
    Accept used to resolve as ACCEPT with the reasons attached as invisible
    noise — and the take shipped. The reasons win."""
    base = world.episode()
    req = world.shot_gate(base, "s00")

    operator.post("/kidsong-review/submit", data={
        "response_path": str(req).replace(".request.json", ".response.json"),
        "accept": "true",
        "reasons": ["wardrobe"],
    })

    assert world.answered(req)["accept"] is False


# ====================================================== refusing to be fooled ===
def test_an_already_answered_gate_is_never_overwritten(operator, world):
    """Two reviewers, or one double-click. The first verdict stands."""
    base = world.episode()
    req = world.shot_gate(base, "s00")
    target = str(req).replace(".request.json", ".response.json")

    operator.post("/kidsong-review/submit", data={"response_path": target, "accept": "false",
                                                  "reasons": ["cast integrity"]})
    first = world.answered(req)

    operator.post("/kidsong-review/submit", data={"response_path": target, "accept": "true"})

    assert world.answered(req) == first, "a second submit overwrote the first verdict"


def test_a_response_with_no_request_beside_it_is_refused(operator, world, tmp_path):
    """The path comes from a form field, so it is attacker-controlled. Writing
    a verdict where no gate exists would leave a file the pipeline never reads
    — or, worse, land outside output_dir."""
    stray = tmp_path / "output" / "not-a-gate.response.json"

    operator.post("/kidsong-review/submit",
                  data={"response_path": str(stray), "accept": "true"})

    assert not stray.exists()


def test_a_path_outside_the_output_directory_is_refused(operator, world, tmp_path):
    escape = tmp_path / "escaped.response.json"
    operator.post("/kidsong-review/submit",
                  data={"response_path": str(escape), "accept": "true"})
    assert not escape.exists()


# ============================================ the queue and the dashboard agree ==
def test_the_dashboard_points_at_the_same_gate_the_queue_lists(operator, world):
    """Two features, one truth. The dashboard counts a base's unanswered gates
    and the queue lists them; if they used different rules the operator would
    be told to go somewhere that has nothing waiting."""
    base = world.episode()
    world.cut_gate(base)

    dash = operator.get("/dashboard").get_data(as_text=True)
    assert "Wartet auf dich" in dash
    assert base in dash
    assert 'href="/kidsong-review"' in dash
    assert _pending_count(operator) == 1


def test_answering_the_gate_clears_it_from_the_dashboard_too(operator, world):
    base = world.episode()
    req = world.cut_gate(base)

    operator.post("/kidsong-review/submit", data={
        "response_path": str(req).replace(".request.json", ".response.json"),
        "accept": "true",
    })

    dash = operator.get("/dashboard").get_data(as_text=True)
    assert "Wartet auf dich" not in dash, (
        "the dashboard still says this episode is waiting on a human"
    )


def test_two_episodes_waiting_are_both_reachable(operator, world):
    a = world.episode("20260731-120000-kidsong-song-a")
    b = world.episode("20260731-130000-kidsong-song-b")
    world.cut_gate(a)
    world.shot_gate(b, "s01")

    page = operator.get("/kidsong-review").get_data(as_text=True)
    assert a in page and b in page
    assert _pending_count(operator) == 2
