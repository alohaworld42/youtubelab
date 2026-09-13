"""The UI is judged by what a person can see and reach, not by a 200.

Every page in this app already returned 200 and every template already rendered.
Opening them in a real browser said something else:

  * the production dashboard was **39,828px tall** — 44 screens, 5,353 DOM nodes
    — because all 32 in-progress episodes rendered their full shot table
    expanded, always;
  * the take-review queue was **23,410px on a 390px phone** (27.7 screens) and
    the Accept button of the first take sat **1,626px down**, two full screens
    below the video it judges — on the page whose entire reason to exist is
    reviewing takes on a phone;
  * the wrapping nav ate **165px** (a fifth of the screen) on every phone page;
  * a take's age rendered as ``wartet 18472s``;
  * and the dashboard showed a green **accept** badge for all 14 shots of an
    episode whose shots directory contains nothing but ``review/`` — the review
    log outlives the takes, so "everything approved" was shown for "nothing is
    there".

These tests hold the fixed shape. They assert on rendered HTML rather than on
pixels, because the pixel measurement needs a browser and this suite must stay
CPU-only — but each assertion names the pixel number it stands in for.
"""
import json
import os
import re

import pytest

from studio.humanize import human_duration


# ------------------------------------------------------------- human_duration --
@pytest.mark.parametrize("seconds,expected", [
    (0, "0 s"),
    (45, "45 s"),
    (60, "1 min"),
    (90, "1 min 30 s"),
    (3600, "1 h"),
    (18472, "5 h 7 min"),          # the literal value the queue was showing raw
    (86400, "1 T"),
    (90000, "1 T 1 h"),
])
def test_a_duration_reads_like_a_person_wrote_it(seconds, expected):
    assert human_duration(seconds) == expected


@pytest.mark.parametrize("bad", [None, "", "abc", float("nan"), float("inf"), object()])
def test_a_broken_timestamp_renders_empty_instead_of_exploding(bad):
    """A queue page must not 500 because one file has a strange mtime."""
    assert human_duration(bad) == ""


def test_a_clock_that_stepped_backwards_is_not_shown_as_negative():
    assert human_duration(-30) == "0 s"


# -------------------------------------------------------- the orphaned verdict --
def test_a_verdict_whose_take_is_gone_is_flagged_orphaned(tmp_path):
    """`review_log.json` is append-only and outlives the .mp4 files it judged.
    Reading a verdict without checking that its take still exists is what put a
    green accept on every shot of an empty episode."""
    from pipeline.kidsong.status import scan_output

    out = tmp_path / "output"
    base = "20260722-135241-kidsong-test"
    review = out / f"{base}-shots" / "review"
    review.mkdir(parents=True)
    (out / f"{base}-shots.json").write_text(json.dumps({"shots": [
        {"id": "s00", "shot_type": "wide", "action": "a"},
        {"id": "s01", "shot_type": "closeup", "action": "b"},
    ]}), encoding="utf-8")
    (review / "review_log.json").write_text(json.dumps([
        {"shot": "s00", "attempt": 0, "verdict": {"accept": True, "score": 1.0, "reasons": []}},
        {"shot": "s01", "attempt": 0, "verdict": {"accept": True, "score": 1.0, "reasons": []}},
    ]), encoding="utf-8")

    data = scan_output({"_root": str(tmp_path), "paths": {"output_dir": str(out)}})
    shots = data["in_progress"][0]["shots"]
    assert [s["take_count"] for s in shots] == [0, 0]
    assert all(s["verdict"]["orphaned"] for s in shots), (
        "a verdict with no take behind it must be marked orphaned"
    )


def test_a_verdict_with_its_take_still_on_disk_is_not_orphaned(tmp_path):
    from pipeline.kidsong.status import scan_output

    out = tmp_path / "output"
    base = "20260722-135241-kidsong-test"
    shots_dir = out / f"{base}-shots"
    (shots_dir / "review").mkdir(parents=True)
    (shots_dir / "s00_a0.mp4").write_bytes(b"\0" * 32)
    (out / f"{base}-shots.json").write_text(
        json.dumps({"shots": [{"id": "s00", "shot_type": "wide", "action": "a"}]}),
        encoding="utf-8")
    (shots_dir / "review" / "review_log.json").write_text(json.dumps([
        {"shot": "s00", "attempt": 0, "verdict": {"accept": True, "reasons": []}},
    ]), encoding="utf-8")

    data = scan_output({"_root": str(tmp_path), "paths": {"output_dir": str(out)}})
    shot = data["in_progress"][0]["shots"][0]
    assert shot["take_count"] == 1
    assert shot["verdict"]["orphaned"] is False


def test_a_reused_shot_is_not_called_orphaned(tmp_path):
    """A `reuse_of` shot legitimately has no take of its own — it borrows one.
    Flagging it "kein Take" would be a second wrong badge in place of the first."""
    from pipeline.kidsong.status import scan_output

    out = tmp_path / "output"
    base = "20260722-135241-kidsong-test"
    shots_dir = out / f"{base}-shots"
    (shots_dir / "review").mkdir(parents=True)
    (shots_dir / "s00_a0.mp4").write_bytes(b"\0" * 32)
    (out / f"{base}-shots.json").write_text(json.dumps({"shots": [
        {"id": "s00", "shot_type": "wide", "action": "a"},
        {"id": "s01", "shot_type": "wide", "action": "a", "reuse_of": "s00"},
    ]}), encoding="utf-8")
    (shots_dir / "review" / "review_log.json").write_text(json.dumps([
        {"shot": "s01", "attempt": 0, "verdict": {"accept": True, "reasons": []}},
    ]), encoding="utf-8")

    data = scan_output({"_root": str(tmp_path), "paths": {"output_dir": str(out)}})
    reused = [s for s in data["in_progress"][0]["shots"] if s["id"] == "s01"][0]
    assert reused["verdict"]["orphaned"] is False


def test_the_dashboard_never_renders_a_green_accept_for_a_missing_take(tmp_path):
    """End to end through the template: the exact defect, in HTML."""
    from jinja2 import Environment, FileSystemLoader

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = Environment(loader=FileSystemLoader(os.path.join(root, "templates")))
    env.filters["duration"] = human_duration
    tmpl = env.get_template("production.html")

    html = tmpl.render(
        data={"done": [], "in_progress": [{
            "base": "ep", "stage": "rendering", "label": "rendering shots (0/2 unique, 0 total takes)",
            "idle_seconds": 18472,
            "shots": [
                {"id": "s00", "shot_type": "wide", "action": "a", "take_count": 0,
                 "verdict": {"accept": True, "reasons": [], "orphaned": True}},
            ],
        }], "gpu": None, "logs": [], "comfy": None},
        pending_count=0,
    )
    assert "badge accept" not in html, "a take that does not exist was shown as accepted"
    assert "kein Take" in html
    assert "5 h 7 min" in html, "the idle age must be human-readable"


# ------------------------------------------- the shape that keeps pages usable --
def _template_text(name):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "templates", name), encoding="utf-8") as f:
        return f.read()


def test_the_dashboard_shot_tables_are_collapsed_by_default():
    """39,828px -> 6,577px. Only the newest still-running episode opens itself;
    keying it on stage=="rendering" was useless because that stage only means
    "has a shot list and no final cut", which was true of all 32."""
    card = _template_text("_production_card.html")
    page = _template_text("production.html")
    assert 'details class="shot-detail"' in card
    assert "{% if open_shots %} open{% endif %}" in card
    assert "{% set open_shots = loop.first %}" in page      # the running list
    assert "{% set open_shots = False %}" in page           # the blocked list
    assert "{% if v.stage == 'rendering' %} open{% endif %}" not in card


def test_episodes_blocked_on_a_human_get_their_own_section():
    """16 of 32 in-progress episodes were sitting on an unanswered gate, wearing
    the same RENDERING badge as the 16 with nothing to do. The only work on the
    page the operator could actually clear was the one thing it did not point
    at."""
    page = _template_text("production.html")
    card = _template_text("_production_card.html")
    assert "Wartet auf dich" in page
    assert "selectattr('awaiting_review')" in page
    assert "rejectattr('awaiting_review')" in page
    assert '<span class="stage blocked">' in card
    assert 'href="/kidsong-review"' in card


def test_the_stage_label_does_not_claim_work_is_happening():
    """"rendering shots (0/16 unique, 0 total takes)" sat directly under a badge
    saying the episode was waiting for a human — the same card contradicting
    itself. The label states a count now, not an activity."""
    import inspect

    from pipeline.kidsong import status

    src = inspect.getsource(status._scan_one)
    assert 'f"rendering shots (' not in src
    assert "unique shots rendered" in src


def test_the_pending_gate_count_matches_the_queue_rule(tmp_path):
    """The dashboard and the review queue must never disagree about what is
    waiting: a gate is pending exactly when the request exists and the response
    does not."""
    from pipeline.kidsong.status import _pending_gate_count

    d = tmp_path / "review"
    d.mkdir()
    (d / "cut.request.json").write_text("{}", encoding="utf-8")
    (d / "s00_a0.request.json").write_text("{}", encoding="utf-8")
    (d / "s00_a0.response.json").write_text("{}", encoding="utf-8")
    (d / "review_log.json").write_text("[]", encoding="utf-8")
    assert _pending_gate_count(str(d)) == 1
    assert _pending_gate_count(str(tmp_path / "nope")) == 0


def test_every_bulk_table_in_the_review_queue_is_collapsed():
    """The cut list, the shot list and the lyrics are reference data — they made
    the queue 27.7 phone screens and pushed the decision below the fold."""
    src = _template_text("review_queue.html")
    assert src.count('<details class="ref">') == 3


def test_the_decision_comes_before_the_reject_form_in_the_markup():
    """DOM order IS scroll order. The Accept/Reject row must appear ahead of the
    reason grid, or the button goes back below the fold (measured 1,626px)."""
    src = _template_text("review_queue.html")
    decision = src.index('class="form-row decision-row"')
    reasons = src.index('class="reason-grid"')
    assert decision < reasons


def test_the_reject_controls_stay_inside_the_same_form():
    """The buttons moved above the reason checkboxes; if that split them out of
    the <form>, ticking a reason would silently stop reaching the server and
    every reject would post as a bare reject with no reasons."""
    src = _template_text("review_queue.html")
    form_open = src.index('<form class="review-form"')
    form_close = src.index("</form>", form_open)
    body = src[form_open:form_close]
    for needle in ('name="accept" value="true"', 'name="accept" value="false"',
                   'name="reasons"', 'name="other_notes"', 'name="score"'):
        assert needle in body, f"{needle} fell outside the form"


def test_the_take_age_is_not_raw_seconds():
    src = _template_text("review_queue.html")
    assert '"%.0f"|format(item.age_seconds) }}s<' not in src
    assert "item.age_seconds|duration" in src


def test_the_phone_nav_is_one_scrolling_row():
    """165px of wrapped nav on a 844px screen, on every page."""
    src = _template_text("base.html")
    assert "@media (max-width:720px)" in src
    assert re.search(r"nav\{flex-wrap:nowrap;overflow-x:auto", src)


def test_keyboard_hints_are_hidden_on_touch_devices():
    assert "@media (pointer:coarse){.kbd-hint{display:none}}" in _template_text("base.html")
    assert 'class="kbd-hint"' in _template_text("review_queue.html")


def test_there_is_a_favicon_so_no_page_load_404s():
    src = _template_text("base.html")
    assert 'rel="icon"' in src
    assert "data:image/svg+xml" in src, "must be inline — an external file is another request"
