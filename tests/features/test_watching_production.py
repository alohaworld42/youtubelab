"""FEATURE: an operator opens the dashboard and learns what is happening.

The dashboard is a filesystem *inference* — renders run as detached processes,
so the page has no job state to read and must deduce a stage from which
artifacts exist. That makes it exactly the kind of feature where every unit
passes and the page still lies: `scan_output` can be perfect while the template
renders the wrong branch of it, and vice versa.

Each test here plants one real on-disk state and asserts what a person then
reads on the page.
"""


def _page(operator):
    return operator.get("/dashboard").get_data(as_text=True)


# ============================================================= the empty case ==
def test_an_empty_studio_says_so_instead_of_looking_broken(operator, world):
    page = _page(operator)
    assert "Noch nichts fertig" in page
    assert "Nichts in Arbeit" in page


# ============================================================== stage by stage ==
def test_a_planned_episode_is_listed_as_in_progress(operator, world):
    base = world.episode(shots=3)
    page = _page(operator)
    assert base in page
    assert "In Arbeit" in page


def test_a_finished_episode_moves_to_the_done_section(operator, world):
    base = world.episode()
    world.finished(base)

    page = _page(operator)
    assert "Fertig (1)" in page
    assert "Download" in page, "a finished episode must be downloadable"


def test_the_shot_count_a_person_scans_for_is_on_the_summary_line(operator, world):
    base = world.episode(shots=4)
    world.rendered_take(base, "s00")

    page = _page(operator)
    assert "4 Shots" in page
    assert "1 mit Take" in page, "the page does not say how many shots have a take"
    assert "3 offen" in page


# ================================================= the page must not overstate ==
def test_an_episode_with_no_takes_is_not_described_as_working(operator, world):
    """"rendering shots (0/4 unique, 0 total takes)" claimed work was happening
    on an episode nothing had touched for hours. The label states a count."""
    world.episode(shots=4)
    page = _page(operator)
    assert "rendering shots (" not in page
    assert "unique shots rendered" in page


def test_a_verdict_whose_take_was_deleted_is_not_shown_as_approved(operator, world):
    """`review_log.json` outlives the .mp4 files it judged. Every shot of an
    empty episode used to carry a green ACCEPT."""
    import json

    base = world.episode(shots=2)
    review = world.output_dir / f"{base}-shots" / "review"
    review.mkdir(parents=True, exist_ok=True)
    (review / "review_log.json").write_text(json.dumps([
        {"shot": "s00", "attempt": 0, "verdict": {"accept": True, "reasons": []}},
        {"shot": "s01", "attempt": 0, "verdict": {"accept": True, "reasons": []}},
    ]), encoding="utf-8")

    page = _page(operator)
    assert "badge accept" not in page, "a take that does not exist is shown as accepted"
    assert "kein Take" in page


def test_how_long_an_episode_has_been_idle_is_readable(operator, world):
    import os
    import time

    base = world.episode()
    old = time.time() - 18472
    for name in os.listdir(world.output_dir):
        p = world.output_dir / name
        if p.is_file() and base in name:
            os.utime(p, (old, old))

    page = _page(operator)
    assert "zuletzt aktiv vor" in page
    assert "18472s" not in page, "raw seconds reached the page"


# =========================================== the operator's next step is visible ==
def test_blocked_episodes_are_separated_from_the_ones_merely_running(operator, world):
    """Sixteen of the corpus's 32 in-progress episodes were blocked on a human
    and wore the same badge as the sixteen that had nothing to do."""
    blocked = world.episode("20260731-120000-kidsong-blocked")
    running = world.episode("20260731-130000-kidsong-running")
    world.cut_gate(blocked)

    page = _page(operator)
    assert "Wartet auf dich (1)" in page
    assert "In Arbeit (1)" in page
    assert page.index("Wartet auf dich") < page.index("In Arbeit (1)"), (
        "the work the operator can clear is below the work they cannot"
    )
    assert blocked in page and running in page


def test_a_blocked_card_links_straight_into_the_queue(operator, world):
    base = world.episode()
    world.cut_gate(base)
    page = _page(operator)
    assert "offene" in page and "Gate" in page
    assert 'href="/kidsong-review"' in page


# ============================================================== serving media ==
def test_a_rendered_take_can_be_played_back_from_the_dashboard(operator, world):
    base = world.episode()
    world.rendered_take(base, "s00")

    resp = operator.get(f"/artifact/{base}-shots/s00_a0.mp4")
    assert resp.status_code == 200
    assert resp.data == b"\0" * 64


def test_the_artifact_route_refuses_to_leave_the_output_directory(operator, world):
    """The path is user-controlled; `config.json` sits one level up."""
    for attack in ("../config.json", "..%2fconfig.json", "../../etc/passwd"):
        assert operator.get(f"/artifact/{attack}").status_code in (403, 404), attack


# =================================================================== resilience ==
def test_one_corrupt_shot_list_does_not_take_down_the_page(operator, world):
    """A half-written JSON file must cost its own card, not the dashboard."""
    good = world.episode("20260731-120000-kidsong-good")
    bad = world.episode("20260731-130000-kidsong-bad")
    (world.output_dir / f"{bad}-shots.json").write_text("{ this is not json",
                                                        encoding="utf-8")

    resp = operator.get("/dashboard")
    assert resp.status_code == 200
    assert good in resp.get_data(as_text=True)
