"""Unit tests for pipeline.kidsong.review.vision_coverage (QM-010).

A take counts as vision-covered iff `<review_dir>/<take>.response.json`
exists AND its `accept` is true — regardless of when the response file was
written (a late backfill from the auto-reviewer or the browser queue counts
exactly the same as one that landed during the run; that is the intentional
recovery path for the generate.py promote gate this feeds). A take whose only
verdict is the in-run heuristic fallback has no response file at all and is
uncovered; a take with a response that itself rejected it is also uncovered.

No GPU/network — plain filesystem fixtures under tmp_path.
"""
import json
import os

from pipeline.kidsong.review import vision_coverage


def _write_response(review_dir, take, accept, **extra):
    os.makedirs(review_dir, exist_ok=True)
    payload = {"accept": accept, "score": 0.9 if accept else 0.1, "reasons": []}
    payload.update(extra)
    with open(os.path.join(review_dir, f"{take}.response.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f)


def test_empty_review_dir_all_uncovered(tmp_path):
    review_dir = str(tmp_path / "review")  # never created
    coverage, uncovered = vision_coverage(review_dir, ["s00_a0", "s01_a0"])
    assert coverage == 0.0
    assert uncovered == ["s00_a0", "s01_a0"]


def test_empty_accepted_takes_is_vacuously_covered(tmp_path):
    review_dir = str(tmp_path / "review")
    coverage, uncovered = vision_coverage(review_dir, [])
    assert coverage == 1.0
    assert uncovered == []


def test_all_covered(tmp_path):
    review_dir = str(tmp_path / "review")
    _write_response(review_dir, "s00_a0", True)
    _write_response(review_dir, "s01_a0", True)
    coverage, uncovered = vision_coverage(review_dir, ["s00_a0", "s01_a0"])
    assert coverage == 1.0
    assert uncovered == []


def test_mixed_covered_and_uncovered(tmp_path):
    review_dir = str(tmp_path / "review")
    _write_response(review_dir, "s00_a0", True)
    # s01_a0 has no response file at all -- heuristic-only fallback.
    coverage, uncovered = vision_coverage(review_dir, ["s00_a0", "s01_a0"])
    assert coverage == 0.5
    assert uncovered == ["s01_a0"]


def test_response_exists_but_rejected_is_uncovered(tmp_path):
    """A response that itself said accept=false must not count as coverage,
    even though the take may still have been used to cover the shot's slot
    (e.g. a shot that exhausted its retry budget)."""
    review_dir = str(tmp_path / "review")
    _write_response(review_dir, "s00_a0", False)
    coverage, uncovered = vision_coverage(review_dir, ["s00_a0"])
    assert coverage == 0.0
    assert uncovered == ["s00_a0"]


def test_late_backfilled_response_counts_as_covered(tmp_path):
    """The whole point of the recovery path: a response written well after
    the run ended (auto-reviewer catching up, or a human answering the
    browser queue later) must count identically to one written during the
    run -- vision_coverage only ever reads what's on disk NOW."""
    review_dir = str(tmp_path / "review")
    # Simulate "run ended without an answer", then a backfill arriving later.
    coverage, uncovered = vision_coverage(review_dir, ["s00_a0"])
    assert coverage == 0.0
    assert uncovered == ["s00_a0"]

    _write_response(review_dir, "s00_a0", True)
    coverage, uncovered = vision_coverage(review_dir, ["s00_a0"])
    assert coverage == 1.0
    assert uncovered == []


def test_response_for_a_different_attempt_is_not_counted(tmp_path):
    """A response filed under a different take id (e.g. a retried attempt)
    must not be mistaken for coverage of the take actually used in the cut."""
    review_dir = str(tmp_path / "review")
    _write_response(review_dir, "s00_a0", True)  # an earlier, discarded attempt
    # The cut actually used attempt 1's take.
    coverage, uncovered = vision_coverage(review_dir, ["s00_a1"])
    assert coverage == 0.0
    assert uncovered == ["s00_a1"]


def test_duplicate_takes_in_accepted_list_are_weighted(tmp_path):
    """A shot whose slot is covered by ANOTHER shot's render (exhausted
    budget / no usable take) appears twice in the takes-actually-used list;
    an uncovered shared take should count against coverage for each shot
    using it, not just once."""
    review_dir = str(tmp_path / "review")
    coverage, uncovered = vision_coverage(review_dir, ["s00_a0", "s00_a0", "s01_a0"])
    assert coverage == 0.0
    assert uncovered == ["s00_a0", "s00_a0", "s01_a0"]

    _write_response(review_dir, "s00_a0", True)
    coverage, uncovered = vision_coverage(review_dir, ["s00_a0", "s00_a0", "s01_a0"])
    assert round(coverage, 4) == round(2 / 3, 4)
    assert uncovered == ["s01_a0"]


def test_unreadable_response_json_counts_as_uncovered(tmp_path):
    review_dir = str(tmp_path / "review")
    os.makedirs(review_dir, exist_ok=True)
    with open(os.path.join(review_dir, "s00_a0.response.json"), "w", encoding="utf-8") as f:
        f.write("{not valid json")
    coverage, uncovered = vision_coverage(review_dir, ["s00_a0"])
    assert coverage == 0.0
    assert uncovered == ["s00_a0"]
