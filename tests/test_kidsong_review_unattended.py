"""Tests for ExternalReviewer's adaptive unattended detection (problem 3):
after N consecutive external-review timeouts in one run, stop blocking the
GPU on every remaining shot and go straight to the heuristic verdict — but
resume waiting if a human answers a backlogged request.

No GPU/network/real ComfyUI clip needed: `_heuristic.review` is monkeypatched
to always accept (stage 1 heuristics are untouched by this feature and are
covered by their own tests), and `contact_sheet`/request writing just touch
the filesystem under tmp_path.
"""
import json
import os
import time

from pipeline.kidsong.review import ExternalReviewer, _EMPTY_HINTS


def _reviewer(unattended_after=2, timeout=0.05, poll_seconds=0.01):
    r = ExternalReviewer(cfg={"kidsong": {"review": {
        "unattended_after_timeouts": unattended_after,
        "external_timeout": timeout,
        # This suite exercises the queue/wait state machine, which only engages
        # when a shot is actually queued. Disable the shot-level auto-accept
        # prefilter so a clean heuristic still writes a request and waits.
        "shot_auto_accept": {"enabled": False},
    }}})
    # Hardcoded in __init__ for production; override here so timeout tests
    # stay fast regardless.
    r.timeout = timeout
    r.poll_seconds = poll_seconds
    r._heuristic.review = lambda shot, video_path: {
        "accept": True, "score": 1.0, "reasons": [], "retry_hints": dict(_EMPTY_HINTS),
    }
    return r


def _shot(i):
    return {"id": f"s{i:02d}", "characters": ["all"]}


def _video_path(tmp_path, i, attempt=0):
    # Doesn't need to exist: contact_sheet() falls back to a blank placeholder
    # frame when the video can't be opened.
    return str(tmp_path / f"s{i:02d}_a{attempt}.mp4")


# ------------------------------------------------------------- engagement --
def test_engages_unattended_after_n_consecutive_timeouts(tmp_path):
    r = _reviewer(unattended_after=2)
    out_dir = str(tmp_path / "review")

    v1 = r.review(_shot(1), _video_path(tmp_path, 1), out_dir=out_dir)
    assert r._unattended is False
    assert r._consecutive_timeouts == 1
    assert "external review timed out" in " ".join(v1["reasons"])

    v2 = r.review(_shot(2), _video_path(tmp_path, 2), out_dir=out_dir)
    assert r._unattended is True                 # 2nd consecutive timeout engages it
    assert r._consecutive_timeouts == 2
    assert "external review timed out" in " ".join(v2["reasons"])


def test_below_threshold_does_not_engage(tmp_path):
    r = _reviewer(unattended_after=3)
    out_dir = str(tmp_path / "review")
    r.review(_shot(1), _video_path(tmp_path, 1), out_dir=out_dir)
    r.review(_shot(2), _video_path(tmp_path, 2), out_dir=out_dir)
    assert r._unattended is False
    assert r._consecutive_timeouts == 2


# --------------------------------------------------------- skip once engaged --
def test_unattended_mode_skips_the_wait(tmp_path):
    r = _reviewer(unattended_after=2)  # fast timeout so *engaging* stays quick
    out_dir = str(tmp_path / "review")
    r.review(_shot(1), _video_path(tmp_path, 1), out_dir=out_dir)
    r.review(_shot(2), _video_path(tmp_path, 2), out_dir=out_dir)
    assert r._unattended is True

    # Now prove the 3rd shot does NOT sit through a wait: bump the timeout to
    # something we'd clearly notice if it blocked, and time the call.
    r.timeout = 5.0
    start = time.perf_counter()
    v3 = r.review(_shot(3), _video_path(tmp_path, 3), out_dir=out_dir)
    elapsed = time.perf_counter() - start

    assert elapsed < 1.0, f"unattended review should not block, took {elapsed}s"
    assert v3["accept"] is True  # heuristic stage-1 verdict is untouched/still used
    assert "unattended mode" in " ".join(v3["reasons"])


def test_unattended_mode_still_writes_the_request(tmp_path):
    r = _reviewer(unattended_after=1)
    out_dir = str(tmp_path / "review")
    r.review(_shot(1), _video_path(tmp_path, 1), out_dir=out_dir)  # engages after 1 timeout
    assert r._unattended is True

    r.review(_shot(2), _video_path(tmp_path, 2), out_dir=out_dir)
    request_path = os.path.join(out_dir, "s02_a0.request.json")
    assert os.path.exists(request_path)  # a returning human can still find/answer it
    with open(request_path, encoding="utf-8") as f:
        req = json.load(f)
    assert req["shot_id"] == "s02"


def test_unattended_tracks_pending_response_paths(tmp_path):
    r = _reviewer(unattended_after=1)
    out_dir = str(tmp_path / "review")
    r.review(_shot(1), _video_path(tmp_path, 1), out_dir=out_dir)  # engages
    r.review(_shot(2), _video_path(tmp_path, 2), out_dir=out_dir)  # skipped, tracked
    expected = os.path.join(out_dir, "s02_a0.response.json")
    assert expected in r._pending_unattended_responses


# ------------------------------------------------------------------ reset --
def test_resets_when_a_backlogged_request_gets_answered(tmp_path):
    r = _reviewer(unattended_after=1)
    out_dir = str(tmp_path / "review")
    r.review(_shot(1), _video_path(tmp_path, 1), out_dir=out_dir)  # engages unattended
    assert r._unattended is True

    v2 = r.review(_shot(2), _video_path(tmp_path, 2), out_dir=out_dir)  # skipped, tracked
    assert "unattended mode" in " ".join(v2["reasons"])
    backlog_path = os.path.join(out_dir, "s02_a0.response.json")
    assert backlog_path in r._pending_unattended_responses

    # A human comes back and answers the backlogged shot 2 request while shot
    # 3 is (notionally) rendering on the GPU.
    with open(backlog_path, "w", encoding="utf-8") as f:
        json.dump({"accept": True, "score": 0.9, "reasons": []}, f)

    # Pre-answer shot 3's own request too, so the now-resumed wait for shot 3
    # returns immediately instead of sleeping out the (long) timeout.
    shot3_response = os.path.join(out_dir, "s03_a0.response.json")
    with open(shot3_response, "w", encoding="utf-8") as f:
        json.dump({"accept": False, "score": 0.1, "reasons": ["bad crop"]}, f)

    v3 = r.review(_shot(3), _video_path(tmp_path, 3), out_dir=out_dir)

    assert r._unattended is False              # resumed waiting
    assert r._consecutive_timeouts == 0
    assert r._pending_unattended_responses == []
    assert v3["accept"] is False                # shot 3's own (human) verdict was honored
    assert v3["reasons"] == ["bad crop"]


def test_reset_checks_full_backlog_not_only_most_recent(tmp_path):
    """A hit on an OLDER backlogged request (not just the immediately
    preceding one) must still trigger the reset."""
    r = _reviewer(unattended_after=1)
    out_dir = str(tmp_path / "review")
    r.review(_shot(1), _video_path(tmp_path, 1), out_dir=out_dir)  # engages
    r.review(_shot(2), _video_path(tmp_path, 2), out_dir=out_dir)  # skipped -> backlog[0]
    r.review(_shot(3), _video_path(tmp_path, 3), out_dir=out_dir)  # skipped -> backlog[1]
    assert len(r._pending_unattended_responses) == 2

    # Answer the OLDEST backlogged request (shot 2), not the newest.
    oldest = os.path.join(out_dir, "s02_a0.response.json")
    with open(oldest, "w", encoding="utf-8") as f:
        json.dump({"accept": True, "reasons": []}, f)

    shot4_response = os.path.join(out_dir, "s04_a0.response.json")
    with open(shot4_response, "w", encoding="utf-8") as f:
        json.dump({"accept": True, "reasons": []}, f)

    r.review(_shot(4), _video_path(tmp_path, 4), out_dir=out_dir)
    assert r._unattended is False
    assert r._consecutive_timeouts == 0


def test_successful_response_resets_counter_without_engaging(tmp_path):
    """A response arriving during ordinary (attended) waiting resets the
    streak, same as before this feature existed — pure regression check."""
    r = _reviewer(unattended_after=2)
    out_dir = str(tmp_path / "review")

    r.review(_shot(1), _video_path(tmp_path, 1), out_dir=out_dir)  # 1 timeout
    assert r._consecutive_timeouts == 1

    response_path = os.path.join(out_dir, "s02_a0.response.json")
    with open(response_path, "w", encoding="utf-8") as f:
        json.dump({"accept": True, "reasons": []}, f)
    v2 = r.review(_shot(2), _video_path(tmp_path, 2), out_dir=out_dir)

    assert v2["accept"] is True
    assert r._consecutive_timeouts == 0
    assert r._unattended is False

    # Streak must restart from zero — needs unattended_after more misses again.
    r.review(_shot(3), _video_path(tmp_path, 3), out_dir=out_dir)
    assert r._consecutive_timeouts == 1
    assert r._unattended is False


# ------------------------------------------------------------- config wiring --
def test_unattended_after_defaults_to_two():
    r = ExternalReviewer(cfg={})
    assert r.unattended_after == 2


def test_unattended_after_reads_config_key():
    r = ExternalReviewer(cfg={"kidsong": {"review": {"unattended_after_timeouts": 5}}})
    assert r.unattended_after == 5


def test_unattended_after_clamped_to_at_least_one():
    r = ExternalReviewer(cfg={"kidsong": {"review": {"unattended_after_timeouts": 0}}})
    assert r.unattended_after == 1
