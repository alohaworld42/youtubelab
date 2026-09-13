"""Tests for the per-shot review PREFILTER and the backlog drain.

The external review queue was flooding — every shot that cleared the hard
heuristic wrote a request and waited for a person, so batches of episodes piled
up hundreds of per-shot requests no one could review. The prefilter auto-accepts
a shot the heuristic clears (unless one of its soft reasons is in
`surface_reasons`) WITHOUT writing a request or queueing it; the drain applies
the same decision retroactively to the backlog on disk.

No GPU/video needed: `_heuristic.review` is monkeypatched to a controlled
verdict, and request/response files are plain JSON under tmp_path.
"""
import json
import os

from pipeline.kidsong.review import (
    ExternalReviewer,
    _EMPTY_HINTS,
    _shot_auto_accept_policy,
)
from pipeline.kidsong import review_queue


def _reviewer(cfg_review):
    r = ExternalReviewer(cfg={"kidsong": {"review": cfg_review}})
    r.timeout = 0.03
    r.poll_seconds = 0.005
    return r


def _with_heuristic(r, reasons, accept=True, score=1.0):
    r._heuristic.review = lambda shot, video_path: {
        "accept": accept, "score": score, "reasons": list(reasons),
        "retry_hints": dict(_EMPTY_HINTS),
    }
    return r


def _shot():
    # A SOLO shot: since the max_children cap, only shots staging at most one
    # child are eligible for blind auto-accept (group shots are where
    # clone-collapse and invented children live, and the heuristic can't see
    # either) — the group-shot behavior has its own tests below.
    return {"id": "s01", "characters": ["Kofi"], "shot_type": "closeup", "action": "clapping"}


def _request_exists(out_dir, take="s01_a0"):
    return os.path.exists(os.path.join(out_dir, f"{take}.request.json"))


# ------------------------------------------------------------- policy parse ---
def test_policy_defaults_to_enabled_surface_nothing():
    enabled, surface, max_children = _shot_auto_accept_policy({})
    assert enabled is True and surface == set()
    assert max_children == 1  # solo shots only, by default


def test_policy_reads_config():
    enabled, surface, max_children = _shot_auto_accept_policy(
        {"kidsong": {"review": {"shot_auto_accept": {
            "enabled": False, "surface_reasons": ["blurry"], "max_children": 3}}}}
    )
    assert enabled is False and surface == {"blurry"} and max_children == 3


def test_policy_accepts_a_single_string_surface_reason():
    _enabled, surface, _max = _shot_auto_accept_policy(
        {"kidsong": {"review": {"shot_auto_accept": {"surface_reasons": "blurry"}}}}
    )
    assert surface == {"blurry"}


def test_policy_max_children_zero_or_none_means_no_cap():
    for raw in (0, None):
        _e, _s, max_children = _shot_auto_accept_policy(
            {"kidsong": {"review": {"shot_auto_accept": {"max_children": raw}}}}
        )
        assert max_children == 0


# --------------------------------------------------------------- prefilter ---
def test_clean_shot_is_auto_accepted_without_a_request(tmp_path):
    """The aggressive default: a heuristic-cleared shot is accepted at the shot
    level and no request file is written (so nothing hits the human queue)."""
    out_dir = str(tmp_path / "review")
    r = _with_heuristic(_reviewer({}), reasons=[])
    verdict = r.review(_shot(), str(tmp_path / "s01_a0.mp4"), out_dir=out_dir)
    assert verdict["accept"] is True
    assert verdict.get("auto_accepted") is True
    assert not _request_exists(out_dir), "an auto-accepted shot must not queue a request"


def test_blurry_only_shot_is_auto_accepted_by_default(tmp_path):
    """`blurry` is a known-unreliable soft flag; with the default (empty
    surface_reasons) a blurry-only shot is still auto-accepted."""
    out_dir = str(tmp_path / "review")
    r = _with_heuristic(_reviewer({}), reasons=["blurry"], score=0.7)
    verdict = r.review(_shot(), str(tmp_path / "s01_a0.mp4"), out_dir=out_dir)
    assert verdict.get("auto_accepted") is True
    assert not _request_exists(out_dir)


def test_surfaced_reason_still_writes_a_request(tmp_path):
    """Listing a soft reason in surface_reasons routes those shots back to the
    human: a request is written and (with no responder) it times out."""
    out_dir = str(tmp_path / "review")
    r = _with_heuristic(
        _reviewer({"shot_auto_accept": {"surface_reasons": ["blurry"]}, "external_timeout": 0.03}),
        reasons=["blurry"], score=0.7,
    )
    verdict = r.review(_shot(), str(tmp_path / "s01_a0.mp4"), out_dir=out_dir)
    assert verdict.get("auto_accepted") is not True
    assert _request_exists(out_dir), "a surfaced shot must write a request for the human"


def test_hard_rejected_shot_is_never_auto_accepted(tmp_path):
    """A hard reject (heuristic accept=False) returns the reject verdict and is
    never dressed up as an auto-accept."""
    out_dir = str(tmp_path / "review")
    r = _with_heuristic(_reviewer({}), reasons=["static"], accept=False, score=0.0)
    verdict = r.review(_shot(), str(tmp_path / "s01_a0.mp4"), out_dir=out_dir)
    assert verdict["accept"] is False
    assert verdict.get("auto_accepted") is not True
    assert not _request_exists(out_dir)


def test_disabled_prefilter_writes_a_request(tmp_path):
    """enabled=false restores the old one-request-per-shot behaviour."""
    out_dir = str(tmp_path / "review")
    r = _with_heuristic(
        _reviewer({"shot_auto_accept": {"enabled": False}, "external_timeout": 0.03}),
        reasons=[],
    )
    r.review(_shot(), str(tmp_path / "s01_a0.mp4"), out_dir=out_dir)
    assert _request_exists(out_dir)


# ------------------------------------------------------------------- drain ---
def _write_request(out_dir, take, heuristic, kind="shot", shot=None):
    os.makedirs(out_dir, exist_ok=True)
    name = "cut" if kind == "cut" else take
    # Default stored shot is SOLO: since the max_children cap, a stored
    # multi-child (or characters-less = ensemble-by-convention) request is
    # surfaced rather than drained — that path has its own test below.
    if shot is None:
        shot = {"id": take.split("_")[0], "characters": ["Kofi"], "shot_type": "closeup"}
    body = {"take": take, "shot_id": take, "shot": shot, "heuristic": heuristic}
    with open(os.path.join(out_dir, f"{name}.request.json"), "w", encoding="utf-8") as f:
        json.dump(body, f)
    return os.path.join(out_dir, f"{name}.response.json")


def _drain_cfg(tmp_path, surface_reasons=None):
    review = {"reviewer": "external"}
    if surface_reasons is not None:
        review["shot_auto_accept"] = {"surface_reasons": surface_reasons}
    return {
        "_root": str(tmp_path),
        "paths": {"output_dir": str(tmp_path / "output")},
        "kidsong": {"review": review},
    }


def test_drain_writes_responses_only_for_clearable_shots(tmp_path):
    cfg = _drain_cfg(tmp_path)
    out = str(tmp_path / "output")
    rd = os.path.join(out, "ep1-shots", "review")
    clean = _write_request(rd, "s00_a0", {"accept": True, "score": 1.0, "reasons": []})
    blurry = _write_request(rd, "s01_a0", {"accept": True, "score": 0.7, "reasons": ["blurry"]})
    # A stored hard-reject (shouldn't normally sit here, but be safe): never drained.
    rejected = _write_request(rd, "s02_a0", {"accept": False, "score": 0.0, "reasons": ["static"]})
    cut = _write_request(rd, "cut", {}, kind="cut")

    summary = review_queue.drain_auto_acceptable(cfg, out_dir=out)
    assert summary["drained"] == 2
    assert summary["surfaced"] == 1        # the hard-reject
    assert summary["skipped_non_shot"] == 1  # the cut request
    assert os.path.exists(clean) and os.path.exists(blurry)
    assert not os.path.exists(rejected)
    # the cut request keeps no response — still surfaced to the human
    assert not os.path.exists(cut)
    assert not os.path.exists(os.path.join(rd, "cut.response.json"))
    # a drained response is a real accept verdict, tagged
    body = json.load(open(clean, encoding="utf-8"))
    assert body["accept"] is True and body["auto_accepted"] is True


def test_drain_respects_surface_reasons(tmp_path):
    cfg = _drain_cfg(tmp_path, surface_reasons=["blurry"])
    out = str(tmp_path / "output")
    rd = os.path.join(out, "ep1-shots", "review")
    clean = _write_request(rd, "s00_a0", {"accept": True, "score": 1.0, "reasons": []})
    blurry = _write_request(rd, "s01_a0", {"accept": True, "score": 0.7, "reasons": ["blurry"]})

    summary = review_queue.drain_auto_acceptable(cfg, out_dir=out)
    assert summary["drained"] == 1 and summary["surfaced"] == 1
    assert os.path.exists(clean)
    assert not os.path.exists(blurry), "a surfaced-reason shot stays in the queue"


def test_drain_dry_run_writes_nothing(tmp_path):
    cfg = _drain_cfg(tmp_path)
    out = str(tmp_path / "output")
    rd = os.path.join(out, "ep1-shots", "review")
    clean = _write_request(rd, "s00_a0", {"accept": True, "score": 1.0, "reasons": []})

    summary = review_queue.drain_auto_acceptable(cfg, out_dir=out, dry_run=True)
    assert summary["drained"] == 1
    assert not os.path.exists(clean), "dry-run must not write any response file"


def test_drain_is_idempotent(tmp_path):
    cfg = _drain_cfg(tmp_path)
    out = str(tmp_path / "output")
    rd = os.path.join(out, "ep1-shots", "review")
    _write_request(rd, "s00_a0", {"accept": True, "score": 1.0, "reasons": []})

    first = review_queue.drain_auto_acceptable(cfg, out_dir=out)
    second = review_queue.drain_auto_acceptable(cfg, out_dir=out)
    assert first["drained"] == 1
    assert second["drained"] == 0, "already-answered requests are not re-drained"


# --------------------------------------------------- max_children group cap ---
def _group_shot():
    return {"id": "s02", "characters": ["all"], "shot_type": "wide", "action": "the kids clap"}


def _pair_shot():
    return {"id": "s03", "characters": ["Zuri", "Kofi"], "shot_type": "medium",
            "action": "Zuri passes the drum to Kofi"}


def test_group_shot_is_never_blind_auto_accepted(tmp_path):
    """The measured failure this cap kills: every multi-child shot of the
    pixar-full episode shipped score-1.0 with zero inspection, while the
    heuristic cannot see clone-collapse or invented children at all."""
    out_dir = str(tmp_path / "review")
    r = _with_heuristic(_reviewer({}), reasons=[])
    verdict = r.review(_group_shot(), str(tmp_path / "s02_a0.mp4"), out_dir=out_dir)
    assert verdict.get("auto_accepted") is not True
    assert _request_exists(out_dir, "s02_a0"), "a group shot must reach the vision queue"


def test_pair_shot_is_never_blind_auto_accepted(tmp_path):
    out_dir = str(tmp_path / "review")
    r = _with_heuristic(_reviewer({}), reasons=[])
    verdict = r.review(_pair_shot(), str(tmp_path / "s03_a0.mp4"), out_dir=out_dir)
    assert verdict.get("auto_accepted") is not True
    assert _request_exists(out_dir, "s03_a0")


def test_max_children_zero_restores_group_auto_accept(tmp_path):
    out_dir = str(tmp_path / "review")
    r = _with_heuristic(
        _reviewer({"shot_auto_accept": {"max_children": 0}}), reasons=[]
    )
    verdict = r.review(_group_shot(), str(tmp_path / "s02_a0.mp4"), out_dir=out_dir)
    assert verdict.get("auto_accepted") is True
    assert not _request_exists(out_dir, "s02_a0")


def test_count_resolution_failure_fails_open_to_auto_accept(tmp_path, monkeypatch):
    """expected_child_count blowing up must keep today's behavior (auto-accept),
    never crash or silently force the queue path."""
    from pipeline.kidsong import cast

    monkeypatch.setattr(cast, "expected_child_count",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    out_dir = str(tmp_path / "review")
    r = _with_heuristic(_reviewer({}), reasons=[])
    verdict = r.review(_group_shot(), str(tmp_path / "s02_a0.mp4"), out_dir=out_dir)
    assert verdict.get("auto_accepted") is True


def test_drain_surfaces_stored_group_requests(tmp_path):
    """A stored request whose shot stages the whole group (or carries no
    characters at all — ensemble by convention) is left for real eyes, exactly
    like the live prefilter since the max_children cap."""
    cfg = _drain_cfg(tmp_path)
    out = str(tmp_path / "output")
    rd = os.path.join(out, "ep1-shots", "review")
    group = _write_request(rd, "s04_a0", {"accept": True, "score": 1.0, "reasons": []},
                           shot={"id": "s04", "characters": ["all"], "shot_type": "wide"})
    legacy_empty = _write_request(rd, "s05_a0", {"accept": True, "score": 1.0, "reasons": []},
                                  shot={})
    summary = review_queue.drain_auto_acceptable(cfg, out_dir=out)
    assert summary["drained"] == 0
    assert summary["surfaced"] == 2
    assert not os.path.exists(group) and not os.path.exists(legacy_empty)
