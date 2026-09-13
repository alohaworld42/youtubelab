"""Run-scoped unattended detection across ALL FOUR external-review gates.

Before this, only `ExternalReviewer.review` (the per-SHOT gate) learned that
nobody was answering. The script, shotlist and cut gates called
`review.poll_response` directly, so each of them paid the full
`external_timeout` on every single episode — measured at 8 minutes of idle
wall clock per run (4 gate-waits x 120s), GPU free the whole time.

These tests pin the shared-session behavior:
  * unattended engages after N unanswered requests ANYWHERE in the run, and
    later gates then return fast;
  * a human answering anything resets it;
  * two runs in one process do not share state;
  * every gate still WRITES its request file when it skips the wait;
  * and the load-bearing one: the programmatic verdicts are unchanged by
    unattended mode, proven against the real rejected artifacts on disk.

No GPU, no network, no ComfyUI: the gates are driven with tiny cfg dicts
pointing at tmp_path, and `external_timeout` is set to a few hundredths of a
second so "waiting" is observable without being slow.
"""
import copy
import json
import os
import time

import pytest

from pipeline.kidsong import review as review_mod
from pipeline.kidsong.review import ExternalReviewer, ReviewSession, _EMPTY_HINTS
from pipeline.kidsong.script_qc import review_script, review_shotlist


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REAL_BASE = os.path.join(
    REPO_ROOT, "output", "20260720-110129-kidsong-zuris-watering-blooms-day"
)
REAL_SONG = _REAL_BASE + "-song.json"
REAL_SHOTS = _REAL_BASE + "-shots.json"


def _cfg(tmp_path, timeout=0.05, unattended_after=2, first_probe=0):
    """A minimal real-shaped cfg whose output_dir is this test's tmp_path."""
    return {
        "_root": str(tmp_path),
        "paths": {"output_dir": str(tmp_path / "output")},
        "kidsong": {
            "review": {
                "reviewer": "external",
                "external_timeout": timeout,
                "unattended_after_timeouts": unattended_after,
                "external_first_probe": first_probe,
                # This suite tests the shared unattended/wait state machine
                # across gates; keep the shot-level prefilter off so a clean
                # heuristic still queues a shot request as these tests expect.
                "shot_auto_accept": {"enabled": False},
            }
        },
    }


def _session(cfg):
    return review_mod._SESSIONS[review_mod._session_key(cfg)]


def _review_dir(cfg):
    from pipeline.kidsong.script_qc import _review_dir as rd

    return rd(cfg)


def _song():
    """A song that FAILS the programmatic script gate (so we can prove the
    reject survives unattended mode). Deliberately awful: two verses, no
    repetition, no rhyme, no action verbs, no cast sentence."""
    return {
        "title": "",
        "characters": "",
        "verses": [
            {"lines": ["Purple elephants juggle enormous mathematics textbooks"]},
            {"lines": ["Silver dolphins compose intricate crystalline symphonies"]},
        ],
    }


def _good_shot(i):
    return {"id": f"s{i:02d}", "characters": ["all"]}


def _external_reviewer(cfg):
    r = ExternalReviewer(cfg=cfg)
    r.poll_seconds = 0.01
    # Stage-1 heuristics are untouched by this feature and have their own
    # tests; force accept so every call reaches the external stage.
    r._heuristic.review = lambda shot, video_path: {
        "accept": True, "score": 1.0, "reasons": [], "retry_hints": dict(_EMPTY_HINTS),
    }
    return r


# ------------------------------------------------------- cross-gate engagement --
def test_unattended_engages_across_different_gates(tmp_path):
    """Two unanswered requests at DIFFERENT gates (script, then shotlist) must
    engage unattended mode — the whole point of sharing the counter."""
    cfg = _cfg(tmp_path, unattended_after=2)
    song = _song()

    review_script(song, cfg)
    s = _session(cfg)
    assert s.unattended is False
    assert s.consecutive_timeouts == 1

    review_shotlist({"shots": []}, song, cfg)
    assert s.unattended is True, "a second unanswered gate must engage unattended"
    assert s.consecutive_timeouts == 2


def test_shot_and_cut_gates_inherit_the_script_gates_discovery(tmp_path):
    """script + shotlist go unanswered; the shot gate must ALREADY be
    unattended on its very first call instead of paying two more timeouts."""
    cfg = _cfg(tmp_path, unattended_after=2)
    song = _song()
    review_script(song, cfg)
    review_shotlist({"shots": []}, song, cfg)
    assert _session(cfg).unattended is True

    r = _external_reviewer(cfg)
    assert r.session is _session(cfg), "the shot gate must join the run's session"
    assert r._unattended is True

    r.timeout = 5.0  # would be glaringly obvious if it blocked
    start = time.perf_counter()
    v = r.review(_good_shot(1), str(tmp_path / "s01_a0.mp4"), out_dir=str(tmp_path / "rev"))
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0, f"shot gate blocked for {elapsed}s despite unattended mode"
    assert "unattended mode" in " ".join(v["reasons"])


def test_later_gates_return_fast_once_unattended(tmp_path):
    """The measured win: with a long timeout configured, the LATER gates of an
    unattended run (shot, cut) must not sit through it.

    Gates are exercised in their real run order — script, shotlist, shot,
    cut — because arriving at an earlier-ranked gate is precisely how a NEW
    run is detected (see test_two_runs_in_one_process_do_not_share_state).
    """
    cfg = _cfg(tmp_path, timeout=0.05, unattended_after=2)
    song = _song()
    review_script(song, cfg)                    # unanswered 1
    review_shotlist({"shots": []}, song, cfg)   # unanswered 2 -> engaged
    assert _session(cfg).unattended is True

    # Now make the timeout something we would definitely notice if paid.
    rev = str(tmp_path / "rev")
    os.makedirs(rev, exist_ok=True)
    start = time.perf_counter()

    r = _external_reviewer(cfg)
    r.timeout = 5.0
    r.review(_good_shot(1), str(tmp_path / "s01_a0.mp4"), out_dir=rev)

    response, skipped = review_mod.external_gate_poll(
        "cut",
        os.path.join(rev, "cut.request.json"),
        os.path.join(rev, "cut.response.json"),
        {"programmatic_verdict": {"accept": False}},
        5.0,
        cfg,
    )
    elapsed = time.perf_counter() - start

    assert skipped is True and response is None
    assert elapsed < 1.0, f"unattended shot+cut gates still waited {elapsed}s"
    # ...and the cut gate's request is on disk for a returning human.
    assert os.path.exists(os.path.join(rev, "cut.request.json"))


# ----------------------------------------------------------- requests written --
@pytest.mark.parametrize("stage", ["script", "shotlist"])
def test_skipped_gates_still_write_their_request(tmp_path, stage):
    """A human coming back later must still find something to answer."""
    cfg = _cfg(tmp_path, unattended_after=1)
    song = _song()
    review_script(song, cfg)  # engages immediately (unattended_after=1)
    assert _session(cfg).unattended is True

    if stage == "script":
        review_script(song, cfg)
    else:
        review_shotlist({"shots": []}, song, cfg)

    request_path = os.path.join(_review_dir(cfg), f"{stage}.request.json")
    assert os.path.exists(request_path), f"{stage} gate skipped WITHOUT writing a request"
    with open(request_path, encoding="utf-8") as f:
        payload = json.load(f)
    # The programmatic verdict must be in the request so a returning human can
    # see what the gate decided on its own.
    assert payload["programmatic_verdict"]["accept"] is False


def test_cut_gate_skips_the_wait_but_still_writes_its_sheet_and_holds_the_reject(tmp_path):
    """End-to-end through cut_qc's own gate: the last gate of a run inherits
    the run's unattended state, still writes its request AND its cut sheet,
    and — critically — a failing cut is still held."""
    from pipeline.kidsong import cut_qc

    cfg = _cfg(tmp_path, unattended_after=1)
    review_script(_song(), cfg)  # engages
    s = _session(cfg)
    assert s.unattended is True

    review_dir = str(tmp_path / "cutrev")
    programmatic = {
        "accept": False,
        "score": 0.2,
        "reasons": ["audio silent: rms 0.0001 <= 0.005"],
        "retry_hints": {"recut": True, "reshoot": []},
    }

    start = time.perf_counter()
    out = cut_qc._external_gate(
        str(tmp_path / "no-such-video.mp4"),
        [{"shot_id": "c0", "start": 0.0, "end": 2.0}],
        programmatic,
        cfg,
        review_dir,
    )
    elapsed = time.perf_counter() - start

    assert elapsed < 1.0, f"cut gate waited {elapsed}s despite unattended mode"
    # The verdict is untouched: still a reject, same score, same reasons, same
    # retry hint that drives the recut loop.
    assert out["accept"] is False
    assert out["score"] == 0.2
    assert out["reasons"][0] == "audio silent: rms 0.0001 <= 0.005"
    assert out["retry_hints"] == {"recut": True, "reshoot": []}
    # Evidence for a returning human is still on disk.
    assert os.path.exists(os.path.join(review_dir, "cut.request.json"))
    assert os.path.exists(os.path.join(review_dir, "cut_sheet.png"))
    assert os.path.join(review_dir, "cut.response.json") in s.pending_responses


def test_skipped_gates_track_their_response_path_for_a_late_answer(tmp_path):
    cfg = _cfg(tmp_path, unattended_after=1)
    song = _song()
    review_script(song, cfg)                    # engages
    review_shotlist({"shots": []}, song, cfg)   # skipped, tracked

    expected = os.path.join(_review_dir(cfg), "shotlist.response.json")
    assert expected in _session(cfg).pending_responses


# ------------------------------------------------------------------- resets --
def test_a_human_answering_a_backlogged_request_resumes_waiting(tmp_path):
    cfg = _cfg(tmp_path, unattended_after=1)
    song = _song()
    review_script(song, cfg)                    # engages
    review_shotlist({"shots": []}, song, cfg)   # skipped -> backlog
    s = _session(cfg)
    assert s.unattended is True

    # A human comes back and answers the shotlist request that got skipped.
    backlog = os.path.join(_review_dir(cfg), "shotlist.response.json")
    with open(backlog, "w", encoding="utf-8") as f:
        json.dump({"accept": True, "score": 0.9, "reasons": ["looks fine"]}, f)

    # Pre-answer the cut gate's own request so the resumed wait returns at once.
    review_dir = str(tmp_path / "cutrev")
    os.makedirs(review_dir, exist_ok=True)

    r = _external_reviewer(cfg)
    shot_response = os.path.join(review_dir, "s01_a0.response.json")
    with open(shot_response, "w", encoding="utf-8") as f:
        json.dump({"accept": False, "score": 0.1, "reasons": ["bad crop"]}, f)

    v = r.review(_good_shot(1), str(tmp_path / "s01_a0.mp4"), out_dir=review_dir)

    assert s.unattended is False, "answering a backlogged request must resume waiting"
    assert s.consecutive_timeouts == 0
    assert s.pending_responses == []
    assert v["accept"] is False and v["reasons"] == ["bad crop"]


def test_answering_a_gate_directly_resets_the_streak(tmp_path):
    """A human answering WHILE a gate is waiting resets the streak.

    The response has to be written during the wait, not before it: a gate
    always deletes any stale response for its fixed path before writing a new
    request, so last run's verdict can never decide this one.
    """
    import threading

    cfg = _cfg(tmp_path, timeout=3.0, unattended_after=3)
    song = _song()
    cfg["kidsong"]["review"]["external_timeout"] = 0.05
    review_script(song, cfg)
    s = _session(cfg)
    assert s.consecutive_timeouts == 1

    cfg["kidsong"]["review"]["external_timeout"] = 3.0
    response_path = os.path.join(_review_dir(cfg), "shotlist.response.json")

    def answer():
        with open(response_path, "w", encoding="utf-8") as f:
            json.dump({"accept": True, "score": 0.8, "reasons": []}, f)

    timer = threading.Timer(0.2, answer)
    timer.start()
    try:
        v = review_shotlist({"shots": []}, song, cfg)
    finally:
        timer.cancel()

    assert v["accept"] is True
    assert s.consecutive_timeouts == 0
    assert s.unattended is False
    assert s.human_seen is True


# -------------------------------------------------------- run isolation --
def test_two_runs_in_one_process_do_not_share_state(tmp_path):
    """A second run must start attended even though the first ended unattended
    — and even though the scheduler hands both runs the SAME cfg object."""
    cfg = _cfg(tmp_path, unattended_after=2)
    song = _song()

    # --- run 1: nobody answers anything, all four gates seen ---
    review_script(song, cfg)
    review_shotlist({"shots": []}, song, cfg)
    run1 = _session(cfg)
    assert run1.unattended is True
    r1 = _external_reviewer(cfg)
    r1.review(_good_shot(1), str(tmp_path / "a.mp4"), out_dir=str(tmp_path / "r1"))
    review_mod.session_for(cfg, "cut")  # the run's last gate
    assert run1.max_stage_rank == 3

    # --- run 2 starts at its script gate, same cfg object ---
    review_script(song, cfg)
    run2 = _session(cfg)
    assert run2 is not run1, "run 2 must get a fresh session"
    assert run2.unattended is False, "run 2 must start out attended"
    assert run2.consecutive_timeouts == 1  # its own first timeout, not run 1's


def test_a_resumed_run_also_gets_a_fresh_session(tmp_path):
    """A resumed run skips the script/shotlist gates and starts at the shot
    gate — that too must rotate in a fresh session."""
    cfg = _cfg(tmp_path, unattended_after=2)
    song = _song()
    review_script(song, cfg)
    review_shotlist({"shots": []}, song, cfg)
    review_mod.session_for(cfg, "cut")
    run1 = _session(cfg)
    assert run1.unattended is True

    r2 = _external_reviewer(cfg)  # resumed run: first gate is "shot"
    assert r2.session is not run1
    assert r2.session.unattended is False


def test_retries_within_a_stage_stay_in_the_same_run(tmp_path):
    """review_script is called twice on a regenerate, review_cut twice on a
    recut. Repeating a stage must NOT be mistaken for a new run, or the
    counter would reset and the run would never go unattended."""
    cfg = _cfg(tmp_path, unattended_after=2)
    song = _song()
    review_script(song, cfg)   # attempt 0
    first = _session(cfg)
    review_script(song, cfg)   # attempt 1 (the regenerate retry)
    assert _session(cfg) is first
    assert first.consecutive_timeouts == 2
    assert first.unattended is True


def test_separate_output_dirs_never_share_a_session(tmp_path):
    a = _cfg(tmp_path / "a", unattended_after=1)
    b = _cfg(tmp_path / "b", unattended_after=1)
    review_script(_song(), a)
    assert _session(a).unattended is True
    assert review_mod._session_key(b) not in review_mod._SESSIONS


# -------------------------------------------- VERDICTS ARE UNCHANGED (real data) --
def _programmatic_only(verdict):
    """Strip the trailing "we didn't get an answer" marker the external gate
    appends, leaving exactly the programmatic verdict."""
    out = copy.deepcopy(verdict)
    out["reasons"] = [
        r for r in out["reasons"]
        if not r.startswith("external review timed out")
        and not r.startswith("unattended mode:")
    ]
    return out


@pytest.mark.skipif(
    not (os.path.exists(REAL_SONG) and os.path.exists(REAL_SHOTS)),
    reason="real 20260720-110129 artifacts not on disk",
)
def test_real_bad_artifacts_get_identical_verdicts_with_and_without_unattended(tmp_path):
    """The load-bearing test. The 20260720-110129 episode's song and shot list
    are real, genuinely-bad artifacts that the programmatic gates reject
    (score 0.0 / 0.0). Unattended mode must change how long we wait and
    NOTHING else: same accept, same score, same reasons, same retry hints.
    """
    with open(REAL_SONG, encoding="utf-8") as f:
        song = json.load(f)
    with open(REAL_SHOTS, encoding="utf-8") as f:
        shotlist = json.load(f)

    # --- attended run: unattended can never engage (threshold far above the
    # --- number of gates we call), so both gates wait and time out.
    attended_cfg = _cfg(tmp_path / "attended", unattended_after=99)
    v_script_attended = review_script(song, attended_cfg)
    v_shots_attended = review_shotlist(shotlist, song, attended_cfg)
    assert _session(attended_cfg).unattended is False

    # --- unattended run: engages on the very first gate.
    un_cfg = _cfg(tmp_path / "unattended", unattended_after=1)
    v_script_un = review_script(song, un_cfg)
    v_shots_un = review_shotlist(shotlist, song, un_cfg)
    assert _session(un_cfg).unattended is True, "unattended must have engaged"

    # Both must still REJECT — a fall-through must never turn a reject into
    # an accept.
    for v in (v_script_attended, v_script_un, v_shots_attended, v_shots_un):
        assert v["accept"] is False, v

    assert v_script_attended["score"] == 0.0
    assert v_shots_attended["score"] == 0.0

    # Byte-identical programmatic verdicts.
    assert json.dumps(_programmatic_only(v_script_un), sort_keys=True) == json.dumps(
        _programmatic_only(v_script_attended), sort_keys=True
    )
    assert json.dumps(_programmatic_only(v_shots_un), sort_keys=True) == json.dumps(
        _programmatic_only(v_shots_attended), sort_keys=True
    )

    # And the retry hints that drive the regenerate/replan loops survive.
    assert v_script_un["retry_hints"]["regenerate"] is True
    assert v_shots_un["retry_hints"]["replan"] is True


@pytest.mark.skipif(
    not os.path.exists(REAL_SONG), reason="real song artifact not on disk"
)
def test_unattended_does_not_turn_a_reject_into_an_accept_at_any_gate(tmp_path):
    """Same guarantee stated as an invariant over a sweep of thresholds."""
    with open(REAL_SONG, encoding="utf-8") as f:
        song = json.load(f)
    for n in (1, 2, 3, 99):
        cfg = _cfg(tmp_path / f"n{n}", unattended_after=n)
        for _ in range(4):
            v = review_script(song, cfg)
            assert v["accept"] is False, (n, v)
            assert v["score"] == 0.0, (n, v)


# ------------------------------------------------------- short first probe --
def test_first_probe_shortens_only_the_pre_human_wait():
    s = ReviewSession(unattended_after=2, first_probe=30.0)
    assert s.wait_seconds(120.0) == 30.0     # nobody has answered yet
    s.note_response("script")                # a human answers something
    assert s.wait_seconds(120.0) == 120.0    # full timeout from here on


def test_first_probe_defaults_off():
    s = ReviewSession(unattended_after=2)
    assert s.wait_seconds(120.0) == 120.0


def test_first_probe_never_exceeds_the_configured_timeout():
    s = ReviewSession(unattended_after=2, first_probe=300.0)
    assert s.wait_seconds(120.0) == 120.0


def test_first_probe_is_off_in_the_shipped_config():
    """The knob exists but must not silently change anyone's behavior."""
    from pipeline.config import load_config

    review_cfg = load_config()["kidsong"]["review"]
    assert review_cfg.get("external_first_probe", 0) == 0


def test_unattended_wait_is_zero():
    s = ReviewSession(unattended_after=1)
    s.note_timeout("script")
    assert s.unattended is True
    assert s.wait_seconds(120.0) == 0.0
