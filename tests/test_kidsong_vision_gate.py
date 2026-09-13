"""Tests for the QM-010 vision-coverage promote gate
(pipeline.kidsong.generate._generate_director's `_vision_gate_check`, called
right before both promote-to-Final sites).

Contract under test (see docs/quality/DEFECT_BACKLOG.md QM-010 and
docs/quality/IMPROVEMENT_LOG.md IMP-008):
  * Before a cut is promoted, coverage is computed (via
    `review.vision_coverage`) over the takes ACTUALLY USED in the cut
    (`renders`, one entry per shot, including reuse_of/covered fallbacks).
  * coverage < kidsong.review.min_vision_coverage (default 0.8) => promotion
    is refused: staging .mp4 stays put, status="needs_review", the same
    clean-ending shape as a cut-QC reject (no crash), and a
    `{"gate": "vision_coverage", ...}` entry lands in review_log.json.
  * coverage >= threshold => promotion proceeds exactly as before.
  * The gate is skipped entirely (always promotes) when
    kidsong.review.min_vision_coverage is 0, OR when
    kidsong.review.reviewer == "heuristic" (no vision path exists by
    configuration) -- byte-identical to pre-gate behavior in both cases.

CPU-only, no GPU, no network, no ffmpeg, no ComfyUI: this drives
generate_kidsong via resume_base with every shot already "rendered" on disk
(so the render loop has zero pending work and reviewer.review is never
called), and every other GPU/media stage (Whisper, beat grid, cut assembly,
cut QC, intro) monkeypatched -- the same shape
tests/test_kidsong_intro_promotion.py uses. review_shots_log is left REAL
(not mocked) so review_log.json is actually written and can be inspected for
the gate entry.
"""
import json
import os
import struct
import wave

import pytest

from pipeline.kidsong import runstate
from pipeline.kidsong.comfy import ComfyClient as _RealComfyClient

_REAL_LOAD_WORKFLOW = _RealComfyClient.load_workflow


def _write_wav(path, seconds=4.0, rate=8000):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(struct.pack("<h", 0) * int(rate * seconds))


def _take(shots_dir, shot_id, attempt, size=1024):
    os.makedirs(shots_dir, exist_ok=True)
    p = os.path.join(shots_dir, f"{shot_id}_a{attempt}.mp4")
    with open(p, "wb") as f:
        f.write(b"\0" * size)
    return p


def _write_response(review_dir, take, accept):
    os.makedirs(review_dir, exist_ok=True)
    payload = {"accept": accept, "score": 0.9 if accept else 0.1, "reasons": []}
    with open(os.path.join(review_dir, f"{take}.response.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f)


@pytest.fixture
def gated_run(tmp_path, monkeypatch):
    """A director run with 2 shots (s00_a0, s01_a0) already fully rendered --
    the render loop never touches ComfyUI or the reviewer. `review_dir` is
    exposed so tests can plant .response.json files before calling
    generate_kidsong; `intro_invocations`/`calls` track whether promotion
    actually happened.
    """
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    base = "20260721-091500-kidsong-vision-gate-test"
    shots_dir = runstate.shots_dir_path(str(out_dir), base)
    review_dir = os.path.join(shots_dir, "review")

    _write_wav(out_dir / f"{base}.wav")
    song = {
        "title": "Vision Gate Test Song",
        "description": "d",
        "tags": ["t"],
        "characters": "Three toddlers: Zuri; Kofi; Nala",
        "verses": [{"scene": "backyard", "lines": ["a", "b"]}],
    }
    (out_dir / f"{base}-song.json").write_text(json.dumps(song), encoding="utf-8")

    shots = []
    for i in range(2):
        shot = {
            "id": f"s{i:02d}", "verse": 0, "start": i * 1.0, "end": (i + 1) * 1.0,
            "shot_type": "medium", "characters": ["all"], "action": "clapping",
            "setting": "a backyard", "camera": "static", "reuse_of": None,
            "seed": 1000 + i, "status": "planned",
        }
        runstate.mark_rendered(shot, _take(shots_dir, shot["id"], 0), score=0.9, attempts=1)
        shots.append(shot)
    runstate.save_shotlist(str(out_dir), base, {"shots": shots})

    calls = []
    intro_invocations = []

    import pipeline.captions as captions_mod
    import pipeline.kidsong.comfy as comfy_mod
    import pipeline.kidsong.cut_qc as cut_qc_mod
    import pipeline.kidsong.edit as edit_mod
    import pipeline.kidsong.review as review_mod
    import pipeline.kidsong.sing as sing_mod

    class FakeComfyClient:
        def __init__(self, cfg):
            self.url = "http://fake:8188"
            self._proc = None

        def load_workflow(self, workflow_name):
            return _REAL_LOAD_WORKFLOW(self, workflow_name)

        def ensure_up(self):
            raise AssertionError("no shot is pending — ensure_up should never be called")

        def render(self, workflow, patches, out_path):
            raise AssertionError("no shot is pending — render should never be called")

        def free(self):
            pass

    class FakeReviewer:
        def review(self, shot, video_path, out_dir=None):
            raise AssertionError("no shot is pending — reviewer.review should never be called")

    def fake_assemble(cfg, cut_list, renders, voice, words, duration, out_path, **kw):
        calls.append("assemble")
        with open(out_path, "wb") as f:
            f.write(b"\0" * 4096)

    def fake_review_cut(staging_path, cut_list, beats, words, duration, cfg, review_dir):
        calls.append("review_cut")
        return {"accept": True, "score": 1.0, "reasons": [], "retry_hints": {}}

    def fake_prepend_intro(video_path, cfg, on_progress=None):
        calls.append("intro")
        intro_invocations.append(video_path)
        return {"applied": True, "intro_seconds": 4.0, "reason": None}

    monkeypatch.setattr(comfy_mod, "ComfyClient", FakeComfyClient)
    monkeypatch.setattr(review_mod, "get_reviewer", lambda cfg: FakeReviewer())
    monkeypatch.setattr(review_mod, "contact_sheet", lambda *a, **k: None)
    # review_shots_log is intentionally left REAL so review_log.json actually
    # lands on disk and the gate entry can be inspected.
    monkeypatch.setattr(captions_mod, "get_word_timestamps", lambda *a, **k: [])
    monkeypatch.setattr(sing_mod, "verse_times_from_words", lambda *a, **k: [(0.0, 4.0)])
    monkeypatch.setattr(edit_mod, "beat_grid", lambda path: [0.0, 1.0, 2.0, 3.0])
    monkeypatch.setattr(edit_mod, "build_cut_list", lambda *a, **k: [])
    monkeypatch.setattr(edit_mod, "assemble", fake_assemble)
    monkeypatch.setattr(cut_qc_mod, "review_cut", fake_review_cut)
    monkeypatch.setattr(edit_mod, "prepend_intro", fake_prepend_intro)

    def _sing_must_not_run(*a, **k):
        raise AssertionError("resume must not re-sing")

    monkeypatch.setattr(sing_mod, "sing_song", _sing_must_not_run)

    cfg = {
        "_root": str(tmp_path),
        "paths": {"output_dir": str(out_dir)},
        "video": {"max_seconds": 60},
        "captions": {"uppercase": True},
        "kidsong": {
            "mode": "director",
            "singer": "ace",
            "seed": 20260717,
            "shot": {"fps": 24, "max_frames": 241, "hires_pass": False},
            "review": {
                "script": False, "cut": True, "max_retries_per_shot": 0,
                "reviewer": "external", "min_vision_coverage": 0.8,
            },
            "intro": {"enabled": True},
        },
    }

    return {
        "cfg": cfg, "base": base, "out_dir": str(out_dir),
        "review_dir": review_dir, "calls": calls,
        "intro_invocations": intro_invocations,
    }


def _review_log(run):
    path = os.path.join(run["review_dir"], "review_log.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------- cut QC enabled ---
def test_gate_blocks_promotion_below_threshold(gated_run):
    from pipeline.kidsong.generate import generate_kidsong

    run = gated_run
    _write_response(run["review_dir"], "s00_a0", True)
    # s01_a0 left uncovered -> 1/2 = 0.5 coverage, below the 0.8 threshold.

    result = generate_kidsong(cfg=run["cfg"], resume_base=run["base"])

    assert result["status"] == "needs_review"
    assert result["video_path"].endswith(".staging.mp4")
    assert not os.path.exists(os.path.join(run["out_dir"], "Final", run["base"] + ".mp4"))
    assert run["intro_invocations"] == []
    assert "intro" not in run["calls"]

    gate_entries = [e for e in _review_log(run) if e.get("gate") == "vision_coverage"]
    assert len(gate_entries) == 1
    entry = gate_entries[0]
    assert entry["coverage"] == 0.5
    assert entry["threshold"] == 0.8
    assert entry["promoted"] is False
    assert "s01_a0" in entry["uncovered"]


def test_gate_allows_promotion_at_or_above_threshold(gated_run):
    from pipeline.kidsong.generate import generate_kidsong

    run = gated_run
    _write_response(run["review_dir"], "s00_a0", True)
    _write_response(run["review_dir"], "s01_a0", True)

    result = generate_kidsong(cfg=run["cfg"], resume_base=run["base"])

    assert result["status"] == "approved"
    assert os.path.exists(result["video_path"])
    assert run["intro_invocations"] == [result["video_path"]]

    gate_entries = [e for e in _review_log(run) if e.get("gate") == "vision_coverage"]
    assert len(gate_entries) == 1
    assert gate_entries[0]["coverage"] == 1.0
    assert gate_entries[0]["promoted"] is True


# -------------------------------------------------------- cut QC disabled ---
def test_gate_blocks_promotion_with_cut_qc_disabled(gated_run):
    """Site 2 (kidsong.review.cut == False) must also be gated."""
    from pipeline.kidsong.generate import generate_kidsong

    run = gated_run
    run["cfg"]["kidsong"]["review"]["cut"] = False
    # No responses at all -> 0.0 coverage.

    result = generate_kidsong(cfg=run["cfg"], resume_base=run["base"])

    assert result["status"] == "needs_review"
    assert run["intro_invocations"] == []
    gate_entries = [e for e in _review_log(run) if e.get("gate") == "vision_coverage"]
    assert len(gate_entries) == 1
    assert gate_entries[0]["coverage"] == 0.0
    assert gate_entries[0]["promoted"] is False


def test_gate_allows_promotion_with_cut_qc_disabled(gated_run):
    from pipeline.kidsong.generate import generate_kidsong

    run = gated_run
    run["cfg"]["kidsong"]["review"]["cut"] = False
    _write_response(run["review_dir"], "s00_a0", True)
    _write_response(run["review_dir"], "s01_a0", True)

    result = generate_kidsong(cfg=run["cfg"], resume_base=run["base"])

    assert result["status"] == "approved"
    assert run["intro_invocations"] == [result["video_path"]]


# ------------------------------------------------------------- disabled gate ---
def test_gate_skipped_when_reviewer_is_heuristic(gated_run):
    """No vision path exists by configuration -- the gate must not refuse
    every episode just because nothing was ever vision-reviewed."""
    from pipeline.kidsong.generate import generate_kidsong

    run = gated_run
    run["cfg"]["kidsong"]["review"]["reviewer"] = "heuristic"
    # No responses at all -- would be 0.0 coverage if the gate ran.

    result = generate_kidsong(cfg=run["cfg"], resume_base=run["base"])

    assert result["status"] == "approved"
    assert run["intro_invocations"] == [result["video_path"]]
    gate_entries = [e for e in _review_log(run) if e.get("gate") == "vision_coverage"]
    assert gate_entries == []


def test_gate_skipped_when_min_vision_coverage_is_zero(gated_run):
    """Full opt-out is still possible — but it now takes BOTH knobs: the
    coverage gate (min_vision_coverage=0) and the group-vision gate
    (require_group_vision=false), which are deliberately independent."""
    from pipeline.kidsong.generate import generate_kidsong

    run = gated_run
    run["cfg"]["kidsong"]["review"]["min_vision_coverage"] = 0
    run["cfg"]["kidsong"]["review"]["require_group_vision"] = False
    # No responses at all -- would be 0.0 coverage if either gate ran.

    result = generate_kidsong(cfg=run["cfg"], resume_base=run["base"])

    assert result["status"] == "approved"
    assert run["intro_invocations"] == [result["video_path"]]
    gate_entries = [e for e in _review_log(run)
                    if e.get("gate") in ("vision_coverage", "group_vision")]
    assert gate_entries == []


def test_group_vision_gate_alone_blocks_promotion(gated_run):
    """min_vision_coverage opted out but require_group_vision left at its
    default: an unreviewed GROUP take still blocks promotion (the group gate
    is exactly for the shots the heuristic cannot judge)."""
    from pipeline.kidsong.generate import generate_kidsong

    run = gated_run
    run["cfg"]["kidsong"]["review"]["min_vision_coverage"] = 0
    # require_group_vision defaults to True — not set on purpose.

    result = generate_kidsong(cfg=run["cfg"], resume_base=run["base"])

    entries = [e for e in _review_log(run) if e.get("gate") == "group_vision"]
    if any(not e.get("promoted") for e in entries):
        assert result["status"] == "needs_review"
        assert run["intro_invocations"] == []
    else:
        # Fixture plans no group shots at all -> gate passes vacuously.
        assert result["status"] == "approved"


def test_gate_defaults_to_point_eight_when_unset(gated_run):
    """kidsong.review.min_vision_coverage absent from cfg -> the 0.8 default
    from config.example.json applies (generate.py reads it with a literal
    0.8 fallback so this holds even for a bare/unit-test cfg dict)."""
    from pipeline.kidsong.generate import generate_kidsong

    run = gated_run
    del run["cfg"]["kidsong"]["review"]["min_vision_coverage"]
    _write_response(run["review_dir"], "s00_a0", True)
    # s01_a0 uncovered -> 0.5, below the implied 0.8 default.

    result = generate_kidsong(cfg=run["cfg"], resume_base=run["base"])

    assert result["status"] == "needs_review"
    gate_entries = [e for e in _review_log(run) if e.get("gate") == "vision_coverage"]
    assert gate_entries[0]["threshold"] == 0.8
