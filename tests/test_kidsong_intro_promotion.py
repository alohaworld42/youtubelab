"""Tests wiring the fixed ZubiBop branding intro (pipeline.kidsong.edit.prepend_intro,
already public/tested/a no-op when disabled) into the two promote-to-Final sites in
pipeline.kidsong.generate._generate_director.

Ordering matters: prepend_intro MUST run after the cut is assembled (color grade is
baked in during assemble()) and after cut QC (cut_qc.review_cut compares video
duration against SONG duration with 0.5s tolerance and probes for black seams at
song-time offsets — prepending 4s any earlier would fail that gate on every render).

CPU-only, no GPU, no network, no ffmpeg, no ComfyUI: this drives generate_kidsong via
resume_base with every shot already "rendered" on disk (so the render loop has zero
pending work and never touches ComfyUI), and every other GPU/media-processing stage
(Whisper, beat grid, cut assembly, cut QC) monkeypatched — the same shape
tests/test_kidsong_runlog_resume.py's `resume_run` fixture uses for its own
GPU-free end-to-end runs.
"""
import json
import os
import struct
import wave

import pytest

from pipeline.kidsong import runstate
from pipeline.kidsong.comfy import ComfyClient as _RealComfyClient
from pipeline.kidsong.edit import prepend_intro as _real_prepend_intro

# Captured at collection time, before any test's monkeypatch runs, so the
# "disabled path" test can restore the true function even though the fixture
# already replaced `edit.prepend_intro` with a mock by the time the test body
# runs (an in-test `from ... import prepend_intro` at that point would just
# re-capture the mock).
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


@pytest.fixture
def promotable_run(tmp_path, monkeypatch):
    """A director run whose 2 shots are already fully rendered — the render
    loop has nothing pending, so it never constructs a real network call to
    ComfyUI. Every stage between "shots on disk" and "promote to Final/" is
    faked; `cut_enabled` and `intro_enabled` are set per-test by the caller.
    """
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    base = "20260720-091500-kidsong-intro-test"
    shots_dir = runstate.shots_dir_path(str(out_dir), base)

    _write_wav(out_dir / f"{base}.wav")
    song = {
        "title": "Intro Test Song",
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

    calls = []  # shared call-order tracker: "assemble" / "review_cut" / "intro"
    intro_invocations = []  # (video_path, cfg) captured per prepend_intro call

    import pipeline.captions as captions_mod
    import pipeline.kidsong.comfy as comfy_mod
    import pipeline.kidsong.cut_qc as cut_qc_mod
    import pipeline.kidsong.edit as edit_mod
    import pipeline.kidsong.review as review_mod
    import pipeline.kidsong.sing as sing_mod

    class FakeComfyClient:
        """No shots are pending in this fixture, so `render`/`ensure_up` must
        never be called — only `load_workflow` (baseline-negative read) runs,
        and it reads the real repo workflow JSON off local disk (no network)."""

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
            return {"accept": True, "score": 1.0, "reasons": [], "retry_hints": {}}

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
    monkeypatch.setattr(review_mod, "review_shots_log", lambda path, entries: path)
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
            "review": {"script": False, "cut": True, "max_retries_per_shot": 0},
            "intro": {"enabled": True},
        },
    }

    return {
        "cfg": cfg, "base": base, "out_dir": str(out_dir),
        "calls": calls, "intro_invocations": intro_invocations,
    }


# ------------------------------------------------------- mocked prepend_intro ---
def test_intro_called_at_promote_site_with_cut_qc_enabled(promotable_run):
    """Site 1: `if cv.get("accept"):` branch."""
    from pipeline.kidsong.generate import generate_kidsong

    run = promotable_run
    result = generate_kidsong(cfg=run["cfg"], resume_base=run["base"])

    assert result["status"] == "approved"
    assert len(run["intro_invocations"]) == 1
    assert run["intro_invocations"][0] == result["video_path"]


def test_intro_called_at_promote_site_with_cut_qc_disabled(promotable_run):
    """Site 2: the `else:` branch when kidsong.review.cut is False."""
    from pipeline.kidsong.generate import generate_kidsong

    run = promotable_run
    run["cfg"]["kidsong"]["review"]["cut"] = False

    result = generate_kidsong(cfg=run["cfg"], resume_base=run["base"])

    assert result["status"] == "approved"
    assert len(run["intro_invocations"]) == 1
    assert run["intro_invocations"][0] == result["video_path"]
    # cut QC was skipped for this site, but the intro still ran.
    assert "review_cut" not in run["calls"]
    assert "intro" in run["calls"]


def test_intro_runs_after_assemble_and_after_cut_qc(promotable_run):
    """Ordering: assemble (bakes in the grade) -> cut QC -> intro. A prepend
    any earlier would shift the video-time/song-time relationship cut_qc's
    duration+black-seam checks depend on."""
    from pipeline.kidsong.generate import generate_kidsong

    run = promotable_run
    generate_kidsong(cfg=run["cfg"], resume_base=run["base"])

    assert run["calls"] == ["assemble", "review_cut", "intro"]


def test_intro_runs_after_assemble_when_cut_qc_disabled(promotable_run):
    from pipeline.kidsong.generate import generate_kidsong

    run = promotable_run
    run["cfg"]["kidsong"]["review"]["cut"] = False

    generate_kidsong(cfg=run["cfg"], resume_base=run["base"])

    assert run["calls"] == ["assemble", "intro"]


# ------------------------------------------------------------- disabled path ---
def test_disabled_intro_is_a_clean_noop_through_the_real_function(promotable_run, monkeypatch):
    """Exercise the REAL prepend_intro (not the mock) with kidsong.intro.enabled
    False, so this actually proves generate.py's call site degrades cleanly —
    no exception, no sidecar, video left exactly as assemble() wrote it."""
    import pipeline.kidsong.edit as edit_mod
    from pipeline.kidsong.generate import generate_kidsong

    # Undo the fixture's mock for this one test — use the real function
    # (captured at module-collection time, before any monkeypatching).
    monkeypatch.setattr(edit_mod, "prepend_intro", _real_prepend_intro)

    run = promotable_run
    run["cfg"]["kidsong"]["intro"] = {"enabled": False}

    result = generate_kidsong(cfg=run["cfg"], resume_base=run["base"])

    assert result["status"] == "approved"
    assert os.path.exists(result["video_path"])
    # The fake assemble() wrote exactly 4096 zero bytes; a real (non-mocked)
    # prepend_intro that is a true no-op must leave that untouched.
    with open(result["video_path"], "rb") as f:
        content = f.read()
    assert content == b"\0" * 4096
    assert not os.path.exists(result["video_path"] + ".intro.json")
    # The mocked-out call tracker never saw "intro" fire, either.
    assert "intro" not in run["calls"]
