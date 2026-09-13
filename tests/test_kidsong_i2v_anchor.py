"""Stage 0 (QM-001/IMP-005/IMP-006) regression tests: prove the reference
anchor (``force_i2v`` -> ``_reference_for_single_subject`` -> i2v render) can
actually fire, and that the per-take ``i2v_anchored`` telemetry (generate.py
~2225-2242) truthfully reflects which attempts rendered i2v-anchored rather
than t2v.

Built on the same fixture shape as test_kidsong_infra_retry.py's `_make_run`:
a resumed run with exactly ONE pending shot so the render loop actually
executes, with every other GPU/media stage faked. CPU-only, no GPU, no
network, no ffmpeg, no real ComfyUI -- except a real Pillow PNG for the cast
reference image, since `_reference_for_single_subject` -> `refs.reference_for`
requires a real, non-empty file on disk, and `_fit_reference_card` opens it
with PIL to check/cover-fit its dimensions.
"""
import json
import os
import struct
import wave

import pytest

from pipeline.kidsong import refs, runstate
from pipeline.kidsong.comfy import ComfyClient as _RealComfyClient
from pipeline.kidsong.render_style import i2v_workflow_name, workflow_name

_REAL_LOAD_WORKFLOW = _RealComfyClient.load_workflow
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _write_wav(path, seconds=2.0, rate=8000):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(struct.pack("<h", 0) * int(rate * seconds))


def _write_reference_png(path, size=(768, 1024)):
    """A real, decodable PNG -- `_fit_reference_card` opens it with PIL to
    check/cover-fit dims, so a stub/empty file would raise there."""
    from PIL import Image

    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("RGB", size, (200, 150, 100)).save(path)


def _make_run(tmp_path, monkeypatch, shot_overrides, verdicts,
              max_retries_per_shot=2, reference_char_id=None,
              reference_anchor_cfg=None):
    """Same fixture shape as test_kidsong_infra_retry.py's `_make_run`,
    parameterized on the single pending shot's overrides (characters,
    shot_type, ...) and the sequence of verdicts `FakeReviewer.review()`
    returns, one per call (extra calls repeat the last verdict).

    `reference_char_id`, when given, writes a real cast-reference PNG to disk
    at `refs.reference_path(cfg, reference_char_id)` BEFORE the run, so
    `_reference_for_single_subject` has something to anchor on.
    """
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    base = "20260802-093000-kidsong-i2v-anchor-test"
    shots_dir = runstate.shots_dir_path(str(out_dir), base)
    os.makedirs(shots_dir, exist_ok=True)

    _write_wav(out_dir / f"{base}.wav")
    song = {
        "title": "I2V Anchor Test Song",
        "description": "d",
        "tags": ["t"],
        "characters": "Three toddlers: Zuri; Kofi; Nala",
        "verses": [{"scene": "bathroom", "lines": ["a", "b"]}],
    }
    (out_dir / f"{base}-song.json").write_text(json.dumps(song), encoding="utf-8")

    shot = {
        "id": "s00", "verse": 0, "start": 0.0, "end": 2.0,
        "shot_type": "closeup", "characters": ["Kofi"],
        "action": "Kofi brushes his teeth with a big smile",
        "setting": "a bright cheerful bathroom", "camera": "static",
        "reuse_of": None, "seed": 1000, "status": "planned",
    }
    shot.update(shot_overrides)
    runstate.save_shotlist(str(out_dir), base, {"shots": [shot]})

    cfg = {
        "_root": _REPO_ROOT,
        "paths": {"output_dir": str(out_dir)},
        "video": {"max_seconds": 60},
        "captions": {"uppercase": True},
        "kidsong": {
            "mode": "director",
            "singer": "ace",
            "seed": 20260717,
            "shot": {"fps": 24, "max_frames": 241, "hires_pass": False, "style_trigger": "P1x4r"},
            "review": {
                "script": False, "cut": True,
                "max_retries_per_shot": max_retries_per_shot,
                "comfy_infra_retries": 3,
            },
            "intro": {"enabled": False},
        },
    }
    if reference_anchor_cfg is not None:
        cfg["kidsong"]["reference_anchor"] = reference_anchor_cfg

    if reference_char_id:
        _write_reference_png(refs.reference_path(cfg, reference_char_id))

    render_calls = []
    stage_calls = []

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
            pass

        def render(self, workflow, patches, out_path):
            render_calls.append((workflow, dict(patches)))
            with open(out_path, "wb") as f:
                f.write(b"\0" * 1024)

        def free(self):
            pass

        def restart_if_hung(self):
            pass

        def stage_input_image(self, src_path):
            staged = f"staged_{os.path.basename(src_path)}"
            stage_calls.append((src_path, staged))
            return staged

    class FakeReviewer:
        def __init__(self):
            self.calls = 0

        def review(self, shot, video_path, out_dir=None):
            v = verdicts[min(self.calls, len(verdicts) - 1)]
            self.calls += 1
            return dict(v)

    logged_entries = []

    def fake_review_shots_log(path, entries):
        logged_entries.extend(entries)
        return path

    def fake_assemble(cfg, cut_list, renders, voice, words, duration, out_path, **kw):
        with open(out_path, "wb") as f:
            f.write(b"\0" * 4096)

    def fake_review_cut(staging_path, cut_list, beats, words, duration, cfg, review_dir):
        return {"accept": True, "score": 1.0, "reasons": [], "retry_hints": {}}

    def fake_prepend_intro(video_path, cfg, on_progress=None):
        return {"applied": False, "intro_seconds": 0.0, "reason": "disabled for test"}

    monkeypatch.setattr(comfy_mod, "ComfyClient", FakeComfyClient)
    monkeypatch.setattr(review_mod, "get_reviewer", lambda cfg: FakeReviewer())
    monkeypatch.setattr(review_mod, "contact_sheet", lambda *a, **k: None)
    monkeypatch.setattr(review_mod, "review_shots_log", fake_review_shots_log)
    monkeypatch.setattr(captions_mod, "get_word_timestamps", lambda *a, **k: [])
    monkeypatch.setattr(sing_mod, "verse_times_from_words", lambda *a, **k: [(0.0, 2.0)])
    monkeypatch.setattr(edit_mod, "beat_grid", lambda path: [0.0, 1.0, 2.0])
    monkeypatch.setattr(edit_mod, "build_cut_list", lambda *a, **k: [])
    monkeypatch.setattr(edit_mod, "assemble", fake_assemble)
    monkeypatch.setattr(cut_qc_mod, "review_cut", fake_review_cut)
    monkeypatch.setattr(edit_mod, "prepend_intro", fake_prepend_intro)

    def _sing_must_not_run(*a, **k):
        raise AssertionError("resume must not re-sing")

    monkeypatch.setattr(sing_mod, "sing_song", _sing_must_not_run)
    monkeypatch.setattr("pipeline.kidsong.generate.time.sleep", lambda s: None)

    return {
        "cfg": cfg, "base": base, "out_dir": str(out_dir),
        "render_calls": render_calls, "stage_calls": stage_calls,
        "logged_entries": logged_entries,
    }


def _shot_entries(run, shot_id="s00"):
    """`review_shots_log` is also called for the cut-level QC verdict (a dict
    with a `stage` key, no `shot` key) -- filter `logged_entries` down to the
    per-take shot log this suite actually cares about, in render order."""
    return [e for e in run["logged_entries"] if e.get("shot") == shot_id]


def test_force_i2v_on_a_single_subject_shot_anchors_on_the_reference(tmp_path, monkeypatch):
    """A solo shot rejected with `force_i2v=true` and a reference PNG on
    disk: the SECOND attempt must actually render image-to-video (the i2v
    workflow, with INPUT_IMAGE/SIGMAS patched from the staged reference), and
    that take's `i2v_anchored` telemetry must be true."""
    from pipeline.kidsong.generate import generate_kidsong

    run = _make_run(
        tmp_path, monkeypatch,
        shot_overrides={"characters": ["Kofi"], "shot_type": "closeup"},
        verdicts=[
            {
                "accept": False, "score": 0.2,
                "reasons": ["cast integrity: non-cast child in Kofi's slot"],
                "retry_hints": {"force_i2v": True},
            },
            {"accept": True, "score": 0.9, "reasons": [], "retry_hints": {}},
        ],
        reference_char_id="kofi",
    )

    result = generate_kidsong(cfg=run["cfg"], resume_base=run["base"])

    assert result["status"] == "approved"
    assert len(run["render_calls"]) == 2

    t2v_workflow = workflow_name(run["cfg"])
    i2v_workflow = i2v_workflow_name(run["cfg"])

    attempt0_workflow, attempt0_patches = run["render_calls"][0]
    attempt1_workflow, attempt1_patches = run["render_calls"][1]
    assert attempt0_workflow == t2v_workflow
    assert "INPUT_IMAGE" not in attempt0_patches
    assert attempt1_workflow == i2v_workflow
    assert "INPUT_IMAGE" in attempt1_patches
    assert "SIGMAS" in attempt1_patches

    # The reference PNG was actually staged (once escalation fired).
    assert len(run["stage_calls"]) == 1

    entries = _shot_entries(run)
    assert len(entries) == 2
    assert entries[0]["i2v_anchored"] is False
    assert entries[0]["i2v_source_kind"] is None
    assert entries[1]["i2v_anchored"] is True
    assert entries[1]["i2v_source_kind"] == "reference"


def test_force_i2v_on_a_group_shot_stays_t2v_and_is_not_marked_anchored(tmp_path, monkeypatch):
    """The documented no-op path: a single reference can only pin ONE
    identity, so a multi-child shot's `force_i2v=true` must never escalate --
    every attempt stays t2v and `i2v_anchored` stays false throughout."""
    from pipeline.kidsong.generate import generate_kidsong

    run = _make_run(
        tmp_path, monkeypatch,
        shot_overrides={"characters": ["Kofi", "Zuri"], "shot_type": "wide"},
        verdicts=[
            {
                "accept": False, "score": 0.2,
                "reasons": ["duplicate characters"],
                "retry_hints": {"force_i2v": True},
            },
            {"accept": True, "score": 0.9, "reasons": [], "retry_hints": {}},
        ],
        # Deliberately no reference PNG written -- even if one existed,
        # expected_child_count == 2 must short-circuit before reference
        # lookup. Proves the group-shot case is a no-op independent of
        # whether a reference happens to be on disk.
    )

    result = generate_kidsong(cfg=run["cfg"], resume_base=run["base"])

    assert result["status"] == "approved"
    assert len(run["render_calls"]) == 2

    t2v_workflow = workflow_name(run["cfg"])
    for workflow, patches in run["render_calls"]:
        assert workflow == t2v_workflow
        assert "INPUT_IMAGE" not in patches

    assert run["stage_calls"] == []

    entries = _shot_entries(run)
    assert len(entries) == 2
    assert entries[0]["i2v_anchored"] is False
    assert entries[1]["i2v_anchored"] is False


def test_i2v_anchored_is_false_on_a_take_that_never_escalated(tmp_path, monkeypatch):
    """Baseline case: a take accepted on its first attempt, no force_i2v ever
    emitted -- `i2v_anchored` must be false and `i2v_source_kind` must be
    None, matching IMP-006's baseline-must-be-0 framing."""
    from pipeline.kidsong.generate import generate_kidsong

    run = _make_run(
        tmp_path, monkeypatch,
        shot_overrides={"characters": ["Kofi"], "shot_type": "closeup"},
        verdicts=[{"accept": True, "score": 1.0, "reasons": [], "retry_hints": {}}],
        reference_char_id="kofi",
    )

    result = generate_kidsong(cfg=run["cfg"], resume_base=run["base"])

    assert result["status"] == "approved"
    entries = _shot_entries(run)
    assert len(entries) == 1
    assert entries[0]["i2v_anchored"] is False
    assert entries[0]["i2v_source_kind"] is None


def test_anchoring_is_not_retroactive(tmp_path, monkeypatch):
    """The pre-escalation attempt(s) must stay `i2v_anchored=false` forever
    -- escalating mid-shot must never rewrite an earlier take's already-logged
    telemetry. Three attempts: reject (fires force_i2v) -> reject (stays
    anchored) -> accept, so both a pre-anchor AND a post-anchor entry exist
    to compare."""
    from pipeline.kidsong.generate import generate_kidsong

    run = _make_run(
        tmp_path, monkeypatch,
        shot_overrides={"characters": ["Kofi"], "shot_type": "closeup"},
        verdicts=[
            {
                "accept": False, "score": 0.2,
                "reasons": ["cast integrity: non-cast child in Kofi's slot"],
                "retry_hints": {"force_i2v": True},
            },
            {
                "accept": False, "score": 0.5,
                "reasons": ["strobe/glitch"],
                "retry_hints": {"seed_bump": True},
            },
            {"accept": True, "score": 0.9, "reasons": [], "retry_hints": {}},
        ],
        max_retries_per_shot=3,
        reference_char_id="kofi",
    )

    result = generate_kidsong(cfg=run["cfg"], resume_base=run["base"])

    assert result["status"] == "approved"
    entries = _shot_entries(run)
    assert len(entries) == 3
    # Attempt 0 rendered BEFORE force_i2v was consumed -- must not be
    # retroactively marked anchored.
    assert entries[0]["i2v_anchored"] is False
    # Attempts 1 and 2 render after escalation and must both be anchored.
    assert entries[1]["i2v_anchored"] is True
    assert entries[2]["i2v_anchored"] is True
