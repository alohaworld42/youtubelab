"""Tests for Phase 3 reference anchoring in pipeline.kidsong.generate:
consuming the (previously dead) `force_i2v` reviewer hint by re-rendering a
single-subject shot image-to-video from the cast member's canonical reference.

Two layers:
  * `_reference_for_single_subject` — the escalation GATE (unit): only fires for
    a single-subject shot, with a reference on disk, when the feature is on.
  * the render loop (integration, via a resumed one-pending-shot run with a fake
    ComfyClient, the same harness shape as test_kidsong_negative_wiring.py):
    proves the shot actually re-renders via the i2v graph with INPUT_IMAGE +
    SIGMAS, and — the load-bearing regression guard — that with the feature OFF
    or no reference present the render is byte-identical to today (t2v, no
    staging, no i2v).

CPU-only, no GPU, no network, no ffmpeg, no real ComfyUI.
"""
import json
import os
import struct
import wave

import pytest

from pipeline.kidsong import refs, runstate
from pipeline.kidsong.comfy import ComfyClient as _RealComfyClient
from pipeline.kidsong.generate import _reference_for_single_subject

_REAL_LOAD_WORKFLOW = _RealComfyClient.load_workflow
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ===================================================== gate: _reference_for ===
def _gate_cfg(tmp_path, enabled=True):
    cfg = {
        "_root": str(tmp_path),
        "paths": {"output_dir": str(tmp_path / "output")},
        "kidsong": {"seed": 20260717, "reference_anchor": {"enabled": enabled}},
    }
    return cfg


def _place_reference(cfg, char_id="kofi"):
    dest = refs.reference_path(cfg, char_id)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "wb") as f:
        f.write(b"PNGDATA")
    return dest


def test_gate_returns_reference_for_single_subject_with_reference(tmp_path):
    cfg = _gate_cfg(tmp_path)
    dest = _place_reference(cfg, "kofi")
    shot = {"id": "s0", "shot_type": "closeup", "characters": ["Kofi"]}
    assert _reference_for_single_subject(shot, cfg) == dest


def test_gate_none_when_feature_disabled(tmp_path):
    cfg = _gate_cfg(tmp_path, enabled=False)
    _place_reference(cfg, "kofi")
    shot = {"id": "s0", "shot_type": "closeup", "characters": ["Kofi"]}
    # disabled -> inert even though a reference exists (byte-identical guarantee)
    assert _reference_for_single_subject(shot, cfg) is None


def test_gate_none_when_no_reference_on_disk(tmp_path):
    cfg = _gate_cfg(tmp_path)  # no reference placed
    shot = {"id": "s0", "shot_type": "closeup", "characters": ["Kofi"]}
    assert _reference_for_single_subject(shot, cfg) is None


def test_gate_none_for_multi_subject_shot(tmp_path):
    cfg = _gate_cfg(tmp_path)
    _place_reference(cfg, "kofi")
    _place_reference(cfg, "nala")
    # two children resolve -> no single correct image to anchor
    shot = {"id": "s0", "shot_type": "wide", "characters": ["Kofi", "Nala"]}
    assert _reference_for_single_subject(shot, cfg) is None


def test_gate_default_enabled_is_true(tmp_path):
    # reference_anchor block absent entirely -> defaults ON, so a placed
    # reference for a single-subject shot still escalates.
    cfg = {"_root": str(tmp_path), "paths": {"output_dir": str(tmp_path / "output")},
           "kidsong": {"seed": 20260717}}
    _place_reference(cfg, "kofi")
    shot = {"id": "s0", "shot_type": "closeup", "characters": ["Kofi"]}
    assert _reference_for_single_subject(shot, cfg) is not None


# ===================================================== integration harness ====
def _write_wav(path, seconds=2.0, rate=8000):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(struct.pack("<h", 0) * int(rate * seconds))


def _make_run(tmp_path, monkeypatch, *, enabled, with_reference, always=False,
              accept_first=False):
    """A resumed director run with ONE pending single-subject (Kofi closeup)
    shot. By default the reviewer rejects the first take with force_i2v and
    accepts the second — so the render loop runs exactly twice and the escalation
    decision is exercised. `accept_first=True` accepts the FIRST take instead, so
    the loop runs once and the PROACTIVE (`reference_anchor.always`) decision at
    attempt 0 is what's under test. `always` sets that config flag. Returns
    captured render + staging calls."""
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    base = "20260721-090000-kidsong-reference-anchor-test"
    shots_dir = runstate.shots_dir_path(str(out_dir), base)
    os.makedirs(shots_dir, exist_ok=True)

    _write_wav(out_dir / f"{base}.wav")
    song = {
        "title": "Anchor Test", "description": "d", "tags": ["t"],
        "characters": "Three toddlers: Zuri; Kofi; Nala",
        "verses": [{"scene": "bathroom", "lines": ["a", "b"]}],
    }
    (out_dir / f"{base}-song.json").write_text(json.dumps(song), encoding="utf-8")

    shot = {
        "id": "s00", "verse": 0, "start": 0.0, "end": 2.0,
        "shot_type": "closeup", "characters": ["Kofi"],
        "action": "Kofi brushes his teeth", "setting": "a bright bathroom",
        "camera": "static", "reuse_of": None, "seed": 1000, "status": "planned",
    }
    runstate.save_shotlist(str(out_dir), base, {"shots": [shot]})

    cfg = {
        "_root": _REPO_ROOT,
        "paths": {"output_dir": str(out_dir)},
        "video": {"max_seconds": 60},
        "captions": {"uppercase": True},
        "kidsong": {
            "mode": "director", "singer": "ace", "seed": 20260717,
            "shot": {"fps": 24, "max_frames": 241, "hires_pass": False,
                     "style_trigger": "P1x4r"},
            "review": {"script": False, "cut": True, "max_retries_per_shot": 1},
            "intro": {"enabled": False},
            "reference_anchor": {"enabled": enabled, "always": always},
        },
    }

    if with_reference:
        dest = refs.reference_path(cfg, "kofi")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as f:
            f.write(b"PNGREF")

    render_calls = []   # (workflow_name, patches)
    staged = []         # src paths passed to stage_input_image

    import pipeline.captions as captions_mod
    import pipeline.kidsong.comfy as comfy_mod
    import pipeline.kidsong.cut_qc as cut_qc_mod
    import pipeline.kidsong.edit as edit_mod
    import pipeline.kidsong.review as review_mod
    import pipeline.kidsong.sing as sing_mod

    class FakeComfyClient:
        def __init__(self, cfg):
            self._proc = None

        def load_workflow(self, workflow_name):
            return _REAL_LOAD_WORKFLOW(self, workflow_name)

        def ensure_up(self):
            pass

        def stage_input_image(self, src_path):
            staged.append(src_path)
            return f"staged_{len(staged)}.png"

        def render(self, workflow, patches, out_path):
            render_calls.append((workflow, patches))
            with open(out_path, "wb") as f:
                f.write(b"\0" * 1024)

        def free(self):
            pass

    class FakeReviewer:
        """Reject the first take with force_i2v, accept the second — unless
        `accept_first`, in which case accept the first take outright (so the
        proactive-anchor decision, not the reject-escalation, is what's tested)."""

        def __init__(self):
            self.n = 0

        def review(self, shot, video_path, out_dir=None):
            self.n += 1
            if self.n == 1 and not accept_first:
                return {"accept": False, "score": 0.2,
                        "reasons": ["duplicate characters"],
                        "retry_hints": {"seed_bump": False, "simplify_action": False,
                                        "force_i2v": True}}
            return {"accept": True, "score": 1.0, "reasons": [], "retry_hints": {}}

    def fake_assemble(cfg, cut_list, renders, voice, words, duration, out_path, **kw):
        with open(out_path, "wb") as f:
            f.write(b"\0" * 4096)

    monkeypatch.setattr(comfy_mod, "ComfyClient", FakeComfyClient)
    # Isolate the anchor GATE (what these tests exercise) from the reference
    # BOOTSTRAP added to the render loop (tested separately below): here
    # ensure_reference only LOOKS UP an existing reference, never generates one,
    # so each test's `with_reference` setup is preserved and no stray reference
    # render lands in `render_calls`.
    monkeypatch.setattr("pipeline.kidsong.refs.ensure_reference",
                        lambda client, cfg, cid: refs.reference_for(cfg, cid))
    monkeypatch.setattr(review_mod, "get_reviewer", lambda cfg: FakeReviewer())
    monkeypatch.setattr(review_mod, "contact_sheet", lambda *a, **k: None)
    monkeypatch.setattr(review_mod, "review_shots_log", lambda path, entries: path)
    monkeypatch.setattr(captions_mod, "get_word_timestamps", lambda *a, **k: [])
    monkeypatch.setattr(sing_mod, "verse_times_from_words", lambda *a, **k: [(0.0, 2.0)])
    monkeypatch.setattr(edit_mod, "beat_grid", lambda path: [0.0, 1.0, 2.0])
    monkeypatch.setattr(edit_mod, "build_cut_list", lambda *a, **k: [])
    monkeypatch.setattr(edit_mod, "assemble", fake_assemble)
    monkeypatch.setattr(cut_qc_mod, "review_cut",
                        lambda *a, **k: {"accept": True, "score": 1.0, "reasons": [],
                                          "retry_hints": {}})
    monkeypatch.setattr(edit_mod, "prepend_intro",
                        lambda *a, **k: {"applied": False, "intro_seconds": 0.0})
    monkeypatch.setattr(sing_mod, "sing_song",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no re-sing")))

    return {"cfg": cfg, "base": base, "render_calls": render_calls, "staged": staged}


# ===================================================== integration tests ======
def test_force_i2v_reference_present_escalates_second_take_to_i2v(tmp_path, monkeypatch):
    from pipeline.kidsong.generate import generate_kidsong

    run = _make_run(tmp_path, monkeypatch, enabled=True, with_reference=True)
    generate_kidsong(cfg=run["cfg"], resume_base=run["base"])

    calls = run["render_calls"]
    assert len(calls) == 2, "expected a rejected t2v take then an i2v re-render"

    (wf0, p0), (wf1, p1) = calls
    # first take: plain t2v (base graph, hires_pass off)
    assert wf0 == "ltx23_t2v_toon"
    assert "INPUT_IMAGE" not in p0 and "SIGMAS" not in p0
    # second take: escalated to the i2v graph, anchored on the staged reference
    assert wf1 == "ltx23_i2v_toon"
    assert p1["INPUT_IMAGE"] == {"image": "staged_1.png"}
    assert "SIGMAS" in p1 and p1["SIGMAS"]["sigmas"].startswith("1.0000")
    # the reference PNG was staged exactly once and reused
    assert run["staged"] == [refs.reference_path(run["cfg"], "kofi")]
    # the common patch titles still ride along on the i2v render
    for title in ("PROMPT", "NEGATIVE", "SEED", "WIDTH", "HEIGHT", "FRAMES",
                  "LORA_STYLE", "FILENAME_PREFIX"):
        assert title in p1


def test_force_i2v_reference_present_escalates_second_take_to_i2v_hires(
    tmp_path, monkeypatch,
):
    """Same escalation as the test above, but with kidsong.shot.hires_pass
    True: the escalated i2v render must follow the hires toggle exactly like
    the t2v path does, landing on the _hires i2v graph instead of the base
    one -- before i2v_workflow_name existed, an anchored shot silently
    skipped the 2x refine pass the config asked for."""
    from pipeline.kidsong.generate import generate_kidsong

    run = _make_run(tmp_path, monkeypatch, enabled=True, with_reference=True)
    run["cfg"]["kidsong"]["shot"]["hires_pass"] = True
    generate_kidsong(cfg=run["cfg"], resume_base=run["base"])

    calls = run["render_calls"]
    assert len(calls) == 2, "expected a rejected t2v take then an i2v re-render"

    (wf0, p0), (wf1, p1) = calls
    assert wf0 == "ltx23_t2v_toon_hires"
    assert wf1 == "ltx23_i2v_toon_hires"
    assert p1["INPUT_IMAGE"] == {"image": "staged_1.png"}
    assert "SIGMAS" in p1 and p1["SIGMAS"]["sigmas"].startswith("1.0000")


def test_force_i2v_disabled_is_byte_identical_t2v(tmp_path, monkeypatch):
    from pipeline.kidsong.generate import generate_kidsong

    run = _make_run(tmp_path, monkeypatch, enabled=False, with_reference=True)
    generate_kidsong(cfg=run["cfg"], resume_base=run["base"])

    calls = run["render_calls"]
    assert len(calls) == 2
    for wf, patches in calls:
        assert wf == "ltx23_t2v_toon"          # never switches graph
        assert "INPUT_IMAGE" not in patches    # never anchors
        assert "SIGMAS" not in patches
    assert run["staged"] == []                 # nothing staged -> nothing leaks


def test_force_i2v_no_reference_stays_on_t2v(tmp_path, monkeypatch):
    from pipeline.kidsong.generate import generate_kidsong

    run = _make_run(tmp_path, monkeypatch, enabled=True, with_reference=False)
    generate_kidsong(cfg=run["cfg"], resume_base=run["base"])

    calls = run["render_calls"]
    assert len(calls) == 2
    for wf, patches in calls:
        assert wf == "ltx23_t2v_toon"
        assert "INPUT_IMAGE" not in patches
    assert run["staged"] == []


# ================================================ proactive always-anchor =====
def test_always_anchor_renders_the_first_take_i2v(tmp_path, monkeypatch):
    """reference_anchor.always: a single-subject shot with a reference is drawn
    image-to-video from attempt 0 — no reject needed. The (accepted) first take
    is the i2v graph anchored on the staged reference."""
    from pipeline.kidsong.generate import generate_kidsong

    run = _make_run(tmp_path, monkeypatch, enabled=True, with_reference=True,
                    always=True, accept_first=True)
    generate_kidsong(cfg=run["cfg"], resume_base=run["base"])

    calls = run["render_calls"]
    assert len(calls) == 1, "accepted on the first take -> exactly one render"
    wf0, p0 = calls[0]
    assert wf0 == "ltx23_i2v_toon"                      # anchored from the start
    assert p0["INPUT_IMAGE"] == {"image": "staged_1.png"}
    assert "SIGMAS" in p0 and p0["SIGMAS"]["sigmas"].startswith("1.0000")
    assert run["staged"] == [refs.reference_path(run["cfg"], "kofi")]


def test_always_anchor_off_first_take_is_t2v(tmp_path, monkeypatch):
    """Default (always absent/False): the first take is plain t2v — byte-identical
    to today, the proactive path stays inert."""
    from pipeline.kidsong.generate import generate_kidsong

    run = _make_run(tmp_path, monkeypatch, enabled=True, with_reference=True,
                    always=False, accept_first=True)
    generate_kidsong(cfg=run["cfg"], resume_base=run["base"])

    calls = run["render_calls"]
    assert len(calls) == 1
    wf0, p0 = calls[0]
    assert wf0 == "ltx23_t2v_toon"
    assert "INPUT_IMAGE" not in p0 and "SIGMAS" not in p0
    assert run["staged"] == []


def test_always_anchor_with_no_reference_stays_t2v(tmp_path, monkeypatch):
    """always=True but no reference on disk -> nothing to anchor on, so the shot
    renders t2v exactly as before (never blocks on a missing reference)."""
    from pipeline.kidsong.generate import generate_kidsong

    run = _make_run(tmp_path, monkeypatch, enabled=True, with_reference=False,
                    always=True, accept_first=True)
    generate_kidsong(cfg=run["cfg"], resume_base=run["base"])

    calls = run["render_calls"]
    assert len(calls) == 1
    wf0, p0 = calls[0]
    assert wf0 == "ltx23_t2v_toon"
    assert "INPUT_IMAGE" not in p0
    assert run["staged"] == []


# ===================================================== i2v patch contract ======
def test_i2v_escalation_patch_set_applies_to_real_i2v_graph():
    """The escalation patch set (common titles + INPUT_IMAGE + SIGMAS) must land
    on the real ltx23_i2v_toon graph with no missing-title KeyError — the exact
    failure mode the plan warns about (workflow name and patches must switch
    together)."""
    client = _RealComfyClient({"comfy": {"autostart": False}})
    wf = client.load_workflow("ltx23_i2v_toon")
    patches = {
        "PROMPT": {"text": "p"}, "NEGATIVE": {"text": "n"},
        "SEED": {"noise_seed": 7}, "WIDTH": {"value": 896}, "HEIGHT": {"value": 512},
        "FRAMES": {"value": 49},
        "LORA_STYLE": {"lora_name": "ltx23_pixar_toon.safetensors", "strength_model": 1.0},
        "FILENAME_PREFIX": {"filename_prefix": "kidsong/x/s00"},
        "INPUT_IMAGE": {"image": "staged_1.png"},
        "SIGMAS": {"sigmas": "1.0000, 0.9937, 0.9875, 0.9812, 0.9750, 0.9094, 0.7250, 0.4219, 0.0"},
    }
    patched = _RealComfyClient._apply_patches(wf, patches)
    input_node = next(n for n in patched.values()
                      if n.get("_meta", {}).get("title") == "INPUT_IMAGE")
    assert input_node["inputs"]["image"] == "staged_1.png"
    sigmas_node = next(n for n in patched.values()
                       if n.get("_meta", {}).get("title") == "SIGMAS")
    assert sigmas_node["inputs"]["sigmas"].startswith("1.0000")


def test_i2v_escalation_patch_set_applies_to_real_i2v_hires_graph():
    """Same contract as test_i2v_escalation_patch_set_applies_to_real_i2v_graph,
    against the hires i2v graph: the runtime patch set (which only ever
    targets the title "SIGMAS", never "SIGMAS_REFINE") must not disturb the
    baked-in refine schedule -- the failure mode that would silently soften
    the 2x refine pass on every escalated shot."""
    client = _RealComfyClient({"comfy": {"autostart": False}})
    wf = client.load_workflow("ltx23_i2v_toon_hires")
    patches = {
        "PROMPT": {"text": "p"}, "NEGATIVE": {"text": "n"},
        "SEED": {"noise_seed": 7}, "WIDTH": {"value": 896}, "HEIGHT": {"value": 512},
        "FRAMES": {"value": 49},
        "LORA_STYLE": {"lora_name": "ltx23_pixar_toon.safetensors", "strength_model": 1.0},
        "FILENAME_PREFIX": {"filename_prefix": "kidsong/x/s00"},
        "INPUT_IMAGE": {"image": "staged_1.png"},
        "SIGMAS": {"sigmas": "1.0000, 0.9937, 0.9875, 0.9812, 0.9750, 0.9094, 0.7250, 0.4219, 0.0"},
    }
    patched = _RealComfyClient._apply_patches(wf, patches)
    input_node = next(n for n in patched.values()
                      if n.get("_meta", {}).get("title") == "INPUT_IMAGE")
    assert input_node["inputs"]["image"] == "staged_1.png"
    sigmas_node = next(n for n in patched.values()
                       if n.get("_meta", {}).get("title") == "SIGMAS")
    assert sigmas_node["inputs"]["sigmas"].startswith("1.0000")
    # The runtime SIGMAS patch must not clobber the refine schedule.
    refine_node = next(n for n in patched.values()
                       if n.get("_meta", {}).get("title") == "SIGMAS_REFINE")
    assert refine_node["inputs"]["sigmas"] == "0.85, 0.7250, 0.4219, 0.0"


# ============================================ bootstrap: _bootstrap_cast_references ===
# The root cause behind IMP-005/006 "the anchor never fired": the identity refs
# were never CREATED, so refs.reference_for always returned None and the anchor
# silently no-op'd. _bootstrap_cast_references makes the render self-heal that.
from pipeline.kidsong.generate import _bootstrap_cast_references  # noqa: E402


class _RecordingRunlog:
    def __init__(self):
        self.stages = []
        self.warnings = 0

    def stage(self, name, **kw):
        self.stages.append((name, kw))

    def warning(self, *a, **k):
        self.warnings += 1


def _fake_bible():
    return {"characters": [{"id": "zuri"}, {"id": "kofi"}, {"id": "nala"}]}


_KF_ON = {"kidsong": {"keyframe_first": {"enabled": True, "identity_refs": True}}}


def test_bootstrap_ensures_a_reference_for_every_cast_member(monkeypatch):
    from pipeline.kidsong import cast, refs

    monkeypatch.setattr(cast, "load_bible", _fake_bible)
    called = []
    monkeypatch.setattr(refs, "ensure_reference",
                        lambda client, cfg, cid: called.append(cid) or f"/refs/{cid}.png")

    log = _RecordingRunlog()
    _bootstrap_cast_references(client="fake", cfg=_KF_ON, runlog=log)

    assert called == ["zuri", "kofi", "nala"]
    assert [s[1]["char"] for s in log.stages if s[0] == "cast_reference"] == ["zuri", "kofi", "nala"]


def test_bootstrap_is_a_noop_without_the_klein_keyframe_anchor(monkeypatch):
    from pipeline.kidsong import cast, refs

    monkeypatch.setattr(cast, "load_bible", _fake_bible)
    called = []
    monkeypatch.setattr(refs, "ensure_reference", lambda *a, **k: called.append(a))

    # keyframe_first enabled but identity_refs OFF -> no work (the anchor that
    # consumes references isn't active)
    _bootstrap_cast_references(
        client="fake",
        cfg={"kidsong": {"keyframe_first": {"enabled": True, "identity_refs": False}}},
        runlog=_RecordingRunlog(),
    )
    # …and an empty cfg (keyframe_first defaults OFF) also does nothing, so the
    # reference render never pollutes a plain render-loop path.
    _bootstrap_cast_references(client="fake", cfg={"kidsong": {}}, runlog=_RecordingRunlog())
    assert called == []


def test_bootstrap_never_raises_when_ensure_reference_throws(monkeypatch):
    from pipeline.kidsong import cast, refs

    monkeypatch.setattr(cast, "load_bible", _fake_bible)

    def boom(*a, **k):
        raise RuntimeError("comfy down")

    monkeypatch.setattr(refs, "ensure_reference", boom)
    log = _RecordingRunlog()
    _bootstrap_cast_references(client=None, cfg=_KF_ON, runlog=log)  # must not raise
    assert log.warnings == 1
