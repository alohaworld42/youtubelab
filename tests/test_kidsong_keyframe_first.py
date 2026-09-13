"""Tests for keyframe-first rendering in pipeline.kidsong.generate:
pre-rendering each pending shot's FIRST FRAME with Z-Image Turbo (a still
model) before LTX ever touches it, so the exact child count is baked into
pixel-space instead of guessed at from text by the video model's t2v graph --
the fix for clone-collapse (the same child rendered 2-3x in a shot).

Same integration-harness shape as tests/test_kidsong_reference_anchor.py: a
resumed director run with pending shots and a fake ComfyClient that captures
every render call. Two DIFFERENCES from that harness:

  * TWO pending shots, not one -- s00 (Kofi closeup) and s01 (["all"] wide) --
    because keyframe-first, unlike reference-anchor, is meant to cover
    wide/group shots too (that is precisely where clone-collapse is worst).
  * the fake reviewer accepts every FIRST take, so each shot renders exactly
    once and the call order (Z-Image stills batch, then video renders) is
    unambiguous.

The prompt-composition units (`_identity_block`, `_keyframe_prompt`) are
covered indirectly here via the captured PROMPT patches; the identity/head-
count sentence contract itself is already pinned by
tests/test_kidsong_shot_prompt.py, which this feature must not disturb -- see
the byte-identical guard in `test_disabled_is_byte_identical_t2v` below.

CPU-only, no GPU, no network, no ffmpeg, no real ComfyUI. `load_workflow`
reads the real repo workflow JSON off local disk (same trick
tests/test_kidsong_reference_anchor.py uses).
"""
import json
import os
import struct
import wave

import pytest

from pipeline.kidsong import runstate
from pipeline.kidsong.comfy import ComfyClient as _RealComfyClient
from pipeline.kidsong.generate import _keyframe_path, _shot_dims

_REAL_LOAD_WORKFLOW = _RealComfyClient.load_workflow
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _write_png(path, w, h):
    """A real, valid PNG at exactly (w, h). The code under test opens
    keyframe files with PIL (a dims check, and on mismatch an ImageOps.fit
    cover-fit) -- a placeholder blob like b"PNGDATA" (fine for reference-
    image fixtures that are never opened, see test_kidsong_reference_anchor)
    would fail here with PIL.UnidentifiedImageError."""
    from PIL import Image

    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("RGB", (w, h), color=(180, 90, 40)).save(path)


def _write_wav(path, seconds=4.0, rate=8000):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(struct.pack("<h", 0) * int(rate * seconds))


def _make_run(tmp_path, monkeypatch, *, keyframe_first_enabled,
              preexisting_wrong_size_keyframe_for_s00=False,
              fail_zimage_for=None, fail_first_video_render=False):
    """A resumed director run with TWO pending shots: s00 (Kofi closeup) and
    s01 (["all"] wide). The reviewer accepts every FIRST take, so each shot
    renders exactly once and the call order is unambiguous. Returns a dict
    with the cfg, base, the real (width, height) `_shot_dims` resolves to,
    and the captured render_calls/staged/free_calls lists.

    `preexisting_wrong_size_keyframe_for_s00`: writes a keyframe PNG at the
    WRONG size to s00's keyframe path before the run starts, simulating a
    keyframe left over from a prior run whose kidsong.shot dimensions have
    since changed. Exercises both "reuse, don't regenerate" (the batch
    phase) and the cover-fit path (the per-shot staging block) at once.

    `fail_zimage_for`: a shot id (e.g. "s00") whose Z-Image still render
    raises instead of succeeding.

    `fail_first_video_render`: the FIRST video (non-zimage) render raises a
    RuntimeError once — the infra-retry then frees VRAM (which in production
    deletes every staged input) and must RE-STAGE the keyframe before the
    retry. Regression harness for the "Invalid image file" 400 measured live
    on job 22.
    """
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    base = "20260722-090000-kidsong-keyframe-first-test"
    shots_dir = runstate.shots_dir_path(str(out_dir), base)
    os.makedirs(shots_dir, exist_ok=True)

    _write_wav(out_dir / f"{base}.wav")
    song = {
        "title": "Keyframe First Test", "description": "d", "tags": ["t"],
        "characters": "Three toddlers: Zuri; Kofi; Nala",
        "verses": [
            {"scene": "bathroom", "lines": ["a", "b"]},
            {"scene": "backyard", "lines": ["c", "d"]},
        ],
    }
    (out_dir / f"{base}-song.json").write_text(json.dumps(song), encoding="utf-8")

    shots = [
        {
            "id": "s00", "verse": 0, "start": 0.0, "end": 2.0,
            "shot_type": "closeup", "characters": ["Kofi"],
            "action": "Kofi brushes his teeth with a big smile",
            "setting": "a bright bathroom", "camera": "static",
            "reuse_of": None, "seed": 1000, "status": "planned",
        },
        {
            "id": "s01", "verse": 1, "start": 2.0, "end": 4.0,
            "shot_type": "wide", "characters": ["all"],
            "action": "The kids clap their hands together",
            "setting": "a sunny backyard", "camera": "static",
            "reuse_of": None, "seed": 2000, "status": "planned",
        },
    ]
    runstate.save_shotlist(str(out_dir), base, {"shots": shots})

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
            "reference_anchor": {"enabled": False},
            "keyframe_first": {"enabled": keyframe_first_enabled},
        },
    }
    real_w, real_h = _shot_dims(cfg)

    if preexisting_wrong_size_keyframe_for_s00:
        wrong_path = _keyframe_path(shots_dir, shots[0], real_w, real_h)
        _write_png(wrong_path, max(16, real_w // 2), max(16, real_h // 2))

    render_calls = []   # (workflow, patches) -- Z-Image stills AND videos
    staged = []         # src paths passed to stage_input_image
    free_calls = []     # snapshot of len(render_calls) at each free() call
    video_failed_once = []  # mutable flag for fail_first_video_render

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
            if workflow == "zimage_ref":
                shot_id = os.path.basename(out_path).split("_")[0]
                if fail_zimage_for and shot_id == fail_zimage_for:
                    raise RuntimeError("Z-Image OOM (simulated)")
                render_calls.append((workflow, patches))
                _write_png(out_path, patches["WIDTH"]["value"], patches["HEIGHT"]["value"])
                return out_path
            if fail_first_video_render and not video_failed_once:
                video_failed_once.append(True)
                raise RuntimeError("ComfyUI hiccup (simulated infra failure)")
            render_calls.append((workflow, patches))
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "wb") as f:
                f.write(b"\0" * 1024)
            return out_path

        def free(self):
            free_calls.append(len(render_calls))

    class FakeReviewer:
        """Accepts every first take -- each shot renders exactly once, so the
        Z-Image-batch-then-video-loop call order is unambiguous."""

        def review(self, shot, video_path, out_dir=None):
            return {"accept": True, "score": 1.0, "reasons": [], "retry_hints": {}}

    def fake_assemble(cfg, cut_list, renders, voice, words, duration, out_path, **kw):
        with open(out_path, "wb") as f:
            f.write(b"\0" * 4096)

    monkeypatch.setattr(comfy_mod, "ComfyClient", FakeComfyClient)
    monkeypatch.setattr(review_mod, "get_reviewer", lambda cfg: FakeReviewer())
    monkeypatch.setattr(review_mod, "contact_sheet", lambda *a, **k: None)
    monkeypatch.setattr(review_mod, "review_shots_log", lambda path, entries: path)
    monkeypatch.setattr(captions_mod, "get_word_timestamps", lambda *a, **k: [])
    monkeypatch.setattr(sing_mod, "verse_times_from_words",
                        lambda *a, **k: [(0.0, 2.0), (2.0, 4.0)])
    monkeypatch.setattr(edit_mod, "beat_grid", lambda path: [0.0, 1.0, 2.0, 3.0, 4.0])
    monkeypatch.setattr(edit_mod, "build_cut_list", lambda *a, **k: [])
    monkeypatch.setattr(edit_mod, "assemble", fake_assemble)
    monkeypatch.setattr(cut_qc_mod, "review_cut",
                        lambda *a, **k: {"accept": True, "score": 1.0, "reasons": [],
                                          "retry_hints": {}})
    monkeypatch.setattr(edit_mod, "prepend_intro",
                        lambda *a, **k: {"applied": False, "intro_seconds": 0.0})
    monkeypatch.setattr(sing_mod, "sing_song",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no re-sing")))

    return {
        "cfg": cfg, "base": base, "shots_dir": shots_dir,
        "real_dims": (real_w, real_h),
        "render_calls": render_calls, "staged": staged, "free_calls": free_calls,
    }


# ===================================================== (a) enabled: batch =====
def test_enabled_keyframes_both_shots_then_animates_via_i2v(tmp_path, monkeypatch):
    from pipeline.kidsong.generate import generate_kidsong

    run = _make_run(tmp_path, monkeypatch, keyframe_first_enabled=True)
    generate_kidsong(cfg=run["cfg"], resume_base=run["base"])

    calls = run["render_calls"]
    assert len(calls) == 4, "2 zimage_ref stills + 2 ltx23_i2v_toon videos"
    real_w, real_h = run["real_dims"]

    workflows_in_order = [wf for wf, _ in calls]
    assert workflows_in_order == [
        "zimage_ref", "zimage_ref", "ltx23_i2v_toon", "ltx23_i2v_toon",
    ], "all zimage_ref calls must strictly precede all video calls"

    zimage_calls = calls[:2]
    video_calls = calls[2:]

    for wf, patches in zimage_calls:
        prompt = patches["PROMPT"]["prompt"]
        assert patches["WIDTH"]["value"] == real_w
        assert patches["HEIGHT"]["value"] == real_h
        assert "P1x4r" not in prompt
        for banned in ("push-in", "pan", "pull-back"):
            assert banned not in prompt, f"{banned!r} leaked camera-move phrasing into {prompt!r}"

    # The wide ["all"] shot (s01, the second pending shot) IS keyframed --
    # unlike reference-anchor, which only ever anchors single-subject shots.
    assert "Exactly three children are on screen" in zimage_calls[1][1]["PROMPT"]["prompt"]
    # And s00's closeup still states its own single-child identity sentence.
    assert "exactly one child" in zimage_calls[0][1]["PROMPT"]["prompt"].lower()

    for wf, patches in video_calls:
        assert wf == "ltx23_i2v_toon"
        assert "SIGMAS" in patches and patches["SIGMAS"]["sigmas"].startswith("1.0000")
    # staged in per-shot-loop order: s00 first, s01 second.
    assert video_calls[0][1]["INPUT_IMAGE"] == {"image": "staged_1.png"}
    assert video_calls[1][1]["INPUT_IMAGE"] == {"image": "staged_2.png"}
    assert run["staged"] == [
        _keyframe_path(run["shots_dir"], {"id": "s00"}, real_w, real_h),
        _keyframe_path(run["shots_dir"], {"id": "s01"}, real_w, real_h),
    ]

    # Z-Image freed before any i2v render started -- 12GB VRAM can't hold
    # the still model and the 22B LTX model at once.
    assert 2 in run["free_calls"], (
        "expected a free() snapshot right after the 2 zimage_ref calls and "
        f"before any video call; got {run['free_calls']}"
    )


def test_enabled_keyframes_both_shots_then_animates_via_i2v_hires(tmp_path, monkeypatch):
    """Same as test_enabled_keyframes_both_shots_then_animates_via_i2v, but
    with kidsong.shot.hires_pass True: the keyframe-first video renders must
    follow the hires toggle exactly like the plain t2v path does, landing on
    the _hires i2v graph. The keyframe STILLS (zimage_ref) are a separate
    model/graph entirely and must be unaffected by the toggle."""
    from pipeline.kidsong.generate import generate_kidsong

    run = _make_run(tmp_path, monkeypatch, keyframe_first_enabled=True)
    run["cfg"]["kidsong"]["shot"]["hires_pass"] = True
    generate_kidsong(cfg=run["cfg"], resume_base=run["base"])

    calls = run["render_calls"]
    assert len(calls) == 4, "2 zimage_ref stills + 2 ltx23_i2v_toon_hires videos"

    workflows_in_order = [wf for wf, _ in calls]
    assert workflows_in_order == [
        "zimage_ref", "zimage_ref", "ltx23_i2v_toon_hires", "ltx23_i2v_toon_hires",
    ], "all zimage_ref calls must strictly precede all video calls"

    video_calls = calls[2:]
    for wf, patches in video_calls:
        assert wf == "ltx23_i2v_toon_hires"
        assert "SIGMAS" in patches and patches["SIGMAS"]["sigmas"].startswith("1.0000")
    assert video_calls[0][1]["INPUT_IMAGE"] == {"image": "staged_1.png"}
    assert video_calls[1][1]["INPUT_IMAGE"] == {"image": "staged_2.png"}


# =================================================== (b) disabled: pure t2v ===
def test_disabled_is_byte_identical_t2v(tmp_path, monkeypatch):
    from pipeline.kidsong.generate import generate_kidsong

    run = _make_run(tmp_path, monkeypatch, keyframe_first_enabled=False)
    generate_kidsong(cfg=run["cfg"], resume_base=run["base"])

    calls = run["render_calls"]
    assert len(calls) == 2
    for wf, patches in calls:
        assert wf == "ltx23_t2v_toon"          # never switches graph
        assert "INPUT_IMAGE" not in patches    # never anchors
        assert "SIGMAS" not in patches
    assert run["staged"] == []                 # nothing staged -> nothing leaks
    assert not any(wf == "zimage_ref" for wf, _ in calls)


# ============================================ (c) pre-existing keyframe file ==
def test_preexisting_keyframe_is_reused_not_regenerated(tmp_path, monkeypatch):
    """s00 already has a (wrong-size) keyframe on disk from a prior run --
    the batch phase must reuse it, not re-render it -- and the per-shot
    staging block must cover-fit it before staging, since its size no
    longer matches kidsong.shot's current dims. s01 gets a fresh keyframe
    at the right size and stages as-is."""
    from pipeline.kidsong.generate import generate_kidsong

    run = _make_run(
        tmp_path, monkeypatch, keyframe_first_enabled=True,
        preexisting_wrong_size_keyframe_for_s00=True,
    )
    generate_kidsong(cfg=run["cfg"], resume_base=run["base"])

    calls = run["render_calls"]
    zimage_calls = [c for c in calls if c[0] == "zimage_ref"]
    assert len(zimage_calls) == 1, "s00 already had a keyframe -- only s01's renders"

    video_calls = [c for c in calls if c[0] != "zimage_ref"]
    assert len(video_calls) == 2
    for wf, _ in video_calls:
        assert wf == "ltx23_i2v_toon"

    # s00's pre-existing keyframe was the WRONG size -> cover-fit before
    # staging; s01's was generated fresh at the right size -> staged as-is.
    assert run["staged"][0].endswith("_fit.png")
    assert os.path.isfile(run["staged"][0])
    assert not run["staged"][1].endswith("_fit.png")


# ===================================== (d) generate_still fails for one shot ==
def test_zimage_failure_for_one_shot_falls_back_to_t2v_for_that_shot_only(
    tmp_path, monkeypatch,
):
    from pipeline.kidsong.generate import generate_kidsong

    run = _make_run(
        tmp_path, monkeypatch, keyframe_first_enabled=True, fail_zimage_for="s00",
    )
    generate_kidsong(cfg=run["cfg"], resume_base=run["base"])

    calls = run["render_calls"]
    zimage_calls = [c for c in calls if c[0] == "zimage_ref"]
    assert len(zimage_calls) == 1, "s00's still render raised -- only s01's succeeded"

    video_workflows = [wf for wf, _ in calls if wf != "zimage_ref"]
    # order preserved: s00 (no keyframe -> t2v) first, s01 (keyframed -> i2v) second.
    assert video_workflows == ["ltx23_t2v_toon", "ltx23_i2v_toon"]
    # s00 never got a staged input; s01 did.
    assert len(run["staged"]) == 1


# ============================== (e) infra-retry re-stages the staged keyframe ==
def test_infra_retry_restages_the_keyframe_after_vram_free(tmp_path, monkeypatch):
    """Measured live on job 22: the infra-retry's client.free() deletes every
    staged input, so retrying the same patches 400'd on LoadImage ("Invalid
    image file"). The retry must RE-STAGE from the kept local source and point
    INPUT_IMAGE at the fresh name."""
    import pipeline.kidsong.generate as generate_mod
    from pipeline.kidsong.generate import generate_kidsong

    run = _make_run(
        tmp_path, monkeypatch, keyframe_first_enabled=True,
        fail_first_video_render=True,
    )
    monkeypatch.setattr(generate_mod.time, "sleep", lambda *_a: None)
    generate_kidsong(cfg=run["cfg"], resume_base=run["base"])

    calls = run["render_calls"]
    video_calls = [(wf, p) for wf, p in calls if wf != "zimage_ref"]
    # both shots still rendered i2v (the failed first try is not in render_calls)
    assert [wf for wf, _ in video_calls] == ["ltx23_i2v_toon", "ltx23_i2v_toon"]
    # s00: staged once, render raised, re-staged (2), retry OK; s01: staged (3).
    assert len(run["staged"]) == 3, run["staged"]
    assert run["staged"][0] == run["staged"][1], "re-stage must reuse the same source file"
    # the successful s00 render carries the RE-staged name, not the deleted one
    assert video_calls[0][1]["INPUT_IMAGE"] == {"image": "staged_2.png"}
    assert video_calls[1][1]["INPUT_IMAGE"] == {"image": "staged_3.png"}


# ==================== (f) keyframe precedence over the reference card =========
def test_keyframe_outranks_single_subject_ref_when_always_on(tmp_path, monkeypatch):
    """INVERTED pin (the old card-over-keyframe precedence WAS the bug): the
    scene keyframe is identity-verified (klein refs + CLIP retry) and carries
    prop/setting/composition; the canonical card is a bare portrait with none
    of them. Animating solo shots from the raw card was the measured root of
    the appearance-loss / vanishing-prop / hazy-face / strobe cluster (every
    single-child take of the pixar-full episode strobe-rejected while morphing
    card -> scene). With BOTH features on, every shot with a keyframe on disk
    animates from that keyframe; the card is only the no-keyframe fallback."""
    from pipeline.kidsong import refs as refs_mod
    from pipeline.kidsong.generate import generate_kidsong

    run = _make_run(tmp_path, monkeypatch, keyframe_first_enabled=True)
    cfg = run["cfg"]
    cfg["kidsong"]["reference_anchor"] = {"enabled": True, "always": True}
    # canonical reference for Kofi (s00's single subject) — a real PNG
    ref = refs_mod.reference_path(cfg, "kofi")
    _write_png(ref, 768, 1024)

    generate_kidsong(cfg=cfg, resume_base=run["base"])

    calls = run["render_calls"]
    zimage_calls = [c for c in calls if c[0] == "zimage_ref"]
    video_calls = [(wf, p) for wf, p in calls if wf != "zimage_ref"]
    # both shots render i2v; keyframes still batch for both (cheap, cached)
    assert [wf for wf, _ in video_calls] == ["ltx23_i2v_toon", "ltx23_i2v_toon"]
    # BOTH shots (solo s00 included) animate from their KEYFRAME — the card
    # stays on disk unused while a keyframe exists.
    assert "keyframes" in run["staged"][0], run["staged"]
    assert "keyframes" in run["staged"][1]
    assert len(zimage_calls) == 2  # batch still rendered both keyframes


def test_card_fallback_is_cover_fit_when_no_keyframe(tmp_path, monkeypatch):
    """No keyframe on disk (batch failed) + reference_anchor.always: the solo
    shot falls back to the card — but COVER-FIT to the exact shot dims first
    (a raw 768x1024 portrait stretched into the 896x512 latent was the
    hazy-oversized-face source), staged under a dims-keyed name."""
    from PIL import Image

    from pipeline.kidsong import refs as refs_mod
    from pipeline.kidsong.generate import generate_kidsong

    run = _make_run(tmp_path, monkeypatch, keyframe_first_enabled=True)
    cfg = run["cfg"]
    cfg["kidsong"]["reference_anchor"] = {"enabled": True, "always": True}
    ref = refs_mod.reference_path(cfg, "kofi")
    _write_png(ref, 768, 1024)  # portrait card, wrong aspect on purpose

    # make the keyframe batch produce nothing so the fallback actually fires
    monkeypatch.setattr(refs_mod, "generate_still",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no still")))

    generate_kidsong(cfg=cfg, resume_base=run["base"])

    staged_solo = run["staged"][0]
    assert staged_solo != ref, "the RAW portrait card must never be staged"
    assert "_fit_" in os.path.basename(staged_solo)
    with Image.open(staged_solo) as im:
        w, h = im.size
    shot_cfg = cfg["kidsong"].get("shot", {})
    assert (w, h) == (int(shot_cfg.get("width", 896)), int(shot_cfg.get("height", 512)))
