"""Tests for studio.gpu_scenes — GPU-rendered scenes for the non-kids 2D
video types (facts/reddit/scary/conversation) via the mature ComfyClient +
render-style registry.

CPU-only: ComfyClient is always stubbed (monkeypatched onto
pipeline.kidsong.comfy.ComfyClient, the module gpu_scenes.py lazily imports
from — same trick tests/test_kidsong_render_style.py uses). No network, no
GPU, no real render.

The load-bearing guarantees under test:
  * the prompt NEVER carries the kids identity_clause, but DOES carry the
    scene's visual and the borrowed style's skeleton/trigger;
  * render patches carry the right titles, vertical dims snapped to a
    multiple of 32, and an 8n+1 frame count;
  * the base seed is deterministic per script title;
  * any failure (disabled, ComfyUI down, a scene render raising) returns
    None rather than raising — the caller falls back to the legacy path;
  * the returned clip dicts match the {path, start, end, duration} contract
    pipeline.assemble.build_video consumes.
"""
import os

import pytest

from pipeline.kidsong.render_style import _PIXAR_TOON_DEFAULTS
from studio.gpu_scenes import (
    DEFAULT_NEGATIVE,
    DEFAULT_PROMPT_TEMPLATE,
    _base_seed,
    _dims,
    _frames_for,
    _resolve_borrowed_style,
    _scene_prompt,
    build_scene_clips_gpu,
)


# --------------------------------------------------------------- sample data ---
def _script(n, title="T"):
    return {
        "title": title,
        "characters": "a fluffy yellow chick",
        "lines": [
            {"speaker": "narrator", "text": f"line {i}", "visual": f"chick does thing {i}"}
            for i in range(n)
        ],
    }


def _segments(durs):
    segs, t = [], 0.0
    for d in durs:
        segs.append({"speaker": "narrator", "text": "x", "start": t, "end": t + d})
        t += d
    return segs


def _registry():
    return {
        "pixar_toon": dict(_PIXAR_TOON_DEFAULTS),
        "flat_storybook": {
            "workflow_base": "ltx23_t2v_toon",
            "style_lora": "ltx23_pixar_toon.safetensors",
            "style_strength": 0.0,
            "style_trigger": "",
            "prompt_skeleton": "a soft flat 2D storybook illustration.",
            "identity_clause": _PIXAR_TOON_DEFAULTS["identity_clause"],
        },
    }


def _cfg(gpu_scenes=None, kidsong=None, video=None, studio_extra=None):
    return {
        "video": {"width": 1080, "height": 1920, "fps": 30, **(video or {})},
        "kidsong": {
            "render_style": "pixar_toon",
            "render_styles": _registry(),
            **(kidsong or {}),
        },
        "studio": {
            **(studio_extra or {}),
            "gpu_scenes": {"enabled": True, **(gpu_scenes or {})},
        },
    }


class FakeComfyClient:
    """Stub for pipeline.kidsong.comfy.ComfyClient — never touches the network."""

    instances = []

    def __init__(self, cfg):
        self.cfg = cfg
        self.freed = False
        self.render_calls = []
        FakeComfyClient.instances.append(self)

    def ensure_up(self):
        pass

    def render(self, workflow, patches, out_path):
        self.render_calls.append((workflow, patches))
        with open(out_path, "wb") as f:
            f.write(b"\0" * 8)
        return out_path

    def free(self):
        self.freed = True


class DownComfyClient(FakeComfyClient):
    def ensure_up(self):
        raise RuntimeError("ComfyUI is not responding")


class FlakyComfyClient(FakeComfyClient):
    """Renders the first scene fine, then blows up on the second."""

    def render(self, workflow, patches, out_path):
        if len(self.render_calls) >= 1:
            raise RuntimeError("node error: OOM")
        return super().render(workflow, patches, out_path)


@pytest.fixture(autouse=True)
def _reset_fake_instances():
    FakeComfyClient.instances.clear()
    yield
    FakeComfyClient.instances.clear()


# ============================================================ prompt building ===
def test_scene_prompt_excludes_kids_identity_clause_and_includes_visual_and_skeleton():
    style = dict(_PIXAR_TOON_DEFAULTS)
    scene = {"visuals": ["a cat chases a laser pointer across the kitchen"], "texts": ["x"]}
    prompt = _scene_prompt(scene, {}, style)
    assert style["identity_clause"] not in prompt
    assert "Black toddlers" not in prompt
    assert "a cat chases a laser pointer across the kitchen" in prompt
    assert style["prompt_skeleton"] in prompt
    assert style["style_trigger"] in prompt


def test_scene_prompt_falls_back_to_texts_when_no_visual():
    style = dict(_PIXAR_TOON_DEFAULTS)
    scene = {"visuals": [], "texts": ["hello world, this is the spoken line"]}
    prompt = _scene_prompt(scene, {}, style)
    assert "hello world, this is the spoken line" in prompt


def test_scene_prompt_empty_trigger_has_no_stray_double_space_or_leading_space():
    style = _registry()["flat_storybook"]
    scene = {"visuals": ["a boat sails across a calm lake"], "texts": []}
    prompt = _scene_prompt(scene, {}, style)
    assert "  " not in prompt
    assert not prompt.startswith(" ")
    assert prompt.startswith("a soft flat 2D storybook illustration")


def test_scene_prompt_honors_a_custom_template():
    style = dict(_PIXAR_TOON_DEFAULTS)
    scene = {"visuals": ["a robot waters a garden"], "texts": []}
    prompt = _scene_prompt(scene, {"prompt_template": "CUSTOM: {visual}"}, style)
    assert prompt == "CUSTOM: a robot waters a garden"


def test_default_prompt_template_has_no_children_clause():
    assert "child" not in DEFAULT_PROMPT_TEMPLATE.lower()
    assert "toddler" not in DEFAULT_PROMPT_TEMPLATE.lower()


def test_default_negative_is_nonempty():
    assert isinstance(DEFAULT_NEGATIVE, str) and DEFAULT_NEGATIVE.strip()


# ================================================================ dimensions ===
def test_dims_default_from_video_snapped_to_32():
    w, h = _dims({"video": {"width": 1080, "height": 1920}}, {})
    assert w % 32 == 0 and h % 32 == 0
    assert w <= 1080 and h <= 1920
    assert w > 1080 - 32 and h > 1920 - 32  # rounds down, not further


def test_dims_conf_override_wins_over_video_defaults():
    w, h = _dims({"video": {"width": 1080, "height": 1920}}, {"width": 640, "height": 640})
    assert (w, h) == (640, 640)


def test_dims_missing_video_config_uses_vertical_default():
    w, h = _dims({}, {})
    assert w % 32 == 0 and h % 32 == 0
    assert h > w  # still vertical/9:16-ish by default


# =================================================================== frames ===
@pytest.mark.parametrize("dur", [1.0, 2.0, 3.0, 5.0, 30.0])
def test_frames_for_is_8n_plus_1_and_capped(dur):
    f = _frames_for(dur, fps=24, max_frames=161)
    assert f % 8 == 1
    assert 49 <= f <= 161


def test_frames_for_respects_a_lower_configured_cap():
    f = _frames_for(30.0, fps=24, max_frames=97)
    assert f <= 97
    assert f % 8 == 1


# ===================================================================== seed ===
def test_base_seed_is_deterministic_per_title():
    assert _base_seed({"title": "My Video"}) == _base_seed({"title": "My Video"})


def test_base_seed_differs_across_titles():
    assert _base_seed({"title": "My Video"}) != _base_seed({"title": "A Totally Different Title"})


def test_base_seed_handles_missing_title():
    # Must not raise; falls back to a fixed "video" seed.
    assert isinstance(_base_seed({}), int)


# ============================================================= style borrow ===
def test_style_defaults_to_the_active_kidsong_render_style():
    style = _resolve_borrowed_style(_cfg(kidsong={"render_style": "pixar_toon"}), {})
    assert style["name"] == "pixar_toon"


def test_gpu_scenes_style_key_overrides_the_active_kidsong_style():
    cfg = _cfg(kidsong={"render_style": "pixar_toon"})
    style = _resolve_borrowed_style(cfg, {"style": "flat_storybook"})
    assert style["name"] == "flat_storybook"


def test_style_defaults_to_pixar_toon_with_no_kidsong_config_at_all():
    style = _resolve_borrowed_style({}, {})
    assert style["name"] == "pixar_toon"


# ========================================================= disabled / empty ===
def test_disabled_returns_none_without_touching_comfy():
    cfg = _cfg(gpu_scenes={"enabled": False})
    assert build_scene_clips_gpu(_script(2), _segments([2.0, 2.0]), cfg, "unused") is None
    assert FakeComfyClient.instances == []  # never even imported/constructed


def test_missing_studio_config_disables_by_default():
    assert build_scene_clips_gpu(_script(2), _segments([2.0, 2.0]), {}, "unused") is None


def test_no_segments_returns_none():
    cfg = _cfg()
    assert build_scene_clips_gpu(_script(0), [], cfg, "unused") is None


# ========================================================== happy path (GPU) ===
def test_build_scene_clips_gpu_happy_path(tmp_path, monkeypatch):
    import pipeline.kidsong.comfy as comfy_mod

    monkeypatch.setattr(comfy_mod, "ComfyClient", FakeComfyClient)

    script = _script(4, title="Five Weird Facts")
    segments = _segments([2.0, 2.0, 2.0, 2.0])
    cfg = _cfg()
    cache_dir = tmp_path / "scenes"

    clips = build_scene_clips_gpu(script, segments, cfg, str(cache_dir))

    assert clips
    for c in clips:
        assert set(c) == {"path", "start", "end", "duration"}
        assert os.path.isfile(c["path"])

    client = FakeComfyClient.instances[0]
    assert client.freed is True
    assert len(client.render_calls) == len(clips)

    workflow, patches = client.render_calls[0]
    assert workflow == "ltx23_t2v_toon"
    assert set(patches) >= {
        "PROMPT", "NEGATIVE", "SEED", "WIDTH", "HEIGHT", "FRAMES",
        "FILENAME_PREFIX", "LORA_STYLE",
    }
    assert patches["WIDTH"]["value"] % 32 == 0
    assert patches["HEIGHT"]["value"] % 32 == 0
    assert patches["HEIGHT"]["value"] > patches["WIDTH"]["value"]  # vertical
    assert patches["FRAMES"]["value"] % 8 == 1
    assert patches["LORA_STYLE"] == {
        "lora_name": _PIXAR_TOON_DEFAULTS["style_lora"],
        "strength_model": _PIXAR_TOON_DEFAULTS["style_strength"],
    }
    assert _PIXAR_TOON_DEFAULTS["identity_clause"] not in patches["PROMPT"]["text"]
    assert patches["NEGATIVE"]["text"] == DEFAULT_NEGATIVE

    # One deterministic seed for the whole video, across every scene.
    seeds = {p["SEED"]["noise_seed"] for _, p in client.render_calls}
    assert len(seeds) == 1
    assert seeds.pop() == _base_seed(script)


def test_build_scene_clips_gpu_uses_the_overridden_style(tmp_path, monkeypatch):
    import pipeline.kidsong.comfy as comfy_mod

    monkeypatch.setattr(comfy_mod, "ComfyClient", FakeComfyClient)

    cfg = _cfg(gpu_scenes={"style": "flat_storybook"})
    clips = build_scene_clips_gpu(_script(2), _segments([2.0, 2.0]), cfg, str(tmp_path / "s"))
    assert clips

    client = FakeComfyClient.instances[0]
    _, patches = client.render_calls[0]
    assert patches["LORA_STYLE"]["strength_model"] == 0.0
    assert not patches["PROMPT"]["text"].startswith("P1x4r")


# ========================================================== fail-soft paths ===
def test_returns_none_when_comfy_server_is_down(tmp_path, monkeypatch):
    import pipeline.kidsong.comfy as comfy_mod

    monkeypatch.setattr(comfy_mod, "ComfyClient", DownComfyClient)

    cfg = _cfg()
    clips = build_scene_clips_gpu(_script(2), _segments([2.0, 2.0]), cfg, str(tmp_path / "s"))
    assert clips is None


def test_returns_none_and_still_frees_when_a_scene_render_raises(tmp_path, monkeypatch):
    import pipeline.kidsong.comfy as comfy_mod

    monkeypatch.setattr(comfy_mod, "ComfyClient", FlakyComfyClient)

    # target_seconds defaults to 5.0 in studio.scenes.build_scenes: these
    # durations deliberately group into >= 2 scenes so the flaky client's
    # second render() call actually happens.
    cfg = _cfg()
    clips = build_scene_clips_gpu(_script(3), _segments([3.0, 3.0, 3.0]), cfg, str(tmp_path / "s"))

    assert clips is None
    client = FakeComfyClient.instances[0]
    assert client.freed is True  # finally: client.free() still ran


def test_import_failure_is_swallowed(tmp_path, monkeypatch):
    """If pipeline.kidsong.comfy can't even be imported, this must still fail
    soft rather than raise into the pipeline."""
    import builtins

    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "pipeline.kidsong.comfy":
            raise ImportError("simulated: module unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    cfg = _cfg()
    clips = build_scene_clips_gpu(_script(2), _segments([2.0, 2.0]), cfg, str(tmp_path / "s"))
    assert clips is None
