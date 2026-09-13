"""Tests for the scene_clips selection wiring in pipeline.generate.generate()
(the non-kidsong 2D dispatcher).

Contract under test:
  * `video.gpu_scene_mode` (default False/absent) gates a NEW attempt to
    render scenes via studio.gpu_scenes.build_scene_clips_gpu BEFORE the
    legacy studio.scenes.generate_scene_clips path.
  * With the flag off (or entirely absent from config — the real-world
    default), behavior must be BYTE-IDENTICAL to the pre-existing code: the
    GPU path is never even imported/called, and the legacy `video.scene_mode`
    path runs exactly as before.
  * When the flag is on and the GPU path returns clips, those clips are what
    reaches build_video, and the legacy path is never called.
  * When the flag is on but the GPU path returns None (disabled/unavailable/
    failed), this falls through to the legacy `scene_mode` path exactly as if
    gpu_scene_mode had never been set.

Every other pipeline stage (script, tts, whisper captions, assemble, thumbnail)
is stubbed so this test is CPU-only and fast — the only thing under test is
which scene-clip source pipeline.generate hands to build_video.
"""
import pytest


def _script():
    return {
        "title": "Five Weird Facts About Octopuses",
        "description": "d",
        "tags": ["t"],
        "characters": "",
        "lines": [{"speaker": "narrator", "text": "hello world", "visual": "an octopus waves"}],
    }


def _segments():
    return [{"speaker": "narrator", "text": "hello world", "start": 0.0, "end": 3.0}]


@pytest.fixture
def stubbed_generate(tmp_path, monkeypatch):
    """Stubs every pipeline stage generate() calls except the scene-clip
    selection under test. Returns a dict exposing `run(cfg)` and a `calls`
    dict recording what build_video actually received as scene_clips."""
    import pipeline.assemble as assemble_mod
    import pipeline.captions as captions_mod
    import pipeline.script_gen as script_gen_mod
    import pipeline.thumbnail as thumbnail_mod
    import pipeline.tts as tts_mod

    calls = {"scene_clips": "UNSET", "build_video_called": False}

    monkeypatch.setattr(script_gen_mod, "generate_script", lambda *a, **k: _script())
    monkeypatch.setattr(
        tts_mod, "synthesize",
        lambda lines, cfg, path: {"duration": 3.0, "segments": _segments()},
    )
    monkeypatch.setattr(captions_mod, "get_word_timestamps", lambda *a, **k: [])
    monkeypatch.setattr(thumbnail_mod, "generate_thumbnail", lambda *a, **k: "thumb.jpg")

    def fake_build_video(cfg, voice_path, segments, words, duration, video_path,
                         on_progress=None, background_path=None, music_path=None,
                         scene_clips=None, graphics=None):
        calls["scene_clips"] = scene_clips
        calls["build_video_called"] = True
        with open(video_path, "wb") as f:
            f.write(b"\0" * 16)

    monkeypatch.setattr(assemble_mod, "build_video", fake_build_video)

    out_dir = tmp_path / "output"
    out_dir.mkdir()

    def make_cfg(video_overrides=None):
        return {
            "_root": str(tmp_path),
            "paths": {"output_dir": str(out_dir)},
            "video": {"max_seconds": 170, "width": 1080, "height": 1920, "fps": 30,
                      **(video_overrides or {})},
            "captions": {"uppercase": True},
            "hook": {"enabled": False},
            "motion": {"enabled": False},
        }

    def run(cfg):
        from pipeline.generate import generate

        return generate("facts", topic="octopuses", cfg=cfg)

    return {"calls": calls, "make_cfg": make_cfg, "run": run}


# =============================================== both flags off/absent (default) ===
def test_both_flags_absent_is_byte_identical_to_no_scene_mode(stubbed_generate, monkeypatch):
    """The real-world default: neither key is even in config.json. Neither
    scene path may be imported/called, and build_video must receive
    scene_clips=None — exactly today's behavior with no scene support at all."""
    import studio.gpu_scenes as gpu_scenes_mod
    import studio.scenes as scenes_mod

    monkeypatch.setattr(
        gpu_scenes_mod, "build_scene_clips_gpu",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("GPU scene path must not run")),
    )
    monkeypatch.setattr(
        scenes_mod, "generate_scene_clips",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("legacy scene path must not run")),
    )

    cfg = stubbed_generate["make_cfg"]()  # no gpu_scene_mode, no scene_mode key at all
    stubbed_generate["run"](cfg)

    assert stubbed_generate["calls"]["build_video_called"] is True
    assert stubbed_generate["calls"]["scene_clips"] is None


def test_gpu_scene_mode_explicitly_false_matches_absent(stubbed_generate, monkeypatch):
    import studio.gpu_scenes as gpu_scenes_mod

    monkeypatch.setattr(
        gpu_scenes_mod, "build_scene_clips_gpu",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("GPU scene path must not run")),
    )

    cfg = stubbed_generate["make_cfg"]({"gpu_scene_mode": False})
    stubbed_generate["run"](cfg)

    assert stubbed_generate["calls"]["scene_clips"] is None


# ============================================ legacy scene_mode still works ===
def test_legacy_scene_mode_unaffected_when_gpu_flag_is_off(stubbed_generate, monkeypatch):
    import studio.scenes as scenes_mod

    legacy_clips = [{"path": "legacy.mp4", "start": 0.0, "end": 3.0, "duration": 3.0}]
    monkeypatch.setattr(scenes_mod, "generate_scene_clips", lambda *a, **k: legacy_clips)

    cfg = stubbed_generate["make_cfg"]({"scene_mode": True})
    stubbed_generate["run"](cfg)

    assert stubbed_generate["calls"]["scene_clips"] == legacy_clips


# ==================================================== gpu_scene_mode = True ===
def test_gpu_scene_mode_true_uses_gpu_clips_and_skips_legacy(stubbed_generate, monkeypatch):
    import studio.gpu_scenes as gpu_scenes_mod
    import studio.scenes as scenes_mod

    gpu_clips = [{"path": "gpu_scene_00.mp4", "start": 0.0, "end": 3.0, "duration": 3.0}]
    monkeypatch.setattr(gpu_scenes_mod, "build_scene_clips_gpu", lambda *a, **k: gpu_clips)
    monkeypatch.setattr(
        scenes_mod, "generate_scene_clips",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("legacy path must not run")),
    )

    cfg = stubbed_generate["make_cfg"]({"gpu_scene_mode": True, "scene_mode": True})
    stubbed_generate["run"](cfg)

    assert stubbed_generate["calls"]["scene_clips"] == gpu_clips


def test_gpu_scene_mode_true_falls_back_to_legacy_when_gpu_returns_none(stubbed_generate, monkeypatch):
    import studio.gpu_scenes as gpu_scenes_mod
    import studio.scenes as scenes_mod

    monkeypatch.setattr(gpu_scenes_mod, "build_scene_clips_gpu", lambda *a, **k: None)
    legacy_clips = [{"path": "legacy.mp4", "start": 0.0, "end": 3.0, "duration": 3.0}]
    monkeypatch.setattr(scenes_mod, "generate_scene_clips", lambda *a, **k: legacy_clips)

    cfg = stubbed_generate["make_cfg"]({"gpu_scene_mode": True, "scene_mode": True})
    stubbed_generate["run"](cfg)

    assert stubbed_generate["calls"]["scene_clips"] == legacy_clips


def test_gpu_scene_mode_true_falls_back_to_background_when_both_return_nothing(
    stubbed_generate, monkeypatch,
):
    import studio.gpu_scenes as gpu_scenes_mod

    monkeypatch.setattr(gpu_scenes_mod, "build_scene_clips_gpu", lambda *a, **k: None)

    cfg = stubbed_generate["make_cfg"]({"gpu_scene_mode": True})  # no legacy scene_mode
    stubbed_generate["run"](cfg)

    assert stubbed_generate["calls"]["scene_clips"] is None


def test_gpu_scene_mode_true_and_gpu_path_raises_falls_back_to_legacy(stubbed_generate, monkeypatch):
    """generate.py must wrap the GPU call in the same kind of try/except the
    legacy scene block already uses — a raised exception must not propagate."""
    import studio.gpu_scenes as gpu_scenes_mod
    import studio.scenes as scenes_mod

    def boom(*a, **k):
        raise RuntimeError("ComfyUI exploded")

    monkeypatch.setattr(gpu_scenes_mod, "build_scene_clips_gpu", boom)
    legacy_clips = [{"path": "legacy.mp4", "start": 0.0, "end": 3.0, "duration": 3.0}]
    monkeypatch.setattr(scenes_mod, "generate_scene_clips", lambda *a, **k: legacy_clips)

    cfg = stubbed_generate["make_cfg"]({"gpu_scene_mode": True, "scene_mode": True})
    stubbed_generate["run"](cfg)  # must not raise

    assert stubbed_generate["calls"]["scene_clips"] == legacy_clips
