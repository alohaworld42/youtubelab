from studio import svm


def _cfg(enabled=True):
    return {"svm": {"enabled": enabled, "host": "http://127.0.0.1:3123"}}


def test_is_available_false_when_disabled():
    assert svm.is_available(_cfg(enabled=False)) is False


def test_script_to_scenes_maps_lines():
    script = {
        "tags": ["space", "facts"],
        "lines": [
            {"speaker": "narrator", "text": "The sun is a giant star"},
            {"speaker": "narrator", "text": "It burns hydrogen"},
        ],
    }
    scenes = svm.script_to_scenes(script)
    assert len(scenes) == 2
    assert scenes[0]["text"] == "The sun is a giant star"
    assert all(isinstance(s["searchTerms"], list) and s["searchTerms"] for s in scenes)
    assert "the" not in scenes[0]["searchTerms"]  # stopword removed


def test_script_to_scenes_uses_tags_when_line_too_short():
    script = {"tags": ["ocean", "blue"], "lines": [{"speaker": "narrator", "text": "Wow!"}]}
    scenes = svm.script_to_scenes(script)
    assert scenes[0]["searchTerms"]  # falls back to tags/defaults


def test_script_to_scenes_empty_falls_back_to_title():
    scenes = svm.script_to_scenes({"title": "Cool Title", "lines": []})
    assert len(scenes) == 1
    assert scenes[0]["text"] == "Cool Title"


def test_build_config_kids_vs_default():
    kids = svm.build_config({"svm": {}}, style="kids")
    base = svm.build_config({"svm": {}}, style=None)
    assert kids["voice"] == svm.KIDS_VOICE
    assert base["voice"] == svm.DEFAULT_VOICE
    assert kids["orientation"] == "portrait"
    assert kids["musicVolume"] == "high"


def test_build_config_respects_explicit_overrides():
    cfg = {"svm": {"voice": "bm_lewis", "music": "lofi", "orientation": "landscape",
                   "music_volume": "muted"}}
    c = svm.build_config(cfg, style="kids")
    assert c["voice"] == "bm_lewis"
    assert c["music"] == "lofi"
    assert c["orientation"] == "landscape"
    assert c["musicVolume"] == "muted"


def test_render_via_svm_raises_when_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(svm, "is_available", lambda cfg: False)
    import pytest

    with pytest.raises(RuntimeError):
        svm.render_via_svm(_cfg(), {"lines": []}, str(tmp_path / "o.mp4"))
