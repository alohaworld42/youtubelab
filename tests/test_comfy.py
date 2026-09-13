import json

from studio import assets, comfy, models


def _cfg(tmp_path, enabled=True):
    return {
        "_root": str(tmp_path),
        "comfyui": {
            "enabled": enabled,
            "host": "http://127.0.0.1:8188",
            "workflow_path": "wf.json",
            "prompt_node_title": "positive",
            "prompt_template": "3d cartoon animation, {topic}, for children",
            "timeout_seconds": 5,
        },
    }


def test_is_enabled_and_disabled(tmp_path):
    assert comfy.is_enabled(_cfg(tmp_path, enabled=True))
    assert not comfy.is_enabled(_cfg(tmp_path, enabled=False))


def test_is_available_false_when_disabled(tmp_path):
    # disabled -> never even hits the network
    assert comfy.is_available(_cfg(tmp_path, enabled=False)) is False


def test_build_prompt_fills_topic(tmp_path):
    p = comfy.build_prompt(_cfg(tmp_path), "counting sheep", None)
    assert "counting sheep" in p and "cartoon" in p


def test_build_prompt_falls_back_to_title(tmp_path):
    p = comfy.build_prompt(_cfg(tmp_path), "", {"title": "Happy Colors"})
    assert "Happy Colors" in p


def test_find_prompt_node_by_title():
    wf = {"6": {"class_type": "CLIPTextEncode", "_meta": {"title": "positive"},
                "inputs": {"text": "x"}}}
    assert comfy._find_prompt_node(wf, "positive") == "6"


def test_find_prompt_node_by_id():
    wf = {"9": {"class_type": "CLIPTextEncode", "inputs": {"text": "x"}}}
    assert comfy._find_prompt_node(wf, "9") == "9"


def test_find_prompt_node_first_clip_fallback():
    wf = {"2": {"class_type": "KSampler", "inputs": {}},
          "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "x"}}}
    assert comfy._find_prompt_node(wf, "nope") == "3"


def test_first_output_prefers_video():
    outputs = {
        "10": {"images": [{"filename": "a.png"}]},
        "11": {"gifs": [{"filename": "b.mp4"}]},
    }
    assert comfy._first_output_file(outputs)["filename"] == "b.mp4"


def test_generate_clip_returns_none_when_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(comfy, "is_available", lambda cfg: False)
    assert comfy.generate_clip(_cfg(tmp_path), "topic", None, str(tmp_path / "o.mp4")) is None


def test_aivideo_tier_skipped_when_unavailable(tmp_db, tmp_path, monkeypatch):
    monkeypatch.setattr(comfy, "is_available", lambda cfg: False)
    monkeypatch.setattr(assets, "_from_stock", lambda kw, cfg: ("stock.mp4", 7))
    cfg = {"_root": str(tmp_path), "paths": {"gameplay_dir": "gp"},
           "assets": {"source_order": ["aivideo", "stock", "generated"]},
           "comfyui": {"enabled": False}}
    path, _ = assets.get_background(None, "colors", None, cfg)
    assert path == "stock.mp4"  # fell through past aivideo


def test_generate_clip_injects_dims_and_fps(tmp_path, monkeypatch):
    wf_path = tmp_path / "wf.json"
    wf_path.write_text(json.dumps({
        "70": {"class_type": "EmptyLTXVLatentVideo", "_meta": {"title": "latent"},
               "inputs": {"width": 1, "height": 1, "length": 1, "batch_size": 1}},
        "69": {"class_type": "LTXVConditioning", "_meta": {"title": "cond"},
               "inputs": {"frame_rate": 1.0}},
        "6": {"class_type": "CLIPTextEncode", "_meta": {"title": "positive"},
              "inputs": {"text": "x"}},
    }))
    cfg = _cfg(tmp_path)
    cfg["comfyui"]["workflow_path"] = str(wf_path)
    cfg["comfyui"]["width"] = 640
    cfg["comfyui"]["height"] = 1088
    cfg["comfyui"]["frames"] = 97
    cfg["comfyui"]["frame_rate"] = 24

    monkeypatch.setattr(comfy, "is_available", lambda c: True)
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured["prompt"] = json["prompt"]
        class R:
            def raise_for_status(self): pass
            def json(self): return {"prompt_id": "p1"}
        return R()

    monkeypatch.setattr(comfy.requests, "post", fake_post)
    monkeypatch.setattr(comfy.requests, "get", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stop after submit")))

    try:
        comfy.generate_clip(cfg, "topic", None, str(tmp_path / "out.mp4"))
    except RuntimeError:
        pass

    latent = captured["prompt"]["70"]["inputs"]
    assert latent["width"] == 640 and latent["height"] == 1088 and latent["length"] == 97
    assert captured["prompt"]["69"]["inputs"]["frame_rate"] == 24


def test_aivideo_tier_used_when_available(tmp_db, tmp_path, monkeypatch):
    clip = tmp_path / "gen.mp4"

    def fake_generate(cfg, topic, script, out, seed=None):
        with open(out, "wb") as f:
            f.write(b"video")
        return out

    monkeypatch.setattr(comfy, "is_available", lambda cfg: True)
    monkeypatch.setattr(comfy, "generate_clip", fake_generate)
    cfg = {"_root": str(tmp_path), "paths": {"gameplay_dir": "gp"},
           "assets": {"source_order": ["aivideo", "stock", "generated"]},
           "comfyui": {"enabled": True, "prompt_template": "{topic}"}}
    path, asset_id = assets.get_background(None, "rainbows", None, cfg)
    assert path.endswith(".mp4")
    assert asset_id is not None
    rows = models.list_assets(kind="aivideo")
    assert len(rows) == 1
