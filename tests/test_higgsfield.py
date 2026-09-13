from studio import higgsfield, scenes


def _cfg(enabled=True, **extra):
    hf = {
        "enabled": enabled,
        "model": "bytedance/seedance/v1/lite/text-to-video",
        "durations": [5, 10],
        "poll_seconds": 0,
        "timeout_seconds": 5,
    }
    hf.update(extra)
    return {"higgsfield": hf, "video": {}}


def _set_keys(monkeypatch, key="k", secret="s"):
    monkeypatch.setenv("HIGGSFIELD_API_KEY", key)
    monkeypatch.setenv("HIGGSFIELD_API_SECRET", secret)


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_unavailable_when_disabled(monkeypatch):
    _set_keys(monkeypatch)
    assert higgsfield.is_available(_cfg(enabled=False)) is False


def test_unavailable_without_keys(monkeypatch):
    monkeypatch.delenv("HIGGSFIELD_API_KEY", raising=False)
    monkeypatch.delenv("HIGGSFIELD_API_SECRET", raising=False)
    assert higgsfield.is_available(_cfg()) is False


def test_available_with_keys(monkeypatch):
    _set_keys(monkeypatch)
    assert higgsfield.is_available(_cfg()) is True


def test_pick_duration_covers_scene():
    assert higgsfield.pick_duration(3.2, [5, 10]) == 5
    assert higgsfield.pick_duration(5.0, [5, 10]) == 5
    assert higgsfield.pick_duration(7.4, [5, 10]) == 10
    assert higgsfield.pick_duration(14.0, [5, 10]) == 10  # assemble loops/slows


def test_generate_clip_happy_path(tmp_path, monkeypatch):
    _set_keys(monkeypatch)
    submitted = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        submitted["url"] = url
        submitted["payload"] = json
        submitted["auth"] = headers["Authorization"]
        return _Resp({"status": "queued", "request_id": "r1",
                      "status_url": "https://platform.higgsfield.ai/requests/r1/status"})

    polls = iter([
        _Resp({"status": "in_progress"}),
        _Resp({"status": "completed", "video": {"url": "https://cdn/x.mp4"}}),
    ])

    class _Download:
        def __init__(self):
            self.status_code = 200

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size=None):
            yield b"video-bytes"

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_get(url, headers=None, stream=False, timeout=None):
        if stream:
            return _Download()
        return next(polls)

    monkeypatch.setattr(higgsfield.requests, "post", fake_post)
    monkeypatch.setattr(higgsfield.requests, "get", fake_get)

    out = str(tmp_path / "clip.mp4")
    assert higgsfield.generate_clip(_cfg(), "a happy chick dances", out, duration=3.0) == out
    with open(out, "rb") as f:
        assert f.read() == b"video-bytes"
    assert submitted["url"].endswith("/bytedance/seedance/v1/lite/text-to-video")
    assert submitted["auth"] == "Key k:s"
    assert submitted["payload"]["prompt"] == "a happy chick dances"
    assert submitted["payload"]["aspect_ratio"] == "9:16"
    assert submitted["payload"]["duration"] == 5


def test_generate_clip_none_on_failed_job(tmp_path, monkeypatch):
    _set_keys(monkeypatch)
    monkeypatch.setattr(higgsfield.requests, "post",
                        lambda *a, **k: _Resp({"status": "queued", "request_id": "r1"}))
    monkeypatch.setattr(higgsfield.requests, "get",
                        lambda *a, **k: _Resp({"status": "nsfw"}))
    assert higgsfield.generate_clip(_cfg(), "p", str(tmp_path / "o.mp4")) is None


def test_generate_clip_none_on_submit_error(tmp_path, monkeypatch):
    _set_keys(monkeypatch)

    def boom(*a, **k):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(higgsfield.requests, "post", boom)
    assert higgsfield.generate_clip(_cfg(), "p", str(tmp_path / "o.mp4")) is None


def test_generate_clip_none_on_timeout(tmp_path, monkeypatch):
    _set_keys(monkeypatch)
    monkeypatch.setattr(higgsfield.requests, "post",
                        lambda *a, **k: _Resp({"status": "queued", "request_id": "r1"}))
    monkeypatch.setattr(higgsfield.requests, "get",
                        lambda *a, **k: _Resp({"status": "in_progress"}))
    cfg = _cfg(timeout_seconds=0)
    assert higgsfield.generate_clip(cfg, "p", str(tmp_path / "o.mp4")) is None


def test_params_passthrough(tmp_path, monkeypatch):
    _set_keys(monkeypatch)
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured.update(json)
        raise RuntimeError("stop after submit")

    monkeypatch.setattr(higgsfield.requests, "post", fake_post)
    higgsfield.generate_clip(_cfg(params={"quality": "high"}), "p", str(tmp_path / "o.mp4"))
    assert captured["quality"] == "high"


def test_scene_provider_prefers_comfy(monkeypatch):
    from studio import comfy

    monkeypatch.setattr(comfy, "is_available", lambda cfg: True)
    monkeypatch.setattr(higgsfield, "is_available", lambda cfg: True)
    assert scenes.pick_provider({"video": {}}) == "comfyui"


def test_scene_provider_falls_back_to_higgsfield(monkeypatch):
    from studio import comfy

    monkeypatch.setattr(comfy, "is_available", lambda cfg: False)
    monkeypatch.setattr(higgsfield, "is_available", lambda cfg: True)
    assert scenes.pick_provider({"video": {}}) == "higgsfield"


def test_scene_provider_pinned(monkeypatch):
    from studio import comfy

    monkeypatch.setattr(comfy, "is_available", lambda cfg: True)
    monkeypatch.setattr(higgsfield, "is_available", lambda cfg: True)
    assert scenes.pick_provider({"video": {"scene_provider": "higgsfield"}}) == "higgsfield"


def test_scene_provider_none(monkeypatch):
    from studio import comfy

    monkeypatch.setattr(comfy, "is_available", lambda cfg: False)
    monkeypatch.setattr(higgsfield, "is_available", lambda cfg: False)
    assert scenes.pick_provider({"video": {}}) is None


def test_scene_clips_via_higgsfield(tmp_path, monkeypatch):
    from studio import comfy

    monkeypatch.setattr(comfy, "is_available", lambda cfg: False)
    monkeypatch.setattr(higgsfield, "is_available", lambda cfg: True)

    def fake_clip(cfg, prompt, out, duration=5.0, on_progress=None):
        with open(out, "wb") as f:
            f.write(b"v")
        return out

    monkeypatch.setattr(higgsfield, "generate_clip", fake_clip)

    script = {"title": "T", "characters": "a chick",
              "lines": [{"speaker": "n", "text": "hello", "visual": "chick waves"}]}
    segs = [{"speaker": "n", "text": "hello", "start": 0.0, "end": 4.0}]
    cfg = {"higgsfield": {"enabled": True, "max_scenes": 6}, "video": {}}
    clips = scenes.generate_scene_clips(script, segs, cfg, str(tmp_path / "sc"))
    assert clips and len(clips) == 1
    assert clips[0]["path"].endswith("scene_00.mp4")
