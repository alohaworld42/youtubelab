"""pipeline.kidsong.seedance -- CPU-only, zero-network tests.

Every HTTP call goes through `seedance.requests.post`/`.get`, monkeypatched
here to fake transports (same pattern as tests/test_higgsfield.py); nothing
in this file ever reaches a real socket. `seedance.budget.dry_run` defaults
true in production, but most tests here explicitly set it false so the
submit/poll/download logic itself gets exercised -- against a mocked
transport, never the network -- per the Stage 2 contract ("dry_run defaults
true and tests never touch the network").
"""
import json
import os

import pytest
import requests

from pipeline.kidsong import seedance

_FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "seedance")


def _load_fixture(name):
    with open(os.path.join(_FIXTURES, name), "r", encoding="utf-8") as f:
        return json.load(f)


def _cfg(enabled=True, dry_run=True, **overrides):
    sd = {
        "enabled": enabled,
        "base_url": "https://ark.seedance.bytedance.com",
        "allowed_hosts": ["ark.seedance.bytedance.com"],
        "model": "seedance-2.5",
        "resolution": "720p",
        "aspect_ratio": "16:9",
        "durations": [4, 6, 8, 12],
        "generate_audio": False,
        "max_reference_images": 1,
        "negative_prompt_field": "negative_prompt",
        "poll_seconds": 0,
        "timeout_seconds": 5,
        "params": {},
        "budget": {
            "dry_run": dry_run,
            "price_per_second_usd": 0.06,
            "max_usd_per_episode": 10.0,
            "max_clips_per_episode": 40,
        },
    }
    for key, val in overrides.items():
        if key == "budget" and isinstance(val, dict):
            sd["budget"].update(val)
        else:
            sd[key] = val
    return {"seedance": sd, "kidsong": {"shot": {"fps": 24}}}


def _set_key(monkeypatch, key="k"):
    monkeypatch.setenv("SEEDANCE_API_KEY", key)
    monkeypatch.delenv("SEEDANCE_BASE_URL", raising=False)


class _Resp:
    def __init__(self, payload=None, status_code=200, text=None):
        self._payload = {} if payload is None else payload
        self.status_code = status_code
        self.text = text if text is not None else json.dumps(self._payload)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error", response=self)

    def json(self):
        return self._payload


class _StreamResp:
    def __init__(self, chunks, status_code=200):
        self.status_code = status_code
        self._chunks = chunks

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error", response=self)

    def iter_content(self, chunk_size=None):
        for c in self._chunks:
            yield c


# ------------------------------------------------------------- availability --
def test_unavailable_when_disabled(monkeypatch):
    _set_key(monkeypatch)
    assert seedance.is_available(_cfg(enabled=False)) is False


def test_unavailable_without_key(monkeypatch):
    monkeypatch.delenv("SEEDANCE_API_KEY", raising=False)
    assert seedance.is_available(_cfg()) is False


def test_available_with_key(monkeypatch):
    _set_key(monkeypatch)
    assert seedance.is_available(_cfg()) is True


# ------------------------------------------------------------ pick_duration --
def test_pick_duration_smallest_that_covers_else_largest():
    choices = [4, 6, 8, 12]
    assert seedance.pick_duration(3.2, choices) == 4
    assert seedance.pick_duration(4.0, choices) == 4
    assert seedance.pick_duration(7.4, choices) == 8
    assert seedance.pick_duration(20.0, choices) == 12  # assemble loops/holds


# ---------------------------------------------------------- host allowlist --
def test_disallowed_aggregator_host_is_rejected(monkeypatch):
    _set_key(monkeypatch)
    cfg = _cfg(base_url="https://cheap-ai-video-reseller.example.com")
    with pytest.raises(seedance.SeedanceRejected):
        seedance.SeedanceBackend(cfg)


def test_allowed_host_constructs_fine(monkeypatch):
    _set_key(monkeypatch)
    backend = seedance.SeedanceBackend(_cfg())
    assert backend.url == "https://ark.seedance.bytedance.com"


# ------------------------------------------------------------- translate ----
def test_translate_strips_style_trigger(monkeypatch):
    _set_key(monkeypatch)
    backend = seedance.SeedanceBackend(_cfg())
    payload, _ = backend._translate({
        "PROMPT": {"text": "P1x4r, a happy chick dances in a sunny yard"},
    })
    # "P1x4r" is pixar_toon's default style_trigger (render_style.py); resolve_style
    # falls back to it even with no kidsong.render_styles configured.
    assert payload["prompt"] == "a happy chick dances in a sunny yard"
    assert "P1x4r" not in payload["prompt"]


def test_translate_drops_lora_style_and_sigmas(monkeypatch):
    _set_key(monkeypatch)
    monkeypatch.setattr(seedance, "_WARNED_ONCE", set())
    warnings = []
    monkeypatch.setattr(seedance.log, "warning", lambda msg, *a: warnings.append(msg % a if a else msg))

    backend = seedance.SeedanceBackend(_cfg())
    patches = {
        "PROMPT": {"text": "hello"},
        "LORA_STYLE": {"lora_name": "x.safetensors", "strength_model": 1.0},
        "SIGMAS": {"sigmas": "1.0, 0.9, 0.0"},
        "FILENAME_PREFIX": {"filename_prefix": "kidsong/ep/s01"},
    }
    payload, _ = backend._translate(patches)

    dumped = json.dumps(payload)
    assert "lora" not in dumped.lower()
    assert "sigma" not in dumped.lower()
    assert "filename_prefix" not in dumped.lower()

    # LORA_STYLE is logged once per process, not once per translate() call.
    backend._translate(patches)
    lora_warnings = [w for w in warnings if "LORA_STYLE" in w]
    assert len(lora_warnings) == 1


def test_frames_snap_up_to_duration_bucket(monkeypatch):
    _set_key(monkeypatch)
    backend = seedance.SeedanceBackend(_cfg())
    # kidsong.shot.max_frames=81 @ kidsong.shot.fps=24 -> 3.375s, which the
    # default durations [4, 6, 8, 12] snap UP to 4 (the smallest bucket that
    # covers it) -- assert the exact chosen value, not just "some" bucket.
    payload, duration = backend._translate({
        "PROMPT": {"text": "hi"},
        "FRAMES": {"value": 81},
    })
    assert duration == 4
    assert payload["duration"] == 4


def test_never_requests_4k(monkeypatch):
    _set_key(monkeypatch)
    for configured in ("4k", "1080p", "2160p", "bogus-resolution"):
        monkeypatch.setattr(seedance, "_WARNED_ONCE", set())
        backend = seedance.SeedanceBackend(_cfg(resolution=configured))
        payload, _ = backend._translate({"PROMPT": {"text": "hi"}})
        assert payload["resolution"] == "720p", configured


def test_resolution_at_or_below_cap_passes_through(monkeypatch):
    _set_key(monkeypatch)
    backend = seedance.SeedanceBackend(_cfg(resolution="480p"))
    payload, _ = backend._translate({"PROMPT": {"text": "hi"}})
    assert payload["resolution"] == "480p"


def test_negative_prompt_maps_to_dedicated_field(monkeypatch):
    _set_key(monkeypatch)
    backend = seedance.SeedanceBackend(_cfg())
    payload, _ = backend._translate({
        "PROMPT": {"text": "hi"},
        "NEGATIVE": {"text": "blurry, extra limbs"},
    })
    assert payload["negative_prompt"] == "blurry, extra limbs"
    assert "avoid:" not in payload["prompt"]


def test_negative_prompt_fallback_when_field_unsupported(monkeypatch):
    _set_key(monkeypatch)
    monkeypatch.setattr(seedance, "_WARNED_ONCE", set())
    backend = seedance.SeedanceBackend(_cfg(negative_prompt_field=""))
    payload, _ = backend._translate({
        "PROMPT": {"text": "hi"},
        "NEGATIVE": {"text": "blurry"},
    })
    assert "negative_prompt" not in payload
    assert payload["prompt"] == "hi avoid: blurry"


def test_generate_audio_always_false(monkeypatch):
    _set_key(monkeypatch)
    backend = seedance.SeedanceBackend(_cfg())
    payload, _ = backend._translate({"PROMPT": {"text": "hi"}})
    assert payload["generate_audio"] is False


def test_reference_images_capped(monkeypatch):
    _set_key(monkeypatch)
    backend = seedance.SeedanceBackend(_cfg(max_reference_images=2))
    payload, _ = backend._translate({
        "PROMPT": {"text": "hi"},
        "REFERENCE_IMAGES": ["h1", "h2", "h3"],
    })
    assert payload["reference_images"] == ["h1", "h2"]


# ------------------------------------------------------- exception taxonomy --
def test_budget_exceeded_is_not_a_runtime_error():
    exc = seedance.SeedanceBudgetExceeded("would exceed episode budget")
    assert not isinstance(exc, RuntimeError)
    assert not isinstance(exc, TimeoutError)
    assert not isinstance(exc, requests.RequestException)
    assert isinstance(exc, seedance.SeedanceError)


def test_seedance_rejected_is_not_a_runtime_error():
    exc = seedance.SeedanceRejected("moderation rejected")
    assert not isinstance(exc, RuntimeError)
    assert not isinstance(exc, TimeoutError)
    assert not isinstance(exc, requests.RequestException)
    assert isinstance(exc, seedance.SeedanceError)


def test_dry_run_is_also_not_a_runtime_error():
    exc = seedance.SeedanceDryRun("dry run")
    assert not isinstance(exc, RuntimeError)
    assert isinstance(exc, seedance.SeedanceError)


# ---------------------------------------------------------------- transient --
def test_submit_5xx_raises_request_exception(monkeypatch):
    _set_key(monkeypatch)
    backend = seedance.SeedanceBackend(_cfg(dry_run=False))
    monkeypatch.setattr(seedance.requests, "post",
                         lambda *a, **k: _Resp(status_code=503, text="unavailable"))
    with pytest.raises(requests.RequestException):
        backend._submit({"prompt": "x"})


def test_render_5xx_bubbles_as_request_exception(monkeypatch, tmp_path):
    _set_key(monkeypatch)
    backend = seedance.SeedanceBackend(_cfg(dry_run=False))
    monkeypatch.setattr(seedance.requests, "post",
                         lambda *a, **k: _Resp(status_code=500, text="boom"))
    out = str(tmp_path / "s01_a1.mp4")
    with pytest.raises(requests.RequestException):
        backend.render("wf", {"PROMPT": {"text": "hi"}, "FRAMES": {"value": 81}}, out)
    assert not os.path.exists(out)


def test_submit_queue_full_429_raises_request_exception(monkeypatch):
    _set_key(monkeypatch)
    backend = seedance.SeedanceBackend(_cfg(dry_run=False))
    monkeypatch.setattr(seedance.requests, "post",
                         lambda *a, **k: _Resp(status_code=429, text="queue full"))
    with pytest.raises(requests.RequestException):
        backend._submit({"prompt": "x"})


# ------------------------------------------------------------------- reject --
def test_submit_auth_failure_raises_seedance_rejected(monkeypatch):
    _set_key(monkeypatch)
    backend = seedance.SeedanceBackend(_cfg(dry_run=False))
    monkeypatch.setattr(seedance.requests, "post",
                         lambda *a, **k: _Resp(status_code=401, text="bad key"))
    with pytest.raises(seedance.SeedanceRejected):
        backend._submit({"prompt": "x"})


def test_poll_moderation_rejected_raises_seedance_rejected(monkeypatch, tmp_path):
    _set_key(monkeypatch)
    backend = seedance.SeedanceBackend(_cfg(dry_run=False))
    monkeypatch.setattr(seedance.requests, "post",
                         lambda *a, **k: _Resp(_load_fixture("submit_queued.json")))
    monkeypatch.setattr(seedance.requests, "get",
                         lambda *a, **k: _Resp(_load_fixture("poll_moderation_rejected.json")))
    out = str(tmp_path / "s01_a1.mp4")
    with pytest.raises(seedance.SeedanceRejected):
        backend.render("wf", {"PROMPT": {"text": "hi"}, "FRAMES": {"value": 81}}, out)
    assert not os.path.exists(out)


def test_submit_402_raises_budget_exceeded(monkeypatch):
    _set_key(monkeypatch)
    backend = seedance.SeedanceBackend(_cfg(dry_run=False))
    monkeypatch.setattr(seedance.requests, "post",
                         lambda *a, **k: _Resp(status_code=402, text="insufficient credits"))
    with pytest.raises(seedance.SeedanceBudgetExceeded):
        backend._submit({"prompt": "x"})


# --------------------------------------------------------------- dry_run ----
def test_dry_run_never_calls_submit(monkeypatch, tmp_path):
    _set_key(monkeypatch)
    backend = seedance.SeedanceBackend(_cfg(dry_run=True))

    def boom(*a, **k):
        raise AssertionError("requests.post must never be called under dry_run")

    monkeypatch.setattr(seedance.requests, "post", boom)
    out = str(tmp_path / "s01_a1.mp4")
    with pytest.raises(seedance.SeedanceDryRun):
        backend.render("wf", {"PROMPT": {"text": "hi"}, "FRAMES": {"value": 81}}, out)
    assert not os.path.exists(out)


def test_dry_run_blocks_image_upload_too(monkeypatch, tmp_path):
    _set_key(monkeypatch)
    backend = seedance.SeedanceBackend(_cfg(dry_run=True))

    def boom(*a, **k):
        raise AssertionError("requests.post must never be called under dry_run")

    monkeypatch.setattr(seedance.requests, "post", boom)
    src = tmp_path / "ref.png"
    src.write_bytes(b"fake png")
    with pytest.raises(seedance.SeedanceDryRun):
        backend.stage_input_image(str(src))


# ------------------------------------------------------------------ budget --
def test_render_raises_budget_exceeded_before_any_http_call(monkeypatch, tmp_path):
    _set_key(monkeypatch)
    backend = seedance.SeedanceBackend(_cfg(dry_run=False, budget={"max_usd_per_episode": 0.01}))

    def boom(*a, **k):
        raise AssertionError("must not submit once the episode budget is exceeded")

    monkeypatch.setattr(seedance.requests, "post", boom)
    out = str(tmp_path / "s01_a1.mp4")
    with pytest.raises(seedance.SeedanceBudgetExceeded):
        backend.render("wf", {"PROMPT": {"text": "hi"}, "FRAMES": {"value": 81}}, out)
    assert not os.path.exists(out)


def test_max_clips_per_episode_enforced(monkeypatch, tmp_path):
    _set_key(monkeypatch)
    backend = seedance.SeedanceBackend(_cfg(dry_run=False, budget={"max_clips_per_episode": 1}))
    out_dir = tmp_path / "ep01-shots"
    out_dir.mkdir()
    # Pre-seed the ledger as if one clip was already billed this episode.
    (out_dir / "spend.json").write_text(json.dumps({"total_usd": 0.24, "clips": 1}), encoding="utf-8")

    def boom(*a, **k):
        raise AssertionError("must not submit past max_clips_per_episode")

    monkeypatch.setattr(seedance.requests, "post", boom)
    out = str(out_dir / "s02_a1.mp4")
    with pytest.raises(seedance.SeedanceBudgetExceeded):
        backend.render("wf", {"PROMPT": {"text": "hi"}, "FRAMES": {"value": 81}}, out)


# ---------------------------------------------------------- input staging ---
def test_stage_input_image_cached_by_mtime(monkeypatch, tmp_path):
    _set_key(monkeypatch)
    backend = seedance.SeedanceBackend(_cfg(dry_run=False))
    src = tmp_path / "ref.png"
    src.write_bytes(b"fake png bytes")

    calls = {"n": 0}

    def fake_upload(path):
        calls["n"] += 1
        return f"handle-{calls['n']}"

    monkeypatch.setattr(backend, "_upload_image", fake_upload)

    h1 = backend.stage_input_image(str(src))
    h2 = backend.stage_input_image(str(src))
    assert h1 == h2 == "handle-1"
    assert calls["n"] == 1

    # A retry that changes mtime (a genuinely different render) is a fresh
    # upload, not a stale cache hit.
    os.utime(src, (os.path.getatime(src) + 10, os.path.getmtime(src) + 10))
    h3 = backend.stage_input_image(str(src))
    assert h3 == "handle-2"
    assert calls["n"] == 2


# --------------------------------------------------------------- download ---
def test_download_is_atomic_on_failure(monkeypatch, tmp_path):
    _set_key(monkeypatch)
    backend = seedance.SeedanceBackend(_cfg(dry_run=False))
    out = tmp_path / "clip.mp4"

    def dying_chunks():
        yield b"partial-bytes"
        raise requests.exceptions.ConnectionError("reset")

    class _DyingStream:
        status_code = 200

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size=None):
            return dying_chunks()

    monkeypatch.setattr(seedance.requests, "get", lambda *a, **k: _DyingStream())
    with pytest.raises(requests.exceptions.ConnectionError):
        backend._download("https://cdn.example.com/x.mp4", str(out))
    assert not out.exists()
    assert not (tmp_path / "clip.mp4.part").exists()


def test_download_succeeds_and_lands(monkeypatch, tmp_path):
    _set_key(monkeypatch)
    backend = seedance.SeedanceBackend(_cfg(dry_run=False))
    out = tmp_path / "clip.mp4"

    monkeypatch.setattr(seedance.requests, "get",
                         lambda *a, **k: _StreamResp([b"video-", b"bytes"]))
    assert backend._download("https://cdn.example.com/x.mp4", str(out)) == str(out)
    assert out.read_bytes() == b"video-bytes"
    assert not (tmp_path / "clip.mp4.part").exists()


# -------------------------------------------------------------- happy path --
def test_render_happy_path_writes_spend_ledger(monkeypatch, tmp_path):
    _set_key(monkeypatch)
    backend = seedance.SeedanceBackend(_cfg(dry_run=False))

    submitted = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        submitted["url"] = url
        submitted["payload"] = json
        return _Resp(_load_fixture("submit_queued.json"))

    polls = iter([
        _Resp(_load_fixture("poll_processing.json")),
        _Resp(_load_fixture("poll_completed.json")),
    ])

    def fake_get(url, headers=None, stream=False, timeout=None):
        if stream:
            return _StreamResp([b"video-bytes"])
        return next(polls)

    monkeypatch.setattr(seedance.requests, "post", fake_post)
    monkeypatch.setattr(seedance.requests, "get", fake_get)

    out_dir = tmp_path / "ep01-shots"
    out_dir.mkdir()
    out_path = str(out_dir / "s01_a1.mp4")
    patches = {
        "PROMPT": {"text": "P1x4r, a happy chick dances"},
        "NEGATIVE": {"text": "blurry"},
        "SEED": {"noise_seed": 7},
        "FRAMES": {"value": 81},
    }

    result = backend.render("ltx23_t2v_toon", patches, out_path)
    assert result == out_path
    with open(out_path, "rb") as f:
        assert f.read() == b"video-bytes"

    assert submitted["url"].endswith("/v1/video/generations")
    assert submitted["payload"]["prompt"] == "a happy chick dances"
    assert submitted["payload"]["negative_prompt"] == "blurry"
    assert submitted["payload"]["seed"] == 7
    assert submitted["payload"]["duration"] == 4
    assert submitted["payload"]["resolution"] == "720p"
    assert submitted["payload"]["generate_audio"] is False

    ledger_path = out_dir / "spend.json"
    assert ledger_path.exists()
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert ledger["clips"] == 1
    assert ledger["total_usd"] == pytest.approx(4 * 0.06)


# -------------------------------------------------------------- duck-typed --
def test_free_is_a_no_op_and_cache_survives(monkeypatch, tmp_path):
    _set_key(monkeypatch)
    backend = seedance.SeedanceBackend(_cfg(dry_run=False))
    src = tmp_path / "ref.png"
    src.write_bytes(b"x")
    monkeypatch.setattr(backend, "_upload_image", lambda path: "handle-1")
    backend.stage_input_image(str(src))
    backend.free()
    assert backend._image_cache  # survived free()


def test_restart_if_hung_returns_false(monkeypatch):
    _set_key(monkeypatch)
    backend = seedance.SeedanceBackend(_cfg())
    assert backend.restart_if_hung() is False


def test_accepts_reference_images_flag():
    assert seedance.SeedanceBackend.accepts_reference_images is True
