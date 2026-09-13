"""Stage 1 tests for the video-render backend seam (kidsong.render_backend).

The seam's whole point is that the DEFAULT path is byte-identical to before
it existed: `make_video_backend` must return the ComfyClient it was handed,
unchanged, whenever `kidsong.render_backend` is absent or "comfy" -- not a
wrapper, the SAME object -- so every existing render-loop call
(`renderer.render(...)`, `renderer.free()`, ...) resolves to exactly the
methods it always did.

CPU-only, no GPU, no network: `ComfyClient(cfg)` only parses config and a
URL at construction (see pipeline/kidsong/comfy.py), it never reaches the
network until a render/ensure_up call, none of which this file makes.
"""
import pytest

from pipeline.kidsong.comfy import ComfyClient
from pipeline.kidsong.video_backend import make_video_backend


def _cfg(render_backend=None):
    cfg = {"kidsong": {"mode": "director"}}
    if render_backend is not None:
        cfg["kidsong"]["render_backend"] = render_backend
    return cfg


def test_default_backend_is_the_comfy_client_itself():
    """The byte-identical proof: with kidsong.render_backend="comfy", the
    returned object IS `client` -- not an equal-but-different wrapper. This
    is what lets every existing `client.foo(...)` call site become
    `renderer.foo(...)` with zero behavior change on the default path."""
    cfg = _cfg(render_backend="comfy")
    client = ComfyClient(cfg)

    assert make_video_backend(cfg, client) is client


def test_absent_key_behaves_like_comfy():
    """No kidsong.render_backend key at all (today's config shape, and every
    config.example.json/live config that predates this seam) must behave
    exactly like the explicit "comfy" case above."""
    cfg = _cfg()
    assert "render_backend" not in cfg["kidsong"]
    client = ComfyClient(cfg)

    assert make_video_backend(cfg, client) is client


def test_unknown_backend_name_raises_valueerror():
    cfg = _cfg(render_backend="wan-2.2-turbo")
    client = ComfyClient(cfg)

    with pytest.raises(ValueError, match="wan-2.2-turbo"):
        make_video_backend(cfg, client)


def test_seedance_without_keys_raises_at_construction_not_mid_episode(monkeypatch):
    """A `render_backend: seedance` config with no SEEDANCE_API_KEY (and/or
    seedance.enabled not true) must fail as soon as make_video_backend is
    called -- i.e. before the render loop, before the song is sung -- not on
    the first render() call several minutes into an episode."""
    pytest.importorskip(
        "pipeline.kidsong.seedance",
        reason="pipeline/kidsong/seedance.py not present in this checkout yet",
    )
    monkeypatch.delenv("SEEDANCE_API_KEY", raising=False)
    monkeypatch.delenv("SEEDANCE_BASE_URL", raising=False)

    cfg = _cfg(render_backend="seedance")
    # seedance.enabled left unset/false on top of the missing key -- either
    # gap alone must be enough for is_available() to say no.
    client = ComfyClient(cfg)

    with pytest.raises(Exception):
        make_video_backend(cfg, client)


def test_reference_images_patch_is_absent_for_comfy():
    """Guards a real landmine: ComfyUI's patch applier raises
    `KeyError: No node titled ...` for a REFERENCE_IMAGES-style patch title
    it doesn't recognize. A ComfyClient-backed renderer must not advertise
    `accepts_reference_images`, so the render loop never builds that patch
    for it."""
    cfg = _cfg(render_backend="comfy")
    client = ComfyClient(cfg)
    backend = make_video_backend(cfg, client)

    assert getattr(backend, "accepts_reference_images", False) is False


def test_the_shipped_config_key_video_backend_is_not_this_seam():
    """Regression: `kidsong.video_backend` ALREADY EXISTS and means something
    else — which local ComfyUI animation model to use, "wan" | "ltx", read by
    `animate.py`. This seam reads `kidsong.render_backend`.

    Before the rename this seam read `video_backend`, so the shipped default
    (`config.example.json`: `"video_backend": "wan"`) made `make_video_backend`
    raise ValueError and every kidsong run died at `generate.py`'s
    `renderer = make_video_backend(cfg, client)` line. The two keys must stay
    independent.
    """
    client = object()
    cfg = {"kidsong": {"video_backend": "wan"}}
    assert make_video_backend(cfg, client) is client

    # and both keys can coexist without interfering
    cfg = {"kidsong": {"video_backend": "wan", "render_backend": "comfy"}}
    assert make_video_backend(cfg, client) is client
