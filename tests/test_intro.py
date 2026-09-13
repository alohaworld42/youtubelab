"""Tests for the fixed ZubiBop branding intro (pipeline/kidsong/intro.py) and
its concat wiring in pipeline/kidsong/edit.py.

No GPU and no network. The couple of cases that genuinely need a container
(the concat helpers) shell out to ffmpeg with tiny synthetic clips and skip
themselves when ffmpeg is not on PATH.
"""
import json
import os
import shutil
import subprocess

import pytest

from pipeline.kidsong import intro as intro_mod

HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
needs_ffmpeg = pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg/ffprobe not on PATH")


# ------------------------------------------------------------------ helpers ---
def _logo(path, size=240, color=(70, 200, 180)):
    """A stand-in for the channel badge: a square PNG with black corners, the
    same shape as the real asset (circle inscribed, no alpha channel)."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (size, size), (0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse((0, 0, size - 1, size - 1), fill=color)
    d.rectangle((size * 0.3, size * 0.55, size * 0.7, size * 0.7), fill=(250, 210, 60))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)
    return path


def _cfg(root, **intro_over):
    intro = {
        "enabled": True,
        "path": "assets/intro/zubibop_intro.mp4",
        "logo": "assets/intro/zubibop_logo.png",
        "seconds": 1.0,
        "target_lufs": -14.0,
        "at_assemble": False,
    }
    intro.update(intro_over)
    return {
        "_root": str(root),
        "video": {"width": 1080, "height": 1920, "fps": 30},
        "kidsong": {"output": {"width": 320, "height": 180}, "intro": intro},
    }


def _write_manifest(params, digest):
    man = intro_mod.manifest_path(params["path"])
    os.makedirs(os.path.dirname(man), exist_ok=True)
    with open(man, "w", encoding="utf-8") as f:
        json.dump({"hash": digest, "params": params, "version": intro_mod._VERSION}, f)


def _tiny_mp4(path, seconds=1.0, freq=440):
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "lavfi", "-i", f"testsrc=size=160x90:rate=30:duration={seconds}",
         "-f", "lavfi", "-i", f"sine=frequency={freq}:duration={seconds}",
         "-c:v", "libx264", "-profile:v", "high", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-ar", "44100", "-ac", "2", "-shortest", path],
        check=True,
    )
    return path


def _duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", path],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return float(out)


class _Rendered(Exception):
    """Raised by the stubbed compositor to prove a rebuild was attempted."""


# ================================================================== caching ===
def test_cache_is_reused_when_inputs_are_unchanged(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    params = intro_mod.intro_params(cfg)
    _logo(params["logo"])
    os.makedirs(os.path.dirname(params["path"]), exist_ok=True)
    open(params["path"], "wb").write(b"pretend-mp4")
    _write_manifest(params, intro_mod.content_hash(params))

    # Any attempt to actually render is a cache miss, and a cache miss here is
    # the bug: branding that re-renders per episode is branding that drifts.
    monkeypatch.setattr(intro_mod, "build_context", lambda p: (_ for _ in ()).throw(_Rendered()))

    assert intro_mod.is_current(params)
    assert intro_mod.build_intro(cfg, on_progress=lambda m: None) == params["path"]


@pytest.mark.parametrize("mutate", ["logo_bytes", "seconds", "version"])
def test_cache_is_rebuilt_when_an_input_changes(tmp_path, monkeypatch, mutate):
    cfg = _cfg(tmp_path)
    params = intro_mod.intro_params(cfg)
    _logo(params["logo"])
    os.makedirs(os.path.dirname(params["path"]), exist_ok=True)
    open(params["path"], "wb").write(b"pretend-mp4")
    _write_manifest(params, intro_mod.content_hash(params))
    assert intro_mod.is_current(params)

    if mutate == "logo_bytes":
        # Same filename, different pixels — the hash must still notice.
        _logo(params["logo"], color=(200, 60, 60))
        new_cfg = cfg
    elif mutate == "seconds":
        new_cfg = _cfg(tmp_path, seconds=2.5)
    else:
        monkeypatch.setattr(intro_mod, "_VERSION", intro_mod._VERSION + 1)
        new_cfg = cfg

    new_params = intro_mod.intro_params(new_cfg)
    assert not intro_mod.is_current(new_params)

    monkeypatch.setattr(intro_mod, "build_context", lambda p: (_ for _ in ()).throw(_Rendered()))
    with pytest.raises(_Rendered):
        intro_mod.build_intro(new_cfg, on_progress=lambda m: None)


def test_rebuild_flag_ignores_a_valid_cache(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    params = intro_mod.intro_params(cfg)
    _logo(params["logo"])
    os.makedirs(os.path.dirname(params["path"]), exist_ok=True)
    open(params["path"], "wb").write(b"pretend-mp4")
    _write_manifest(params, intro_mod.content_hash(params))

    monkeypatch.setattr(intro_mod, "build_context", lambda p: (_ for _ in ()).throw(_Rendered()))
    with pytest.raises(_Rendered):
        intro_mod.build_intro(cfg, rebuild=True, on_progress=lambda m: None)


def test_hash_is_stable_across_calls(tmp_path):
    cfg = _cfg(tmp_path)
    params = intro_mod.intro_params(cfg)
    _logo(params["logo"])
    assert intro_mod.content_hash(params) == intro_mod.content_hash(params)


# ==================================================================== audio ===
def test_sting_has_the_requested_length_and_is_not_silent():
    np = pytest.importorskip("numpy")
    sr = 22050
    samples = intro_mod.synth_sting(seconds=2.0, sr=sr)

    assert samples.shape == (int(2.0 * sr), 2)
    assert samples.dtype == np.float32
    assert float(np.max(np.abs(samples))) > 0.2, "sting is silent or far too quiet"
    assert float(np.max(np.abs(samples))) <= 1.0, "sting clips"
    # Real content, not DC or a single click: energy spread over many samples.
    assert float(np.mean(np.abs(samples))) > 0.005


def test_sting_ends_in_silence_so_it_never_overlaps_the_song():
    """At the shipping length the sting must be gone, not merely fading: the
    song starts on the very next frame and must not be talked over."""
    np = pytest.importorskip("numpy")
    sr = 44100
    samples = intro_mod.synth_sting(seconds=4.0, sr=sr)
    peak = float(np.max(np.abs(samples)))

    last_100ms = float(np.max(np.abs(samples[-int(0.1 * sr):])))
    assert 20 * np.log10(max(last_100ms, 1e-12) / peak) < -40.0, "sting is still audible at the cut"
    # Below a 16-bit LSB (1/32768 ~ 3e-5), i.e. digital silence once encoded.
    assert float(np.max(np.abs(samples[-20:]))) < 1e-6, "sting does not reach silence"


def test_sting_is_deterministic():
    np = pytest.importorskip("numpy")
    a = intro_mod.synth_sting(seconds=1.0, sr=22050)
    b = intro_mod.synth_sting(seconds=1.0, sr=22050)
    assert np.array_equal(a, b), "branding audio must be identical every build"


def test_motif_notes_land_inside_the_sting():
    for start, note, vel in intro_mod.MOTIF:
        assert 0.0 <= start < 4.0
        assert 0.0 < vel <= 1.0
        assert 21 <= note <= 108
    chord_start, chord_notes, _ = intro_mod.RESOLVE_CHORD
    assert chord_start >= intro_mod.MOTIF[-1][0], "the chord must resolve after the run"
    assert len(chord_notes) >= 3


# ================================================================== picture ===
def test_composed_frames_have_the_configured_size(tmp_path):
    np = pytest.importorskip("numpy")
    cfg = _cfg(tmp_path)
    params = intro_mod.intro_params(cfg)
    _logo(params["logo"])

    ctx = intro_mod.build_context(params)
    for t in (0.0, 0.3, params["seconds"] * 0.99):
        frame = intro_mod.compose_frame(t, ctx)
        assert frame.shape == (params["height"], params["width"], 3)
        assert frame.dtype == np.uint8


def test_animation_actually_progresses(tmp_path):
    np = pytest.importorskip("numpy")
    cfg = _cfg(tmp_path, seconds=4.0)
    params = intro_mod.intro_params(cfg)
    _logo(params["logo"])
    ctx = intro_mod.build_context(params)

    early = intro_mod.compose_frame(0.0, ctx).astype(np.int16)
    entering = intro_mod.compose_frame(0.25, ctx).astype(np.int16)
    accent = intro_mod.compose_frame(intro_mod.T_ACCENT + 0.4, ctx).astype(np.int16)

    assert np.abs(entering - early).mean() > 1.0, "badge never appears"
    assert np.abs(accent - entering).mean() > 0.5, "nothing happens on the accent"


def test_badge_stays_inside_the_frame_through_the_overshoot(tmp_path):
    """The spring overshoots past 1.0 — the badge must still fit."""
    peak = max(intro_mod._spring(e / 200.0) for e in range(1, 400))
    assert peak > 1.0, "no overshoot: the entrance would read as a plain fade"
    assert intro_mod.BADGE_HEIGHT_RATIO * peak < 1.0, "badge is clipped at its largest"


def test_spring_starts_at_zero_and_settles_at_one():
    assert intro_mod._spring(0.0) == 0.0
    assert intro_mod._spring(-1.0) == 0.0
    assert intro_mod._spring(4.0) == pytest.approx(1.0, abs=1e-3)


# ============================================================ graceful skip ===
def test_missing_logo_does_not_crash_a_render(tmp_path):
    """A missing brand asset degrades to "no intro", never to a failed episode."""
    cfg = _cfg(tmp_path)  # logo deliberately not created
    params = intro_mod.intro_params(cfg)
    assert not os.path.exists(params["logo"])

    with pytest.raises(intro_mod.IntroAssetMissing):
        intro_mod.content_hash(params)

    logged = []
    assert intro_mod.ensure_intro(cfg, on_progress=logged.append) is None
    assert any("skip" in m.lower() for m in logged), f"skip was not logged: {logged}"


def test_disabled_intro_is_skipped_without_touching_the_asset(tmp_path):
    cfg = _cfg(tmp_path, enabled=False)
    params = intro_mod.intro_params(cfg)
    _logo(params["logo"])
    assert intro_mod.ensure_intro(cfg, on_progress=lambda m: None) is None
    assert not os.path.exists(params["path"])


def test_prepend_intro_leaves_the_video_untouched_when_disabled(tmp_path):
    from pipeline.kidsong.edit import prepend_intro

    video = tmp_path / "episode.mp4"
    video.write_bytes(b"an-episode")
    cfg = _cfg(tmp_path, enabled=False)

    result = prepend_intro(str(video), cfg, on_progress=lambda m: None)
    assert result["applied"] is False
    assert result["reason"] == "disabled"
    assert video.read_bytes() == b"an-episode"


def test_prepend_intro_leaves_the_video_untouched_when_the_logo_is_gone(tmp_path):
    from pipeline.kidsong.edit import prepend_intro

    video = tmp_path / "episode.mp4"
    video.write_bytes(b"an-episode")
    cfg = _cfg(tmp_path)  # enabled, but no logo on disk

    logged = []
    result = prepend_intro(str(video), cfg, on_progress=logged.append)
    assert result["applied"] is False
    assert video.read_bytes() == b"an-episode"
    assert any("skip" in m.lower() for m in logged), f"skip was not logged: {logged}"


# =================================================================== concat ===
@needs_ffmpeg
def test_concat_copy_produces_the_summed_duration(tmp_path):
    from pipeline.kidsong.edit import _concat_files

    a = _tiny_mp4(str(tmp_path / "a.mp4"), seconds=1.0, freq=660)
    b = _tiny_mp4(str(tmp_path / "b.mp4"), seconds=2.0, freq=330)
    out = str(tmp_path / "joined.mp4")

    _concat_files([a, b], out, reencode=False)
    assert _duration(out) == pytest.approx(_duration(a) + _duration(b), abs=0.12)


@needs_ffmpeg
def test_concat_reencode_fallback_produces_the_summed_duration(tmp_path):
    from pipeline.kidsong.edit import _concat_files

    a = _tiny_mp4(str(tmp_path / "a.mp4"), seconds=1.0, freq=660)
    b = _tiny_mp4(str(tmp_path / "b.mp4"), seconds=2.0, freq=330)
    out = str(tmp_path / "joined.mp4")

    _concat_files([a, b], out, reencode=True, fps=30)
    assert _duration(out) == pytest.approx(_duration(a) + _duration(b), abs=0.2)


@needs_ffmpeg
def test_prepend_intro_joins_and_records_the_offset(tmp_path, monkeypatch):
    """End-to-end with a stand-in intro: the episode gains exactly the intro's
    length, and the song's new start time is written down rather than implied."""
    from pipeline.kidsong import edit as edit_mod

    fake_intro = _tiny_mp4(str(tmp_path / "intro.mp4"), seconds=1.0, freq=880)
    episode = _tiny_mp4(str(tmp_path / "episode.mp4"), seconds=2.0, freq=220)
    episode_len = _duration(episode)

    cfg = _cfg(tmp_path)
    # prepend_intro imports the intro module lazily, so patching the module
    # attribute is what the real call path will see.
    monkeypatch.setattr(intro_mod, "ensure_intro", lambda c, on_progress=None: fake_intro)

    result = edit_mod.prepend_intro(episode, cfg, on_progress=lambda m: None)
    assert result["applied"] is True

    total = _duration(episode)
    assert total == pytest.approx(episode_len + _duration(fake_intro), abs=0.15)

    with open(episode + ".intro.json", encoding="utf-8") as f:
        sidecar = json.load(f)
    assert sidecar["song_starts_at"] == pytest.approx(result["intro_seconds"], abs=1e-3)
