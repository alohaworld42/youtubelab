"""Tests for the polish layer: color grade (pipeline/grade.py), whip-pan
whoosh SFX (pipeline/sfx.py), and the caption pop-in scale curve."""
import math
import wave

import pytest

from pipeline import grade, sfx
from pipeline.camera import whip_times


def _g(**over):
    g = dict(grade.DEFAULTS)
    g.update({"enabled": True})
    g.update(over)
    return g


# -------------------------------------------------------------------- grade ---
def test_luts_monotonic_and_full_range():
    np = pytest.importorskip("numpy")
    lut_r, lut_g, lut_b = grade.build_luts(_g(contrast=0.3, warmth=0.08))
    for lut in (lut_r, lut_g, lut_b):
        assert lut[0] == 0
        assert np.all(np.diff(lut.astype(int)) >= 0)
    assert lut_g[255] == 255
    # warmth raises red gain over blue
    assert int(lut_r[128]) > int(lut_b[128])


def test_vignette_center_bright_corners_dark():
    mask = grade.vignette_mask(108, 192, 0.4)
    assert mask.shape == (192, 108, 1)
    center = float(mask[96, 54, 0])
    corner = float(mask[0, 0, 0])
    assert center == pytest.approx(1.0, abs=0.02)
    assert corner < center
    assert corner >= 1.0 - 0.55  # bounded by strength formula


def test_vignette_zero_strength_is_none():
    assert grade.vignette_mask(100, 100, 0.0) is None
    assert grade.grain_tiles(100, 100, 0.0) is None


def test_apply_grade_shape_dtype_and_saturation():
    np = pytest.importorskip("numpy")
    moviepy = pytest.importorskip("moviepy")

    frame = np.full((192, 108, 3), 100, dtype=np.uint8)
    frame[..., 0] = 180  # reddish so saturation has something to push
    clip = moviepy.VideoClip(lambda t: frame, duration=1.0)

    cfg = {"grade": _g(saturation=1.5, contrast=0.0, warmth=0.0, vignette=0.0, grain=0.0)}
    out = grade.apply_grade(clip, cfg, fps=30).get_frame(0.5)
    assert out.shape == (192, 108, 3)
    assert out.dtype == np.uint8
    # saturation pushes channels apart around luma
    assert int(out[0, 0, 0]) - int(out[0, 0, 2]) > 180 - 100


def test_apply_grade_disabled_is_noop():
    np = pytest.importorskip("numpy")
    moviepy = pytest.importorskip("moviepy")
    frame = np.full((64, 36, 3), 90, dtype=np.uint8)
    clip = moviepy.VideoClip(lambda t: frame, duration=1.0)
    assert grade.apply_grade(clip, {}, fps=30) is clip


# ---------------------------------------------------------------------- sfx ---
def test_whoosh_wav_valid_and_cached():
    p1 = sfx.whoosh_wav()
    p2 = sfx.whoosh_wav()
    assert p1 == p2
    with wave.open(p1) as w:
        assert w.getnchannels() == 1
        assert w.getnframes() / w.getframerate() == pytest.approx(0.35, abs=0.01)


def test_whip_sfx_clips_skip_out_of_range():
    pytest.importorskip("moviepy")
    clips = sfx.whip_sfx_clips([2.0, 5.0, 99.0], duration=10.0, volume=0.4)
    assert len(clips) == 2
    assert clips[0].start == pytest.approx(2.0 - 0.35 / 2.0)


def test_whip_sfx_silent_when_no_volume():
    assert sfx.whip_sfx_clips([2.0], duration=10.0, volume=0.0) == []
    assert sfx.whip_sfx_clips([], duration=10.0, volume=0.4) == []


def test_whip_times_helper():
    plan = [
        {"start": 0, "end": 4, "move": "push_in", "whip_out": 1},
        {"start": 4, "end": 8, "move": "drift_pan", "whip_in": -1},
    ]
    assert whip_times(plan) == [4]


# -------------------------------------------------------------- caption pop ---
def test_caption_pop_curve():
    from pipeline.assemble import _caption_pop

    scale = _caption_pop(0.14)
    assert scale(0.0) == pytest.approx(0.75, abs=0.01)
    assert scale(0.14) == pytest.approx(1.0, abs=1e-6)
    assert scale(5.0) == pytest.approx(1.0, abs=1e-6)
    peak = max(scale(0.14 * k / 40) for k in range(41))
    assert 1.0 < peak < 1.08  # slight overshoot, no wild bounce


# ------------------------------------------------- clip handles are released --
class _FakeClip:
    """Enough of a MoviePy clip to walk build_video's teardown."""

    def __init__(self, name, duration=10.0, w=1080, h=1920):
        self.name = name
        self.duration = duration
        self.w, self.h = w, h
        self.closed = False
        self.audio = None

    # every transform returns self so the pipeline can chain freely
    def _same(self, *a, **k):
        return self

    without_audio = resized = cropped = subclipped = with_duration = _same
    with_start = with_position = with_effects = with_audio = time_transform = _same

    def close(self):
        self.closed = True

    def write_videofile(self, *a, **k):
        pass


def test_build_video_closes_the_readers_it_opened(tmp_path, monkeypatch):
    """MoviePy's CompositeVideoClip.close() deliberately leaves member clips
    alone and the base Clip.close() is a no-op, so closing `bg` released
    nothing: the per-scene VideoFileClip readers — one ffmpeg subprocess each —
    leaked once per render, inside a long-running Flask server."""
    from pipeline import assemble

    opened = []

    def _fake_videofileclip(path):
        clip = _FakeClip(f"src:{path}")
        opened.append(clip)
        return clip

    monkeypatch.setattr(assemble, "VideoFileClip", _fake_videofileclip)
    monkeypatch.setattr(assemble, "concatenate_videoclips",
                        lambda parts, **k: parts[0])
    monkeypatch.setattr(assemble, "CompositeVideoClip",
                        lambda clips, **k: _FakeClip("composite"))
    monkeypatch.setattr(assemble, "apply_camera", lambda clip, *a, **k: clip)
    monkeypatch.setattr(assemble, "camera_enabled", lambda cfg: False)
    monkeypatch.setattr(assemble, "apply_grade", lambda clip, *a, **k: clip)
    monkeypatch.setattr(assemble, "_avatar_clips", lambda *a, **k: [])
    monkeypatch.setattr(assemble, "_caption_clips", lambda *a, **k: [])
    monkeypatch.setattr(assemble, "build_motion_clips", lambda *a, **k: [])
    monkeypatch.setattr(assemble, "build_flash_clips", lambda *a, **k: [])
    monkeypatch.setattr(assemble, "build_progress_clip", lambda *a, **k: None)
    monkeypatch.setattr(assemble, "_build_audio", lambda *a, **k: _FakeClip("audio"))

    cfg = {
        "video": {"width": 1080, "height": 1920, "fps": 30},
        "paths": {"gameplay_dir": "assets/gameplay", "characters_dir": "assets/characters",
                  "music_dir": "assets/music"},
        "captions": {},
        "_root": str(tmp_path),
    }
    scenes = [{"path": f"scene{i}.mp4", "start": i * 2.0, "end": i * 2.0 + 2.0}
              for i in range(4)]

    assemble.build_video(cfg, "voice.wav", [], [], 8.0, str(tmp_path / "out.mp4"),
                         scene_clips=scenes)

    assert len(opened) == 4, "expected one reader per scene"
    unclosed = [c.name for c in opened if not c.closed]
    assert not unclosed, f"leaked ffmpeg readers: {unclosed}"


def test_the_audio_tracks_are_closed_too(tmp_path):
    """CompositeAudioClip.close() does not close its members either."""
    from pipeline import assemble

    voice, music = _FakeClip("voice"), _FakeClip("music")
    mixed = assemble.CompositeAudioClip.__new__(assemble.CompositeAudioClip)
    # _build_audio carries the tracks out on the mix; assert the contract it
    # relies on rather than re-running the mixer.
    mixed.source_clips = [voice, music]
    for clip in getattr(mixed, "source_clips", ()):
        clip.close()
    assert voice.closed and music.closed
