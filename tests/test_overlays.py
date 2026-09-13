"""Tests for the hook title card, progress bar, reveal flash, and camera
zoom impulses."""
import random

import pytest

from pipeline import camera
from pipeline.generate import _hook_spec
from pipeline.motion import (
    _make_component,
    build_flash_clips,
    build_motion_clips,
    build_progress_clip,
)

CFG = {
    "motion": {
        "enabled": True,
        "font_path": "C:/Windows/Fonts/arialbd.ttf",
        "accent_color": [255, 214, 40],
        "card_color": [22, 20, 30],
        "flash": True,
        "flash_opacity": 0.16,
        "flash_seconds": 0.1,
    },
    "hook": {"enabled": True, "seconds": 1.4, "font_size": 80},
    "progress_bar": {"enabled": True, "height": 10, "opacity": 0.85, "track_opacity": 0.18},
}


# ---------------------------------------------------------------- hook spec ---
def test_hook_spec_default():
    spec = _hook_spec(CFG, "Cats are evil 😼", 30.0)
    assert spec["type"] == "title"
    assert spec["start"] == pytest.approx(0.1)
    assert spec["end"] == pytest.approx(1.5)


def test_hook_spec_disabled_or_empty():
    assert _hook_spec({"hook": {"enabled": False}}, "Title", 30.0) is None
    assert _hook_spec(CFG, "   ", 30.0) is None
    assert _hook_spec(CFG, None, 30.0) is None


def test_hook_spec_no_room_in_tiny_video():
    assert _hook_spec(CFG, "Title", 0.4) is None


# --------------------------------------------------------------- title card ---
def test_title_card_renders_and_strips_emoji():
    comp = _make_component({"type": "title", "text": "Cats are evil 😼🔥"}, CFG, 1080)
    assert comp is not None
    assert all("😼" not in ln and "🔥" not in ln for ln in comp.lines)
    img = comp.render(0.0, 1.4)  # max oversize frame — must fit its canvas
    assert img.size == comp.canvas
    mid = comp.render(0.7, 1.4)
    assert mid.getextrema()[3][1] > 200  # visible at mid-hold
    assert img.getextrema()[3][1] == 0  # fully transparent at t=0


def test_title_card_canvas_fits_stamp_scale():
    comp = _make_component({"type": "title", "text": "A REALLY LONG SHOCKING TITLE THAT WRAPS"}, CFG, 1080)
    assert comp.canvas[0] >= int(comp.card_w * 1.3)
    assert comp.canvas[1] >= int(comp.card_h * 1.3)


def test_y_ratio_positions_title_higher():
    pytest.importorskip("moviepy")
    specs = [
        {"type": "title", "text": "HOOK", "start": 0.0, "end": 1.4, "y_ratio": 0.18},
        {"type": "card", "text": "normal card", "start": 2.0, "end": 3.0},
    ]
    clips = build_motion_clips(CFG, specs, (1080, 1920), 30)
    assert len(clips) == 2
    y_title = clips[0].pos(0)[1]
    y_card = clips[1].pos(0)[1]
    assert y_title < y_card


# ------------------------------------------------------------- progress bar ---
def test_progress_bar_fills_over_time():
    np = pytest.importorskip("numpy")
    pytest.importorskip("moviepy")
    clip = build_progress_clip(CFG, 10.0, (1080, 1920), 30)
    assert clip is not None
    assert clip.pos(0) == (0, 1920 - 10)
    a_half = clip.mask.get_frame(5.0)
    assert a_half.shape == (10, 1080)
    assert a_half[0, 100] == pytest.approx(0.85)   # filled left
    assert a_half[0, 1000] == pytest.approx(0.18)  # track right
    a_end = clip.mask.get_frame(9.99)
    assert float(np.mean(a_end > 0.5)) > 0.99


def test_progress_bar_disabled():
    assert build_progress_clip({"progress_bar": {"enabled": False}}, 10.0, (1080, 1920), 30) is None


# ------------------------------------------------------------------- flash ---
def test_flash_clips_count_and_decay():
    pytest.importorskip("moviepy")
    clips = build_flash_clips(CFG, [1.0, 5.0, 98.0], (108, 192), 30, video_duration=10.0)
    assert len(clips) == 2
    assert clips[0].start == pytest.approx(1.0)
    m0 = clips[0].mask.get_frame(0.0)
    m_late = clips[0].mask.get_frame(0.09)
    assert m0[0, 0] == pytest.approx(0.16)
    assert m_late[0, 0] < m0[0, 0]


def test_flash_disabled():
    cfg = {"motion": {"flash": False}}
    assert build_flash_clips(cfg, [1.0], (108, 192), 30, 10.0) == []


# ---------------------------------------------------------- camera impulses ---
def _static_cam(**over):
    cam = dict(camera.DEFAULTS)
    cam.update({"enabled": True, "moves": ["static"], "shake": 0.0,
                "transition": "none", "impulse_zoom": 0.08, "impulse_seconds": 0.4})
    cam.update(over)
    return cam


def _flat_clip():
    np = pytest.importorskip("numpy")
    moviepy = pytest.importorskip("moviepy")
    grad = np.linspace(0, 255, 384, dtype=np.uint8)
    frame = np.tile(grad[None, :, None], (216, 1, 3))
    return moviepy.VideoClip(lambda t: frame, duration=4.0)


def test_impulse_zooms_at_reveal_time():
    np = pytest.importorskip("numpy")
    plan = [{"start": 0.0, "end": 4.0, "move": "static", "dir": 1,
             "diag": ((0.7, 0.7), (-0.7, -0.7)), "phases": (0, 0, 0, 0)}]
    clip = _flat_clip()

    quiet = camera.apply_camera(clip, [dict(plan[0])], _static_cam(), (108, 192), 30)
    punched = camera.apply_camera(
        clip, [dict(plan[0])], _static_cam(_impulse_times=[2.0]), (108, 192), 30)

    before = (quiet.get_frame(1.5), punched.get_frame(1.5))
    at = (quiet.get_frame(2.05), punched.get_frame(2.05))
    assert np.array_equal(*before)          # identical before the impulse
    assert not np.array_equal(*at)          # zoom punch visible right after
    after = (quiet.get_frame(2.9), punched.get_frame(2.9))
    assert np.array_equal(*after)           # fully decayed again
