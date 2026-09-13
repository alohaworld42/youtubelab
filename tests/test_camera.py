"""Tests for the virtual camera engine (pipeline/camera.py).

The math core (plan_segments / camera_state / crop_box) is pure and tested
exhaustively; apply_camera gets a small synthetic-clip smoke test.

Manual E2E check (slow, not part of CI):
    python -m pipeline.generate facts --topic "test camera"
with camera.enabled=true in config.json — background should push/pan/shake
per spoken line with whip-pan cuts between beats.
"""
import math
import random

import pytest

from pipeline import camera


def _cam(**over):
    cam = dict(camera.DEFAULTS)
    cam.update({"enabled": True})
    cam.update(over)
    return cam


def _lines(*windows):
    return [{"start": s, "end": e} for s, e in windows]


# ---------------------------------------------------------------- planning ---
def test_plan_covers_duration_exactly():
    cam = _cam()
    plan = camera.plan_segments(_lines((0, 3), (3, 6.2), (6.2, 11)), 11.0, cam, random.Random(1))
    assert plan[0]["start"] == 0.0
    assert plan[-1]["end"] == 11.0
    for a, b in zip(plan, plan[1:]):
        assert a["end"] == pytest.approx(b["start"])


def test_plan_fills_gaps_and_clamps_to_duration():
    # gap between lines and a line past the video end
    plan = camera.plan_segments(_lines((0, 3), (5, 8), (9, 40)), 12.0, _cam(), random.Random(1))
    assert plan[0]["start"] == 0.0
    assert plan[-1]["end"] == 12.0
    total = sum(s["end"] - s["start"] for s in plan)
    assert total == pytest.approx(12.0)


def test_plan_merges_short_beats():
    cam = _cam(min_beat_seconds=2.5, max_beat_seconds=7.0)
    plan = camera.plan_segments(_lines((0, 1), (1, 1.8), (1.8, 5)), 5.0, cam, random.Random(1))
    for seg in plan:
        assert seg["end"] - seg["start"] >= 2.5 - 1e-6


def test_plan_splits_long_beats():
    cam = _cam(min_beat_seconds=2.5, max_beat_seconds=7.0)
    plan = camera.plan_segments(_lines((0, 20)), 20.0, cam, random.Random(1))
    assert len(plan) >= 3
    for seg in plan:
        assert seg["end"] - seg["start"] <= 7.0 + 1e-6


def test_plan_no_consecutive_duplicate_moves():
    cam = _cam(moves=["push_in", "pull_out", "drift_pan"])
    windows = [(k * 3.0, (k + 1) * 3.0) for k in range(20)]
    plan = camera.plan_segments(_lines(*windows), 60.0, cam, random.Random(7))
    for a, b in zip(plan, plan[1:]):
        assert a["move"] != b["move"]


def test_plan_hint_pins_move():
    segs = _lines((0, 4), (4, 8), (8, 12))
    segs[1]["camera_hint"] = "punch_in"
    plan = camera.plan_segments(segs, 12.0, _cam(), random.Random(3))
    hit = [s for s in plan if s["start"] == pytest.approx(4.0)]
    assert hit and hit[0]["move"] == "punch_in"


def test_plan_without_segments_falls_back_to_fixed_beats():
    plan = camera.plan_segments([], 30.0, _cam(), random.Random(1))
    assert plan[0]["start"] == 0.0
    assert plan[-1]["end"] == 30.0
    assert len(plan) > 1


def test_plan_whip_flags_paired_and_opposite():
    cam = _cam(transition="whip_pan")
    windows = [(k * 4.0, (k + 1) * 4.0) for k in range(5)]
    plan = camera.plan_segments(_lines(*windows), 20.0, cam, random.Random(2))
    for a, b in zip(plan, plan[1:]):
        assert a.get("whip_out") in (-1, 1)
        assert b.get("whip_in") == -a["whip_out"]


def test_plan_no_whip_flags_when_transition_off():
    cam = _cam(transition="none")
    plan = camera.plan_segments(_lines((0, 4), (4, 8)), 8.0, cam, random.Random(2))
    assert not any("whip_out" in s or "whip_in" in s for s in plan)


def test_plan_seed_determinism():
    cam = _cam(seed=42)
    windows = [(k * 4.0, (k + 1) * 4.0) for k in range(8)]
    p1 = camera.plan_segments(_lines(*windows), 32.0, cam, camera.make_rng(cam))
    p2 = camera.plan_segments(_lines(*windows), 32.0, cam, camera.make_rng(cam))
    assert p1 == p2


def test_plan_for_scenes_variety_and_whips():
    cam = _cam(transition="whip_pan")
    plans = camera.plan_for_scenes([5.0, 5.0, 5.0], cam, random.Random(4))
    assert len(plans) == 3
    for a, b in zip(plans, plans[1:]):
        assert a[0]["move"] != b[0]["move"]
        assert b[0]["whip_in"] == -a[0]["whip_out"]


# ------------------------------------------------------------ camera_state ---
def _seg(move, start=0.0, end=4.0, **extra):
    seg = {"start": start, "end": end, "move": move, "dir": 1,
           "diag": ((0.7, 0.7), (-0.7, -0.7)),
           "phases": (0.1, 0.2, 0.3, 0.4)}
    seg.update(extra)
    return seg


def test_push_in_endpoints():
    cam = _cam(intensity=0.6)
    z0, _, _ = camera.camera_state(_seg("push_in"), 0.0, cam)
    z1, _, _ = camera.camera_state(_seg("push_in"), 4.0, cam)
    assert z0 == pytest.approx(1.0)
    assert z1 == pytest.approx(1.0 + 0.22 * 0.6)


def test_pull_out_reverses_push():
    cam = _cam(intensity=0.6)
    z0, _, _ = camera.camera_state(_seg("pull_out"), 0.0, cam)
    z1, _, _ = camera.camera_state(_seg("pull_out"), 4.0, cam)
    assert z0 == pytest.approx(1.0 + 0.22 * 0.6)
    assert z1 == pytest.approx(1.0)


def test_drift_pan_travels_edge_to_edge():
    cam = _cam(intensity=0.6)
    _, x0, _ = camera.camera_state(_seg("drift_pan"), 0.0, cam)
    _, x1, _ = camera.camera_state(_seg("drift_pan"), 4.0, cam)
    assert x0 == pytest.approx(-0.8)
    assert x1 == pytest.approx(0.8)


def test_ken_burns_moves_along_diagonal():
    cam = _cam(intensity=0.6)
    _, x0, y0 = camera.camera_state(_seg("ken_burns"), 0.0, cam)
    _, x1, y1 = camera.camera_state(_seg("ken_burns"), 4.0, cam)
    assert (x0, y0) == pytest.approx((0.7, 0.7))
    assert (x1, y1) == pytest.approx((-0.7, -0.7))


def test_punch_in_hits_fast_then_holds():
    cam = _cam(intensity=0.6)
    z_hit, _, _ = camera.camera_state(_seg("punch_in"), 4.0 * 0.18, cam)
    z_end, _, _ = camera.camera_state(_seg("punch_in"), 4.0, cam)
    assert z_hit == pytest.approx(1.0 + 0.25 * 0.6)
    assert z_end == pytest.approx(z_hit)


def test_static_is_neutral():
    z, x, y = camera.camera_state(_seg("static"), 2.0, _cam())
    assert (z, x, y) == (1.0, 0.0, 0.0)
    assert camera.shake_px(2.0, _seg("static"), _cam()) == (0.0, 0.0)


def test_max_zoom_caps_state():
    cam = _cam(intensity=1.0, max_zoom=1.1)
    z, _, _ = camera.camera_state(_seg("punch_in"), 4.0, cam)
    assert z == pytest.approx(1.1)


def test_shake_bounded_and_smooth():
    cam = _cam(shake=0.4)
    seg = _seg("handheld")
    amp = 10.0 * 0.4 * 2.5  # handheld boost
    prev = camera.shake_px(0.0, seg, cam)
    for k in range(1, 300):
        t = k / 30.0
        dx, dy = camera.shake_px(t, seg, cam)
        assert abs(dx) <= 1.5 * amp + 1e-6
        assert abs(dy) <= 1.5 * amp * 0.7 + 1e-6
        # smooth: bounded per-frame velocity, no white-noise jumps
        assert abs(dx - prev[0]) < amp
        prev = (dx, dy)


# ----------------------------------------------------------------- crop_box ---
def test_crop_box_aspect_and_cover():
    x0, y0, x1, y1 = camera.crop_box(1920, 1080, 1080, 1920, 1.0, 0.0, 0.0)
    assert (x1 - x0) / (y1 - y0) == pytest.approx(1080 / 1920)
    assert (y1 - y0) == pytest.approx(1080)  # full height at zoom 1


def test_crop_box_always_inside_source():
    rng = random.Random(9)
    for _ in range(500):
        sw, sh = rng.randint(320, 4096), rng.randint(320, 4096)
        z = rng.uniform(1.0, 1.3)
        ox, oy = rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5)
        shx, shy = rng.uniform(-30, 30), rng.uniform(-30, 30)
        x0, y0, x1, y1 = camera.crop_box(sw, sh, 1080, 1920, z, ox, oy, shx, shy)
        assert 0.0 <= x0 <= x1 <= sw + 1e-6
        assert 0.0 <= y0 <= y1 <= sh + 1e-6
        assert (x1 - x0) / (y1 - y0) == pytest.approx(1080 / 1920)


def test_crop_box_zoom_shrinks_window():
    a = camera.crop_box(1920, 1080, 1080, 1920, 1.0, 0.0, 0.0)
    b = camera.crop_box(1920, 1080, 1080, 1920, 1.2, 0.0, 0.0)
    assert (b[2] - b[0]) == pytest.approx((a[2] - a[0]) / 1.2)


# ------------------------------------------------------------- apply_camera ---
def _synthetic_clip(w=384, h=216, duration=1.0):
    np = pytest.importorskip("numpy")
    moviepy = pytest.importorskip("moviepy")
    grad = np.linspace(0, 255, w, dtype=np.uint8)
    frame = np.tile(grad[None, :, None], (h, 1, 3))

    return moviepy.VideoClip(lambda t: frame, duration=duration)


def test_apply_camera_smoke():
    np = pytest.importorskip("numpy")
    cam = _cam(moves=["drift_pan"], shake=0.0, transition="none", intensity=0.8)
    clip = _synthetic_clip()
    plan = camera.plan_segments([], 1.0, cam, random.Random(5))
    out = camera.apply_camera(clip, plan, cam, (108, 192), fps=30)

    f_start = out.get_frame(0.0)
    f_late = out.get_frame(0.9)
    f_end = out.get_frame(1.0 - 1e-9)  # boundary must not raise
    assert f_start.shape == (192, 108, 3)
    assert f_end.shape == (192, 108, 3)
    # source is static; any frame difference comes from the camera pan
    assert not np.array_equal(f_start, f_late)


def test_apply_camera_empty_plan_is_plain_cover_crop():
    cam = _cam()
    clip = _synthetic_clip()
    out = camera.apply_camera(clip, [], cam, (108, 192), fps=30)
    assert out.get_frame(0.5).shape == (192, 108, 3)
