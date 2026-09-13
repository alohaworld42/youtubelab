"""
camera.py — Virtual camera engine: Higgsfield-style cinematic moves on the CPU.

Emulates AI camera presets (push-in, pull-out, drift pan, ken burns, punch-in,
handheld) by animating a sub-pixel crop window over the source video, plus
whip-pan / crossfade transitions between beats. Pure Pillow + numpy per frame —
no GPU, no AI model.

The math core (plan_segments / camera_state / crop_box) is dependency-free and
unit-testable; apply_camera is the thin MoviePy 2.x glue.

Beats follow the spoken script lines: each line window gets one camera move,
long lines are split, short ones merged. A line may carry a "camera_hint"
(from the LLM script) that pins its move — e.g. punch_in on the payoff line.
"""
import bisect
import math
import random
import zlib

_DEFAULT_MOVES = ["push_in", "pull_out", "drift_pan", "ken_burns", "punch_in", "handheld"]
# "static" is only reachable via an explicit hint, never picked randomly.
VALID_MOVES = set(_DEFAULT_MOVES) | {"static"}

DEFAULTS = {
    "enabled": False,
    "intensity": 0.6,
    "moves": list(_DEFAULT_MOVES),
    "sync": "lines",
    "min_beat_seconds": 2.5,
    "max_beat_seconds": 7.0,
    "max_zoom": 1.3,
    "shake": 0.35,
    "transition": "whip_pan",
    "transition_seconds": 0.24,
    "apply_to_generated": False,
    "seed": None,
}


def get_camera_cfg(cfg):
    cam = dict(DEFAULTS)
    cam.update(cfg.get("camera") or {})
    return cam


def camera_enabled(cfg):
    # Missing block = disabled, so existing config.json files keep today's output.
    return bool((cfg.get("camera") or {}).get("enabled", False))


def make_rng(cam, key=None):
    seed = cam.get("seed")
    if seed is not None:
        return random.Random(seed)
    if key:
        return random.Random(zlib.crc32(str(key).encode("utf-8")))
    return random.Random()


# ------------------------------------------------------------------- easing ---
def ease(p):
    """Smoothstep: zero velocity at both ends."""
    p = min(max(p, 0.0), 1.0)
    return p * p * (3.0 - 2.0 * p)


def ease_out(p):
    p = min(max(p, 0.0), 1.0)
    return 1.0 - (1.0 - p) ** 3


# ----------------------------------------------------------------- planning ---
def _rand_diag(rng):
    x = rng.choice((-1.0, 1.0)) * 0.7
    y = rng.choice((-1.0, 1.0)) * 0.7
    return ((x, y), (-x, -y))


def plan_segments(segments, duration, cam, rng):
    """Cut [0, duration] into camera beats and assign a move to each.

    segments: script-line windows ({"start","end", optional "camera_hint"}).
    Returns [{"start","end","move","dir","diag","phases", opt "whip_out"/"whip_in"}]
    covering [0, duration] exactly.
    """
    duration = float(duration)
    min_b = float(cam.get("min_beat_seconds", 2.5))
    max_b = float(cam.get("max_beat_seconds", 7.0))

    beats = []
    for s in segments or []:
        st = float(s.get("start", 0.0))
        en = float(s.get("end", 0.0))
        if st >= duration or en <= st:
            continue
        beats.append({"start": st, "end": min(en, duration), "hint": s.get("camera_hint")})

    if not beats:
        n = max(1, round(duration / ((min_b + max_b) / 2.0)))
        step = duration / n
        beats = [{"start": k * step, "end": (k + 1) * step, "hint": None} for k in range(n)]

    # Force a gapless, monotonic cover of [0, duration] (silences attach to the
    # previous line's beat).
    beats.sort(key=lambda b: (b["start"], b["end"]))
    cover, cursor = [], 0.0
    for b in beats:
        en = min(b["end"], duration)
        if en <= cursor + 1e-6:
            continue
        cover.append({"start": cursor, "end": en, "hint": b["hint"]})
        cursor = en
    if not cover:
        cover = [{"start": 0.0, "end": duration, "hint": None}]
    cover[-1]["end"] = duration

    # Merge beats that are too short for a readable move.
    merged = []
    for b in cover:
        if merged and (b["end"] - b["start"]) < min_b:
            merged[-1]["end"] = b["end"]
            merged[-1]["hint"] = merged[-1]["hint"] or b["hint"]
        else:
            merged.append(b)
    if len(merged) > 1 and (merged[0]["end"] - merged[0]["start"]) < min_b:
        merged[1]["start"] = merged[0]["start"]
        merged[1]["hint"] = merged[1]["hint"] or merged[0]["hint"]
        merged.pop(0)

    # Split beats that are too long; the hint stays on the first slice.
    final = []
    for b in merged:
        length = b["end"] - b["start"]
        n = max(1, math.ceil(length / max_b - 1e-9))
        step = length / n
        for k in range(n):
            final.append({
                "start": b["start"] + k * step,
                "end": b["start"] + (k + 1) * step,
                "hint": b["hint"] if k == 0 else None,
            })
    final[-1]["end"] = duration

    moves = [m for m in cam.get("moves", _DEFAULT_MOVES) if m in VALID_MOVES] or list(_DEFAULT_MOVES)
    plan, prev = [], None
    for b in final:
        hint = b.get("hint")
        if hint in VALID_MOVES:
            mv = hint
        else:
            pool = [m for m in moves if m != prev] or moves
            mv = rng.choice(pool)
        plan.append({
            "start": b["start"],
            "end": b["end"],
            "move": mv,
            "dir": rng.choice((-1, 1)),
            "diag": _rand_diag(rng),
            "phases": tuple(rng.uniform(0.0, 2.0 * math.pi) for _ in range(4)),
        })
        prev = mv

    if str(cam.get("transition", "")) == "whip_pan":
        tau = float(cam.get("transition_seconds", 0.24))
        for a, b in zip(plan, plan[1:]):
            if (a["end"] - a["start"]) > tau and (b["end"] - b["start"]) > tau:
                d = rng.choice((-1, 1))
                a["whip_out"] = d
                b["whip_in"] = -d
    return plan


def plan_for_scenes(durations, cam, rng):
    """One single-beat plan per scene clip (scene mode renders each separately).

    Adjacent scenes get matching whip-out/whip-in flags so the hard cut between
    them reads as one fast pan.
    """
    moves = [m for m in cam.get("moves", _DEFAULT_MOVES) if m in VALID_MOVES] or list(_DEFAULT_MOVES)
    plans, prev = [], None
    for d in durations:
        pool = [m for m in moves if m != prev] or moves
        mv = rng.choice(pool)
        plans.append([{
            "start": 0.0,
            "end": float(d),
            "move": mv,
            "dir": rng.choice((-1, 1)),
            "diag": _rand_diag(rng),
            "phases": tuple(rng.uniform(0.0, 2.0 * math.pi) for _ in range(4)),
        }])
        prev = mv
    if str(cam.get("transition", "")) == "whip_pan":
        for a, b in zip(plans, plans[1:]):
            d = rng.choice((-1, 1))
            a[0]["whip_out"] = d
            b[0]["whip_in"] = -d
    return plans


# ------------------------------------------------------------- camera state ---
def camera_state(seg, t, cam):
    """(zoom, off_x, off_y) for time t inside segment seg.

    Offsets are normalized to the available pan margin: -1..1 edge to edge.
    """
    i = float(cam.get("intensity", 0.6))
    dur = max(seg["end"] - seg["start"], 1e-6)
    p = min(max((t - seg["start"]) / dur, 0.0), 1.0)
    move = seg.get("move", "static")
    z, ox, oy = 1.0, 0.0, 0.0

    if move == "push_in":
        z = 1.0 + 0.22 * i * ease(p)
    elif move == "pull_out":
        z = 1.0 + 0.22 * i * (1.0 - ease(p))
    elif move == "drift_pan":
        z = 1.0 + 0.10 * i
        ox = seg.get("dir", 1) * 0.8 * (2.0 * ease(p) - 1.0)
    elif move == "ken_burns":
        e = ease(p)
        z = 1.0 + (0.06 + 0.12 * e) * i
        (x0, y0), (x1, y1) = seg["diag"]
        ox = x0 + (x1 - x0) * e
        oy = y0 + (y1 - y0) * e
    elif move == "punch_in":
        z = 1.0 + 0.25 * i * ease_out(min(p / 0.18, 1.0))
    elif move == "handheld":
        z = 1.0 + 0.08 * i
    # "static": neutral

    return min(z, float(cam.get("max_zoom", 1.3))), ox, oy


def shake_px(t, seg, cam):
    """Smooth handheld shake (sum of sines, seeded phases) in output pixels."""
    amp = 10.0 * float(cam.get("shake", 0.35))
    move = seg.get("move", "static")
    if move == "static" or amp <= 0.0:
        return 0.0, 0.0
    if move == "handheld":
        amp *= 2.5
    p1, p2, p3, p4 = seg.get("phases", (0.0, 0.0, 0.0, 0.0))
    dx = amp * (math.sin(2 * math.pi * 0.9 * t + p1) + 0.5 * math.sin(2 * math.pi * 2.3 * t + p2))
    dy = 0.7 * amp * (math.sin(2 * math.pi * 1.1 * t + p3) + 0.5 * math.sin(2 * math.pi * 2.7 * t + p4))
    return dx, dy


def crop_box(src_w, src_h, out_w, out_h, zoom, off_x, off_y, shake_x=0.0, shake_y=0.0):
    """Float crop box (x0, y0, x1, y1) in source coords: cover-aspect rect,
    zoomed and panned, always fully inside the source. shake_x/shake_y are in
    output pixels and converted to source scale here."""
    src_w, src_h = float(src_w), float(src_h)
    zoom = max(1.0, float(zoom))
    if src_w / src_h >= out_w / out_h:
        base_h = src_h
        base_w = src_h * out_w / out_h
    else:
        base_w = src_w
        base_h = src_w * out_h / out_w
    cw, ch = base_w / zoom, base_h / zoom
    mx, my = (src_w - cw) / 2.0, (src_h - ch) / 2.0
    off_x = min(max(float(off_x), -1.0), 1.0)
    off_y = min(max(float(off_y), -1.0), 1.0)
    cx = src_w / 2.0 + off_x * mx + shake_x * (cw / out_w)
    cy = src_h / 2.0 + off_y * my + shake_y * (ch / out_h)
    x0 = min(max(cx - cw / 2.0, 0.0), src_w - cw)
    y0 = min(max(cy - ch / 2.0, 0.0), src_h - ch)
    return x0, y0, x0 + cw, y0 + ch


def whip_times(plan):
    """Boundary times (segment ends) that have a whip-pan transition."""
    return [s["end"] for s in plan if "whip_out" in s]


# -------------------------------------------------------------- MoviePy glue ---
def _hblur(arr, k):
    """Cheap directional blur for whip pans: average with ±k px shifted copies."""
    import numpy as np

    k = int(min(k, arr.shape[1] // 4))
    if k <= 0:
        return arr
    w = arr.shape[1]
    pad = np.pad(arr, ((0, 0), (k, k), (0, 0)), mode="edge").astype(np.uint16)
    out = (pad[:, :w] + pad[:, k:k + w] + pad[:, 2 * k:2 * k + w]) // 3
    return out.astype(arr.dtype)


def apply_camera(clip, plan, cam, out_size, fps):
    """Wrap clip in a VideoClip that renders the animated crop each frame.

    One pass per frame: numpy-slice the integer bounding box, then a single
    bilinear resize with a float `box` for sub-pixel motion (no stepping).
    Output is always exactly out_size, opaque — no mask, no x264 dimension
    issues. No frame cache: background frames are unique per t.
    """
    import numpy as np
    from moviepy import VideoClip
    from PIL import Image

    out_w, out_h = int(out_size[0]), int(out_size[1])
    duration = clip.duration
    starts = [s["start"] for s in plan]
    eps = 1.0 / max(float(fps), 1.0)
    tau = float(cam.get("transition_seconds", 0.24))
    half = max(tau / 2.0, 1e-6)
    # zoom impulses: short decaying punch at each graphic-reveal time
    imp_times = sorted(cam.get("_impulse_times") or [])
    imp_zoom = float(cam.get("impulse_zoom", 0.05))
    imp_dur = max(float(cam.get("impulse_seconds", 0.35)), 1e-3)
    # Blur length scales with frame time so slower fps doesn't strobe.
    kmax = min(40, max(8, int(900.0 / max(float(fps), 1.0))))

    def frame(t):
        tt = min(max(float(t), 0.0), max(duration - eps, 0.0))
        src = clip.get_frame(tt)
        if src.dtype != np.uint8:
            src = np.clip(src, 0, 255).astype(np.uint8)
        src_h, src_w = src.shape[0], src.shape[1]

        blur = 0.0
        if plan:
            idx = max(bisect.bisect_right(starts, tt) - 1, 0)
            seg = plan[idx]
            zoom, ox, oy = camera_state(seg, tt, cam)
            # Blend in from the previous segment's end state so move changes
            # never pop (whip transitions mask the cut themselves).
            if idx > 0 and "whip_in" not in seg:
                bl = min(0.3, (seg["end"] - seg["start"]) / 2.0)
                if bl > 0 and tt < seg["start"] + bl:
                    e = ease((tt - seg["start"]) / bl)
                    pz, pox, poy = camera_state(plan[idx - 1], plan[idx - 1]["end"], cam)
                    zoom = pz + (zoom - pz) * e
                    ox = pox + (ox - pox) * e
                    oy = poy + (oy - poy) * e
            d_out = seg.get("whip_out")
            d_in = seg.get("whip_in")
            if d_out and tt > seg["end"] - half:
                q = min(max((tt - (seg["end"] - half)) / half, 0.0), 1.0)
                ox = ox + (d_out - ox) * (q * q)
                blur = q
            elif d_in and tt < seg["start"] + half:
                q = min(max((tt - seg["start"]) / half, 0.0), 1.0)
                inv = 1.0 - q
                ox = ox + (d_in - ox) * (inv * inv)
                blur = inv
            sx, sy = shake_px(tt, seg, cam)
        else:
            zoom, ox, oy, sx, sy = 1.0, 0.0, 0.0, 0.0, 0.0

        if imp_times and imp_zoom > 0:
            j = bisect.bisect_right(imp_times, tt) - 1
            if j >= 0:
                q = (tt - imp_times[j]) / imp_dur
                if q < 1.0:
                    zoom = min(zoom + imp_zoom * (1.0 - ease(q)), float(cam.get("max_zoom", 1.3)))

        x0, y0, x1, y1 = crop_box(src_w, src_h, out_w, out_h, zoom, ox, oy, sx, sy)
        ix0, iy0 = int(x0), int(y0)
        ix1 = min(int(math.ceil(x1)), src_w)
        iy1 = min(int(math.ceil(y1)), src_h)
        view = np.ascontiguousarray(src[iy0:iy1, ix0:ix1])
        img = Image.fromarray(view).resize(
            (out_w, out_h), Image.BILINEAR,
            box=(x0 - ix0, y0 - iy0, x1 - ix0, y1 - iy0),
        )
        arr = np.asarray(img)
        if blur > 0.01:
            arr = _hblur(arr, int(round(kmax * blur)))
        return arr

    return VideoClip(frame, duration=duration).with_fps(fps)
