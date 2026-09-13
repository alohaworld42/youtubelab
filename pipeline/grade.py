"""
grade.py — Cinematic color grade for the background layer, CPU-only.

One numpy pass per frame: per-channel tone LUT (S-curve contrast + warmth
tint), saturation around luma, radial vignette, optional film grain. Applied
to the background only — captions, avatars and motion graphics stay clean on
top. Config lives in the "grade" block; styles override it.
"""
import math

DEFAULTS = {
    "enabled": False,
    "saturation": 1.25,
    "contrast": 0.15,
    "warmth": 0.06,
    "vignette": 0.35,
    "grain": 0.0,
}


def get_grade_cfg(cfg):
    g = dict(DEFAULTS)
    g.update(cfg.get("grade") or {})
    return g


def grade_enabled(cfg):
    return bool((cfg.get("grade") or {}).get("enabled", False))


def build_luts(g):
    """(lut_r, lut_g, lut_b) uint8 arrays of 256 entries: S-curve contrast
    blended over identity, plus a warm/cool white-balance tint."""
    import numpy as np

    x = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    c = min(max(float(g.get("contrast", 0.15)), 0.0), 1.0)
    s_curve = x * x * (3.0 - 2.0 * x)
    tone = x + (s_curve - x) * c
    w = min(max(float(g.get("warmth", 0.06)), -0.3), 0.3)
    luts = []
    for gain in (1.0 + w, 1.0, 1.0 - w):
        luts.append(np.clip(tone * gain * 255.0 + 0.5, 0, 255).astype(np.uint8))
    return tuple(luts)


def vignette_mask(width, height, strength):
    """float32 (H, W, 1) multiplier: 1.0 at center, darker toward corners."""
    import numpy as np

    strength = min(max(float(strength), 0.0), 1.0)
    if strength <= 0.0:
        return None
    yy = (np.arange(height, dtype=np.float32) - height / 2.0) / (height / 2.0)
    xx = (np.arange(width, dtype=np.float32) - width / 2.0) / (width / 2.0)
    r = np.sqrt(xx[None, :] ** 2 + yy[:, None] ** 2) / math.sqrt(2.0)
    mask = 1.0 - 0.55 * strength * (r ** 2.2)
    return mask[..., None].astype(np.float32)


def grain_tiles(width, height, amount, count=8, seed=11):
    """Pre-rendered noise frames cycled per output frame (cheap add, no
    per-frame RNG so renders stay reproducible)."""
    import numpy as np

    amount = min(max(float(amount), 0.0), 1.0)
    if amount <= 0.0:
        return None
    rng = np.random.default_rng(seed)
    sigma = 12.0 * amount
    return [rng.normal(0.0, sigma, size=(height, width, 1)).astype(np.float32)
            for _ in range(count)]


def apply_grade(clip, cfg, fps):
    """Wrap clip so every frame gets the grade. No-op when disabled."""
    if not grade_enabled(cfg):
        return clip
    import numpy as np

    g = get_grade_cfg(cfg)
    lut_r, lut_g, lut_b = build_luts(g)
    sat = min(max(float(g.get("saturation", 1.25)), 0.0), 3.0)
    vig = vignette_mask(clip.w, clip.h, g.get("vignette", 0.35))
    tiles = grain_tiles(clip.w, clip.h, g.get("grain", 0.0))

    def fn(get_frame, t):
        arr = get_frame(t)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        out = np.stack(
            (np.take(lut_r, arr[..., 0]),
             np.take(lut_g, arr[..., 1]),
             np.take(lut_b, arr[..., 2])),
            axis=-1,
        )
        needs_float = sat != 1.0 or vig is not None or tiles is not None
        if not needs_float:
            return out
        out = out.astype(np.float32)
        if sat != 1.0:
            luma = out[..., 0:1] * 0.299 + out[..., 1:2] * 0.587 + out[..., 2:3] * 0.114
            out = luma + (out - luma) * sat
        if vig is not None:
            out *= vig
        if tiles is not None:
            out += tiles[int(t * fps) % len(tiles)]
        return np.clip(out, 0.0, 255.0).astype(np.uint8)

    return clip.transform(fn)
