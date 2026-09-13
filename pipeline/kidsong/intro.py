"""
kidsong.intro — The fixed ZubiBop branding sting that opens every episode.

This is *branding*, so the one hard requirement is that it never drifts: the
intro is composed once, cached to `assets/intro/zubibop_intro.mp4`, and every
subsequent render reuses that exact file byte-for-byte. It is only re-rendered
when something that actually affects the picture changes — the logo file, the
timing/geometry constants, this module's `_VERSION`, or the `kidsong.intro`
config block — all folded into a content hash stored next to the mp4 in
`zubibop_intro.json`. `--rebuild` forces a fresh render.

Everything here is deterministic and CPU-only:

  * picture — Pillow composition driven by a damped-spring easing curve
    (`_spring`), rendered through a MoviePy 2.x `VideoClip(frame_function=...)`
    so the 1920x1080 frames are produced lazily instead of held in RAM.
  * audio — `synth_sting()` builds the sting from `MOTIF` with additive
    bar/glockenspiel synthesis (inharmonic `PARTIALS`, fast attack, natural
    exponential decay) plus a seeded exponential-noise reverb. No model, no
    GPU, no sample library.

The output is conformed to exactly the container the kidsong edit produces
(1920x1080 h264 High/yuv420p at `video.fps`, aac 44100 stereo) so
`pipeline.kidsong.edit.prepend_intro` can concatenate it with `-c copy` —
no re-encode of the finished episode, no A/V drift at the seam.

CLI:  python -m pipeline.kidsong.intro --rebuild
"""
import hashlib
import json
import math
import os
import subprocess
import tempfile

# --------------------------------------------------------------- versioning ---
# Bump whenever the *look* or *sound* changes in a way the config hash cannot
# see (i.e. any edit to the composition/synthesis code below). This is what
# forces a rebuild of an already-cached intro after a code change.
_VERSION = 3

# ------------------------------------------------------------------ palette ---
# Sampled directly out of the channel logo so the intro cannot drift away from
# the badge it frames.
SKY_TOP = (190, 235, 251)      # the logo's sky
SKY_BOTTOM = (168, 230, 201)   # the logo's grass
GLOW_COLOR = (255, 247, 226)   # warm cream halo (the TV character's face)
SHADOW_COLOR = (28, 74, 78)    # deep teal, not black — black reads as dirt

# The ZubiBop wordmark letters, in order. Used for sparkles and bokeh so every
# accent colour on screen is a colour the viewer already saw in the logo.
WORDMARK_COLORS = (
    (62, 193, 179),    # Z teal
    (245, 194, 60),    # u yellow
    (242, 112, 94),    # b coral
    (160, 124, 216),   # i purple
    (140, 198, 63),    # B green
    (245, 148, 60),    # o orange
    (74, 168, 224),    # p blue
)

# -------------------------------------------------------------- timing ------
# All in seconds from the first frame. The chord, the sparkle burst and the
# ring ripple all land on T_ACCENT, which is where the badge's spring settles —
# that sync is the whole reason the sting reads as "designed" rather than
# "music playing under a logo".
T_BADGE_IN = 0.12
T_ACCENT = 1.05
SPRING_OMEGA = 7.0      # rad/s — one visible overshoot, then rest
SPRING_DAMP = 5.5       # ~8.5% overshoot: a bounce, not a boing
BADGE_HEIGHT_RATIO = 0.74   # badge diameter as a fraction of frame height
BADGE_ENTRY_DROP = 46       # px it rises through on the way in
BADGE_ENTRY_TILT = -7.0     # degrees it unwinds from
BADGE_SCALE_START = 0.15
BREATH_HZ = 0.42            # gentle idle scale so the hold is never a freeze
BREATH_DEPTH = 0.012
RING_SECONDS = 0.95
RING_MAX_SCALE = 1.30
SPARKLE_COUNT = 14
SPARKLE_LIFE = 1.15

# --------------------------------------------------------------- audio ------
SAMPLE_RATE = 44100

# The motif. Edit this line to change the sting: (start_seconds, midi_note,
# velocity). A bright ascending C-major arpeggio.
MOTIF = (
    (0.10, 72, 0.80),   # C5
    (0.32, 76, 0.82),   # E5
    (0.54, 79, 0.88),   # G5
    (0.76, 84, 0.95),   # C6  — the peak of the run
)
# ...resolving onto a wide, soft C major exactly on the badge settle.
RESOLVE_CHORD = (T_ACCENT, (60, 64, 67, 72, 76), 0.85)

# Inharmonic partial ratios of a struck metal bar (glockenspiel/celesta):
# (frequency ratio, amplitude, decay multiplier). Upper partials die fast,
# which is what makes a mallet read as a mallet and not as an organ.
PARTIALS = (
    (1.00, 1.00, 1.0),
    (2.76, 0.42, 2.2),
    (5.40, 0.20, 3.4),
    (8.93, 0.09, 4.6),
)
NOTE_DECAY = 1.35        # seconds to ~1/e for the fundamental
NOTE_ATTACK = 0.003      # 3ms — a strike, not a swell
CHORD_DECAY = 2.10
FM_INDEX = 0.55          # a touch of shimmer on the attack
REVERB_SECONDS = 1.40
REVERB_PREDELAY = 0.025
REVERB_MIX = 0.30
REVERB_SEED = 20260720
SPARKLE_SEED = 8613
BOKEH_SEED = 4471

# Songs are mastered at roughly -14 LUFS (measured -13.9 on the reference
# episode). The sting is normalised to the same figure so the intro never
# blasts a toddler ahead of the song.
TARGET_LUFS = -14.0
TARGET_PEAK_DBFS = -1.5

_DEFAULT_LOGO = "channel_icon/6213b3cf-db3c-4a63-8321-dbae84bda1ab.png"
_DEFAULT_OUT = "assets/intro/zubibop_intro.mp4"


class IntroAssetMissing(Exception):
    """The logo (or another required source asset) is not on disk.

    Callers are expected to degrade gracefully: an episode without its intro
    is a worse episode, a crashed render is a lost one.
    """


# ============================================================== parameters ===
def intro_params(cfg):
    """Resolve the `kidsong.intro` config block into concrete render params."""
    from pipeline.config import abspath

    ks = (cfg.get("kidsong") or {})
    ic = (ks.get("intro") or {})
    out = (ks.get("output") or {})

    return {
        "enabled": bool(ic.get("enabled", True)),
        "logo": abspath(cfg, ic.get("logo") or _DEFAULT_LOGO),
        "path": abspath(cfg, ic.get("path") or _DEFAULT_OUT),
        "seconds": float(ic.get("seconds", 4.0)),
        "width": int(out.get("width", cfg.get("video", {}).get("width", 1920))),
        "height": int(out.get("height", cfg.get("video", {}).get("height", 1080))),
        "fps": int(cfg.get("video", {}).get("fps", 30)),
        "target_lufs": float(ic.get("target_lufs", TARGET_LUFS)),
        "version": _VERSION,
    }


def content_hash(params):
    """Hash of everything that can change the rendered intro.

    Covers the resolved params (minus the output path, which is *where* it is
    written, not *what* is written) plus the bytes of the logo itself, so
    swapping the logo file rebuilds even if its filename is unchanged.
    """
    h = hashlib.sha256()
    payload = {k: v for k, v in params.items() if k not in ("path", "enabled")}
    payload["motif"] = MOTIF
    payload["chord"] = RESOLVE_CHORD
    h.update(json.dumps(payload, sort_keys=True, default=str).encode("utf-8"))
    try:
        with open(params["logo"], "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    except OSError as exc:
        raise IntroAssetMissing(f"channel logo not readable: {params['logo']} ({exc})")
    return h.hexdigest()


def manifest_path(intro_mp4_path):
    return os.path.splitext(intro_mp4_path)[0] + ".json"


def is_current(params):
    """True when the cached mp4 on disk was built from exactly these inputs."""
    mp4 = params["path"]
    man = manifest_path(mp4)
    if not (os.path.exists(mp4) and os.path.exists(man)):
        return False
    try:
        with open(man, "r", encoding="utf-8") as f:
            stored = json.load(f)
    except (OSError, ValueError):
        return False
    try:
        return stored.get("hash") == content_hash(params)
    except IntroAssetMissing:
        return False


# ================================================================== easing ===
def _spring(elapsed, omega=SPRING_OMEGA, damp=SPRING_DAMP):
    """Damped spring settling 0 -> 1, overshooting once on the way.

    Deliberately not an ease-out curve: the overshoot is what separates a
    logo that *arrives* from a logo that merely fades up.
    """
    if elapsed <= 0.0:
        return 0.0
    return 1.0 - math.exp(-damp * elapsed) * math.cos(omega * elapsed)


def _smoothstep(x):
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def _ease_out_cubic(x):
    x = max(0.0, min(1.0, x))
    return 1.0 - (1.0 - x) ** 3


# ================================================================ picture ====
def _load_badge_master(logo_path):
    """The logo, cropped to its circular badge with an antialiased edge.

    The source is a square PNG with the badge inscribed and black corners (no
    alpha), so the circle is cut here rather than trusting the file.
    """
    import numpy as np
    from PIL import Image

    if not os.path.exists(logo_path):
        raise IntroAssetMissing(f"channel logo not found: {logo_path}")

    img = Image.open(logo_path).convert("RGB")
    w, h = img.size
    side = min(w, h)
    img = img.crop(((w - side) // 2, (h - side) // 2,
                    (w - side) // 2 + side, (h - side) // 2 + side))

    # Antialiased circular alpha, pulled 3px inside the badge so none of the
    # source's black corner pixels survive as a dark fringe.
    cy, cx = np.ogrid[:side, :side]
    center = (side - 1) / 2.0
    dist = np.sqrt((cx - center) ** 2 + (cy - center) ** 2)
    radius = center - 3.0
    alpha = np.clip((radius - dist) + 0.5, 0.0, 1.0)

    out = img.convert("RGBA")
    out.putalpha(Image.fromarray((alpha * 255).astype(np.uint8)))
    return out


def _background(width, height):
    """Vertical sky->grass gradient with a soft centre lift, as one RGBA image."""
    import numpy as np
    from PIL import Image

    ys = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    ramp = ys * ys * (3.0 - 2.0 * ys)  # smoothstep: no visible banding seam
    top = np.array(SKY_TOP, dtype=np.float32)
    bot = np.array(SKY_BOTTOM, dtype=np.float32)
    grad = top[None, None, :] * (1.0 - ramp[..., None]) + bot[None, None, :] * ramp[..., None]
    grad = np.repeat(grad, width, axis=1)

    # Very gentle radial lift behind where the badge will sit, so the frame
    # has a centre of gravity without a hard vignette.
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    r = np.sqrt(((xx - width / 2.0) / (width * 0.62)) ** 2
                + ((yy - height / 2.0) / (height * 0.62)) ** 2)
    lift = np.clip(1.0 - r, 0.0, 1.0) ** 2 * 0.16
    grad = grad + (np.array(GLOW_COLOR, dtype=np.float32)[None, None, :] - grad) * lift[..., None]

    rgb = np.clip(grad, 0, 255).astype(np.uint8)
    return Image.fromarray(rgb).convert("RGBA")


def _radial_sprite(size, color, gamma=2.2):
    """Soft radial blob, opaque at the centre, feathering to nothing."""
    import numpy as np
    from PIL import Image

    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    c = (size - 1) / 2.0
    r = np.sqrt((xx - c) ** 2 + (yy - c) ** 2) / c
    a = np.clip(1.0 - r, 0.0, 1.0) ** gamma
    rgba = np.zeros((size, size, 4), dtype=np.uint8)
    rgba[..., 0], rgba[..., 1], rgba[..., 2] = color
    rgba[..., 3] = (a * 255).astype(np.uint8)
    return Image.fromarray(rgba)


def _star_sprite(size=192):
    """A white four-point sparkle, soft-edged, ready to be tinted."""
    from PIL import Image, ImageDraw, ImageFilter

    ss = size * 2
    img = Image.new("L", (ss, ss), 0)
    d = ImageDraw.Draw(img)
    c = ss / 2.0
    outer, inner = c * 0.96, c * 0.16
    pts = []
    for k in range(8):
        ang = math.pi / 2.0 * (k / 2.0)
        rad = outer if k % 2 == 0 else inner
        pts.append((c + rad * math.cos(ang), c + rad * math.sin(ang)))
    d.polygon(pts, fill=255)
    d.ellipse((c - inner * 1.7, c - inner * 1.7, c + inner * 1.7, c + inner * 1.7), fill=255)
    img = img.filter(ImageFilter.GaussianBlur(ss * 0.012))
    img = img.resize((size, size), Image.LANCZOS)

    out = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    out.putalpha(img)
    return out


def _tint(sprite, color):
    from PIL import Image

    solid = Image.new("RGBA", sprite.size, (*color, 0))
    solid.putalpha(sprite.getchannel("A"))
    return solid


def _fade(sprite, alpha):
    """Scale a sprite's alpha channel by `alpha` (0..1)."""
    a = max(0.0, min(1.0, alpha))
    if a >= 0.999:
        return sprite
    out = sprite.copy()
    out.putalpha(sprite.getchannel("A").point(lambda v: int(v * a)))
    return out


def _paste(base, sprite, center):
    """Alpha-composite `sprite` onto opaque `base`, centred on `center`."""
    x = int(round(center[0] - sprite.width / 2.0))
    y = int(round(center[1] - sprite.height / 2.0))
    base.paste(sprite, (x, y), sprite)


def build_context(params):
    """Precompute every static layer once; the frame function only transforms.

    Raises IntroAssetMissing when the logo is absent.
    """
    import numpy as np
    from PIL import Image, ImageDraw, ImageFilter

    W, H = params["width"], params["height"]
    badge_d = int(round(BADGE_HEIGHT_RATIO * H))

    badge = _load_badge_master(params["logo"])

    # Drop shadow: the badge silhouette, blurred, in deep teal.
    shadow = Image.new("RGBA", (badge_d + 120, badge_d + 120), (*SHADOW_COLOR, 0))
    ImageDraw.Draw(shadow).ellipse(
        (60, 60, 60 + badge_d, 60 + badge_d), fill=(*SHADOW_COLOR, 110)
    )
    shadow = shadow.filter(ImageFilter.GaussianBlur(26))

    glow = _radial_sprite(int(badge_d * 1.95), GLOW_COLOR, gamma=2.6)
    star = _star_sprite(160)

    rng = np.random.default_rng(SPARKLE_SEED)
    sparkles = []
    for i in range(SPARKLE_COUNT):
        sparkles.append({
            "angle": (i / SPARKLE_COUNT) * 2 * math.pi + float(rng.uniform(-0.16, 0.16)),
            "delay": float(rng.uniform(0.0, 0.30)),
            "travel": float(rng.uniform(0.10, 0.30)) * badge_d,
            "size": float(rng.uniform(0.045, 0.085)) * badge_d,
            "spin": float(rng.uniform(-70, 70)),
            "color": WORDMARK_COLORS[i % len(WORDMARK_COLORS)],
        })

    brng = np.random.default_rng(BOKEH_SEED)
    bokeh_sprites, bokeh = [], []
    for i in range(10):
        color = WORDMARK_COLORS[i % len(WORDMARK_COLORS)]
        size = int(brng.uniform(0.10, 0.22) * H)
        bokeh_sprites.append(_radial_sprite(size, color, gamma=1.7))
        bokeh.append({
            "x": float(brng.uniform(0.04, 0.96)) * W,
            "y": float(brng.uniform(0.05, 1.05)) * H,
            "drift": float(brng.uniform(10.0, 26.0)),
            "alpha": float(brng.uniform(0.10, 0.19)),
            "phase": float(brng.uniform(0.0, 2 * math.pi)),
        })

    return {
        "W": W, "H": H,
        "seconds": params["seconds"],
        "badge_d": badge_d,
        "badge": badge,
        "bg": _background(W, H),
        "shadow": shadow,
        "glow": glow,
        "star": star,
        "sparkles": sparkles,
        "bokeh": bokeh,
        "bokeh_sprites": bokeh_sprites,
        "center": (W / 2.0, H / 2.0),
    }


def compose_frame(t, ctx):
    """Render the intro frame at time `t` as an HxWx3 uint8 array."""
    import numpy as np
    from PIL import Image, ImageDraw

    W, H = ctx["W"], ctx["H"]
    cx, cy = ctx["center"]
    frame = ctx["bg"].copy()

    # --- drifting bokeh: slow, low-contrast, never busy ---------------------
    for spec, sprite in zip(ctx["bokeh"], ctx["bokeh_sprites"]):
        y = spec["y"] - spec["drift"] * t
        sway = math.sin(spec["phase"] + t * 0.45) * 9.0
        _paste(frame, _fade(sprite, spec["alpha"]), (spec["x"] + sway, y))

    elapsed = t - T_BADGE_IN
    s = _spring(elapsed)

    # --- warm halo behind the badge, with one soft pulse on the accent ------
    pulse = 0.0
    if t >= T_ACCENT:
        pulse = 0.26 * math.exp(-3.2 * (t - T_ACCENT))
    glow_a = 0.34 * min(1.0, max(0.0, s)) + pulse
    if glow_a > 0.004:
        gs = ctx["glow"]
        gscale = 0.85 + 0.25 * min(1.2, max(0.0, s))
        gw = max(2, int(gs.width * gscale))
        _paste(frame, _fade(gs.resize((gw, gw), Image.BILINEAR), glow_a), (cx, cy))

    # --- single ring ripple leaving the badge on the accent -----------------
    if T_ACCENT <= t < T_ACCENT + RING_SECONDS:
        u = (t - T_ACCENT) / RING_SECONDS
        e = _ease_out_cubic(u)
        r = ctx["badge_d"] / 2.0 * (1.0 + (RING_MAX_SCALE - 1.0) * e)
        a = 0.5 * (1.0 - u) ** 2
        wdt = max(2, int(round(11 * (1.0 - e) + 2)))
        pad = wdt * 2 + 8
        box = int(r * 2 + pad) * 2  # drawn at 2x, downsampled for antialiasing
        layer = Image.new("RGBA", (box, box), (255, 255, 255, 0))
        d = ImageDraw.Draw(layer)
        c2 = box / 2.0
        d.ellipse((c2 - r * 2, c2 - r * 2, c2 + r * 2, c2 + r * 2),
                  outline=(255, 255, 255, int(a * 255)), width=wdt * 2)
        layer = layer.resize((box // 2, box // 2), Image.LANCZOS)
        _paste(frame, layer, (cx, cy))

    # --- the badge: spring scale, unwinding tilt, rising into place ---------
    if s > 0.001:
        breath = 1.0
        if elapsed > 0.9:
            ramp = _smoothstep((elapsed - 0.9) / 0.6)
            breath = 1.0 + BREATH_DEPTH * ramp * math.sin(2 * math.pi * BREATH_HZ * (elapsed - 0.9))
        scale = (BADGE_SCALE_START + (1.0 - BADGE_SCALE_START) * s) * breath
        scale = max(0.02, scale)
        d = max(8, int(round(ctx["badge_d"] * scale)))
        by = cy + BADGE_ENTRY_DROP * (1.0 - min(1.0, s))
        alpha = _smoothstep(elapsed / 0.22)
        tilt = BADGE_ENTRY_TILT * (1.0 - s)

        sh = ctx["shadow"]
        shd = max(8, int(round(sh.width * scale)))
        _paste(frame, _fade(sh.resize((shd, shd), Image.BILINEAR), alpha * 0.9),
               (cx, by + ctx["badge_d"] * 0.030 * scale))

        badge = ctx["badge"].resize((d, d), Image.LANCZOS)
        if abs(tilt) > 0.05:
            badge = badge.rotate(tilt, resample=Image.BICUBIC, expand=True)
        _paste(frame, _fade(badge, alpha), (cx, by))

    # --- sparkles popping off the rim on the accent -------------------------
    for sp in ctx["sparkles"]:
        u = (t - (T_ACCENT + sp["delay"])) / SPARKLE_LIFE
        if not (0.0 < u < 1.0):
            continue
        e = _ease_out_cubic(u)
        r = ctx["badge_d"] / 2.0 * 0.93 + sp["travel"] * e
        # fast bloom, slow fade — a twinkle, never a flash
        a = math.sin(math.pi * min(1.0, u ** 0.55)) * 0.9
        size = max(4, int(round(sp["size"] * (0.45 + 0.75 * math.sin(math.pi * u)))))
        star = _tint(ctx["star"], sp["color"]).resize((size, size), Image.LANCZOS)
        star = star.rotate(sp["spin"] * u, resample=Image.BICUBIC)
        _paste(frame, _fade(star, a),
               (cx + r * math.cos(sp["angle"]), cy + r * math.sin(sp["angle"])))

    return np.asarray(frame.convert("RGB"), dtype=np.uint8)


# ================================================================== audio ====
def _midi_hz(note):
    return 440.0 * (2.0 ** ((note - 69) / 12.0))


def _note(freq, velocity, decay, sr, length):
    """One struck-bar note: inharmonic partials, 3ms attack, natural decay."""
    import numpy as np

    n = int(length * sr)
    t = np.arange(n, dtype=np.float64) / sr
    attack = 1.0 - np.exp(-t / NOTE_ATTACK)
    out = np.zeros(n, dtype=np.float64)

    # A little FM on the attack gives the strike its metallic shimmer without
    # the partials themselves having to be loud enough to sound harsh.
    fm = FM_INDEX * np.exp(-t * 9.0) * np.sin(2 * np.pi * freq * 3.5 * t)
    for ratio, amp, dmul in PARTIALS:
        env = np.exp(-t * dmul / decay)
        out += amp * env * np.sin(2 * np.pi * freq * ratio * t + fm)

    # Mallet contact: a tiny seeded noise transient, entirely gone in ~12ms.
    rng = np.random.default_rng(int(freq) * 7 + 13)
    click_n = int(0.012 * sr)
    click = rng.standard_normal(click_n) * np.exp(-np.arange(click_n) / (0.0022 * sr))
    out[:click_n] += click * 0.06

    out *= attack * velocity
    return out / (len(PARTIALS) * 0.55)


def _reverb_ir(sr, seconds=REVERB_SECONDS, seed=REVERB_SEED):
    """Seeded exponential-noise impulse response, smoothed so it is a room
    rather than a hiss. Two decorrelated channels give natural width."""
    import numpy as np

    n = int(seconds * sr)
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64) / sr
    env = np.exp(-t * (6.0 / seconds))
    ir = rng.standard_normal((n, 2)) * env[:, None]

    # Cheap low-pass (moving average) — a bright noise tail sounds like static.
    k = 9
    kernel = np.ones(k) / k
    for ch in range(2):
        ir[:, ch] = np.convolve(ir[:, ch], kernel, mode="same")

    pre = int(REVERB_PREDELAY * sr)
    ir = np.vstack([np.zeros((pre, 2)), ir])
    ir /= np.max(np.abs(ir)) or 1.0
    return ir


def synth_sting(seconds=4.0, sr=SAMPLE_RATE, motif=MOTIF, chord=RESOLVE_CHORD):
    """Render the branding sting to a float32 (n, 2) stereo array in [-1, 1].

    Deterministic: identical inputs always produce identical samples. Peak
    normalised only — loudness matching to the songs happens in `build_intro`,
    which needs ffmpeg to measure LUFS.
    """
    import numpy as np

    n = int(round(seconds * sr))
    dry = np.zeros(n, dtype=np.float64)

    def place(start, samples):
        i0 = int(round(start * sr))
        if i0 >= n:
            return
        i1 = min(n, i0 + len(samples))
        dry[i0:i1] += samples[: i1 - i0]

    for start, note, vel in motif:
        place(start, _note(_midi_hz(note), vel, NOTE_DECAY, sr, min(seconds, 2.2)))

    if chord:
        c_start, c_notes, c_vel = chord
        # Slightly softer per voice so a 5-note chord does not out-shout the run.
        for k, note in enumerate(c_notes):
            spread = k * 0.006  # a hair of strum, so it breathes
            place(c_start + spread,
                  _note(_midi_hz(note), c_vel / (len(c_notes) ** 0.55),
                        CHORD_DECAY, sr, min(seconds, 2.8)))

    try:
        from scipy.signal import fftconvolve as _conv
    except Exception:  # pragma: no cover - scipy ships with librosa here
        _conv = None

    ir = _reverb_ir(sr)
    wet = np.zeros((n, 2), dtype=np.float64)
    for ch in range(2):
        if _conv is not None:
            w = _conv(dry, ir[:, ch])[:n]
        else:
            w = np.convolve(dry, ir[:, ch])[:n]
        wet[:, ch] = w
    peak = np.max(np.abs(wet)) or 1.0
    wet = wet / peak * (np.max(np.abs(dry)) or 1.0)

    out = np.stack([dry, dry], axis=1) * (1.0 - REVERB_MIX) + wet * REVERB_MIX

    # Hard guarantee that the sting is finished before the song starts: the
    # last 12% of the clip is ramped to true silence.
    tail = int(n * 0.12)
    if tail > 1:
        out[n - tail:] *= np.linspace(1.0, 0.0, tail)[:, None] ** 2

    out /= (np.max(np.abs(out)) or 1.0)
    return (out * 0.98).astype(np.float32)


# ============================================================== loudness =====
def measure_lufs(path):
    """Integrated loudness of `path` via ffmpeg's ebur128 filter, or None."""
    try:
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-nostats", "-i", path,
             "-af", "ebur128", "-f", "null", "-"],
            capture_output=True, text=True,
        )
    except OSError:
        return None
    text = proc.stderr or ""
    marker = "Integrated loudness:"
    if marker not in text:
        return None
    for line in text.split(marker, 1)[1].splitlines():
        line = line.strip()
        if line.startswith("I:"):
            try:
                return float(line.split()[1])
            except (IndexError, ValueError):
                return None
    return None


def _normalized_sting(seconds, sr, target_lufs, log):
    """Sting, gain-matched to `target_lufs` with a true-peak safety ceiling.

    A single measure-then-scale pass (not ffmpeg's two-pass loudnorm, whose
    dynamic mode is unreliable on clips this short and would make the cached
    intro depend on filter-version behaviour).
    """
    import numpy as np

    samples = synth_sting(seconds=seconds, sr=sr)

    tmp = os.path.join(tempfile.gettempdir(), f"zubibop_sting_{os.getpid()}.wav")
    try:
        _write_wav(tmp, samples, sr)
        measured = measure_lufs(tmp)
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass

    if measured is None:
        log("  loudness measurement unavailable — keeping peak-normalised sting")
        return samples * (10 ** (TARGET_PEAK_DBFS / 20.0))

    gain = 10 ** ((target_lufs - measured) / 20.0)
    ceiling = 10 ** (TARGET_PEAK_DBFS / 20.0)
    peak = float(np.max(np.abs(samples))) or 1.0
    if peak * gain > ceiling:
        gain = ceiling / peak
        log(f"  sting gain limited by true peak ({TARGET_PEAK_DBFS} dBFS)")
    log(f"  sting measured {measured:.1f} LUFS -> target {target_lufs:.1f} LUFS "
        f"(gain {20 * math.log10(gain):+.1f} dB)")
    return (samples * gain).astype(np.float32)


def _write_wav(path, samples, sr):
    import wave

    import numpy as np

    pcm = np.clip(samples, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    with wave.open(path, "w") as wf:
        wf.setnchannels(pcm.shape[1] if pcm.ndim > 1 else 1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


# =============================================================== rendering ===
def _conform(src, dst, params):
    """Re-mux/encode to exactly the kidsong episode container.

    Matching pix_fmt, profile, frame rate, sample rate and channel layout is
    what lets `prepend_intro` concatenate with `-c copy`; a mismatch here shows
    up as a re-encode (slow, lossy) or a stream discontinuity at the seam.
    """
    # Through a .part sibling: `dst` is the CACHED intro, and `is_current`
    # trusts it whenever a matching manifest sits beside it. A rebuild
    # (rebuild=True, or any forced rerun with unchanged params) that died
    # mid-encode left a truncated mp4 that the manifest then vouched for, and
    # it would be prepended to every episode from then on.
    from pipeline.atomicio import atomic_path

    with atomic_path(dst) as tmp_dst:
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", src,
             "-c:v", "libx264", "-profile:v", "high", "-pix_fmt", "yuv420p",
             "-preset", "medium", "-crf", "18",
             "-r", str(params["fps"]), "-vsync", "cfr",
             "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
             "-movflags", "+faststart", tmp_dst],
            check=True,
        )


def build_intro(cfg, rebuild=False, on_progress=None):
    """Render (or reuse) the cached branding intro; returns its path.

    Raises IntroAssetMissing if the logo is gone — callers that render episodes
    must catch this and continue without an intro.
    """
    from moviepy import AudioArrayClip, VideoClip

    def log(msg):
        if on_progress:
            on_progress(msg)
        else:
            print(msg)

    params = intro_params(cfg)
    out_path = params["path"]

    if not rebuild and is_current(params):
        log(f"Intro cache hit: {out_path}")
        return out_path

    digest = content_hash(params)  # raises IntroAssetMissing if the logo is gone
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    log(f"Building branding intro ({params['seconds']:.1f}s @ "
        f"{params['width']}x{params['height']} {params['fps']}fps)…")

    ctx = build_context(params)
    audio = _normalized_sting(params["seconds"], SAMPLE_RATE, params["target_lufs"], log)

    video = VideoClip(frame_function=lambda t: compose_frame(t, ctx),
                      duration=params["seconds"])
    aclip = AudioArrayClip(audio.astype("float64"), fps=SAMPLE_RATE)
    video = video.with_audio(aclip).with_duration(params["seconds"])

    raw = out_path + ".raw.mp4"
    try:
        log("  compositing frames…")
        video.write_videofile(
            raw, fps=params["fps"], codec="libx264", audio_codec="aac",
            preset="medium", audio_fps=SAMPLE_RATE,
            threads=os.cpu_count() or 4, logger=None,
        )
        log("  conforming container to the episode format…")
        _conform(raw, out_path, params)
    finally:
        for clip in (video, aclip):
            try:
                clip.close()
            except Exception:
                pass
        try:
            os.remove(raw)
        except OSError:
            pass

    # Manifest last and atomically: it is the token that makes `is_current`
    # trust the mp4 beside it, so it must never be half-written either.
    from pipeline.atomicio import atomic_write_json

    atomic_write_json(
        manifest_path(out_path),
        {"hash": digest, "params": params, "version": _VERSION},
        indent=2,
    )

    log(f"Wrote {out_path}")
    return out_path


def ensure_intro(cfg, on_progress=None):
    """Path to a ready-to-use intro, or None when it is off/unavailable.

    Never raises: a missing logo or a failed build degrades to "no intro", it
    does not take an episode render down with it.
    """
    params = intro_params(cfg)
    if not params["enabled"]:
        return None
    try:
        return build_intro(cfg, rebuild=False, on_progress=on_progress)
    except IntroAssetMissing as exc:
        if on_progress:
            on_progress(f"Intro skipped: {exc}")
        return None
    except Exception as exc:  # pragma: no cover - defensive
        if on_progress:
            on_progress(f"Intro skipped (build failed: {exc})")
        return None


# ===================================================================== cli ===
def main(argv=None):
    import argparse

    from pipeline.config import load_config

    ap = argparse.ArgumentParser(
        prog="python -m pipeline.kidsong.intro",
        description="Build the fixed ZubiBop branding intro used before every episode.",
    )
    ap.add_argument("--rebuild", action="store_true",
                    help="re-render even when the cache is up to date")
    args = ap.parse_args(argv)

    cfg = load_config()
    try:
        path = build_intro(cfg, rebuild=args.rebuild)
    except IntroAssetMissing as exc:
        print(f"ERROR: {exc}")
        return 1
    print(f"\nIntro: {os.path.abspath(path)}")
    print(f"Manifest: {os.path.abspath(manifest_path(path))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
