"""
motion.py — Native animated motion-graphics for the video pipeline.

This is the local, no-Docker answer to the Remotion-style "graphics synced to
the transcript" look: small animated components (stat counter, statement card,
comparison bars) drawn per-frame with Pillow and composited by MoviePy, each
timed to a line's spoken start/end.

Each component animates in (slide up + scale + fade), holds, then fades out.
Numbers count up. No GPU, no external service — just Pillow + numpy + MoviePy.

Public API:
    build_motion_clips(cfg, graphics, video_size, fps) -> [MoviePy clips]

`graphics` is a list of spec dicts, each with a "type" and timing:
    {"type": "stat",    "value": "3", "label": "hearts",   "start": t, "end": t}
    {"type": "card",    "text": "territorial marking",      "start": t, "end": t}
    {"type": "compare", "a_label": "cats", "a_value": 90,
                        "b_label": "dogs", "b_value": 40,   "start": t, "end": t}
"""
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy import VideoClip

_FONT_CACHE = {}


# --------------------------------------------------------------- primitives ---
def _font(path, size):
    key = (path, size)
    f = _FONT_CACHE.get(key)
    if f is None:
        try:
            f = ImageFont.truetype(path, size)
        except Exception:
            f = ImageFont.load_default()
        _FONT_CACHE[key] = f
    return f


def _ease_out(p):
    """Cubic ease-out: fast then settling. p in [0,1]."""
    p = min(1.0, max(0.0, p))
    return 1.0 - (1.0 - p) ** 3


def _timeline(t, dur):
    """Return (appear, alpha, count) animation factors for local time t.

    appear: 0->1 slide/scale-in factor over the intro
    alpha:  fades in over the intro and out over the outro
    count:  0->1 progress used to roll numbers up over the first ~55%
    """
    intro = min(0.45, dur * 0.35)
    outro = min(0.35, dur * 0.25)
    appear = _ease_out(t / intro) if intro > 0 else 1.0
    fade_out = _ease_out((dur - t) / outro) if outro > 0 else 1.0
    alpha = min(appear, fade_out)
    count = _ease_out(t / (dur * 0.55)) if dur > 0 else 1.0
    return appear, alpha, count


def _measure(draw, text, font):
    b = draw.textbbox((0, 0), text, font=font)
    return b[2] - b[0], b[3] - b[1]


def _wrap(draw, text, font, max_w):
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if _measure(draw, trial, font)[0] <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def _rounded(size, radius, fill):
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    ImageDraw.Draw(img).rounded_rectangle([0, 0, size[0], size[1]], radius=radius, fill=fill)
    return img


def _split_number(value):
    """('90%') -> (90, '90%'); ('3') -> (3, '3'); ('x') -> (None, 'x')."""
    s = str(value).strip()
    m = re.search(r"(\d[\d,]*)", s)
    if not m:
        return None, s
    return int(m.group(1).replace(",", "")), s


def _fmt_count(target_int, target_str, count):
    """Interpolate target_int by `count` and re-insert into target_str's shape."""
    cur = int(round(target_int * count))
    return re.sub(r"\d[\d,]*", f"{cur:,}" if target_int >= 1000 else str(cur), target_str, count=1)


def _compose(canvas, content, scale, slide_y, alpha):
    """Scale + center + vertically offset `content` on a fixed-size transparent
    canvas, then apply a global alpha multiplier."""
    out = Image.new("RGBA", canvas, (0, 0, 0, 0))
    if scale != 1.0:
        content = content.resize(
            (max(1, int(content.width * scale)), max(1, int(content.height * scale))),
            Image.LANCZOS,
        )
    x = (canvas[0] - content.width) // 2
    y = (canvas[1] - content.height) // 2 + int(slide_y)
    out.alpha_composite(content, (x, max(0, y)))
    if alpha < 1.0:
        r, g, b, a = out.split()
        a = a.point(lambda v: int(v * alpha))
        out = Image.merge("RGBA", (r, g, b, a))
    return out


# ---------------------------------------------------------------- components ---
class _StatCounter:
    """Big number that rolls up from 0, with a caption underneath."""

    def __init__(self, spec, cfg, vw):
        m = cfg["motion"]
        self.num_font = _font(m["font_path"], int(m.get("stat_font_size", 200)))
        self.label_font = _font(m["font_path"], int(m.get("label_font_size", 72)))
        self.accent = tuple(spec.get("color") or m.get("accent_color", [255, 214, 40]))
        self.num_color = tuple(m.get("stat_text_color", [18, 16, 8]))
        self.label_color = tuple(m.get("label_text_color", [255, 255, 255]))
        self.radius = int(m.get("radius", 44))
        self.target_int, self.target_str = _split_number(spec.get("value", "0"))
        if self.target_int is None:
            self.target_int, self.target_str = 0, str(spec.get("value", ""))
        self.label = str(spec.get("label", "")).upper()

        probe = ImageDraw.Draw(Image.new("RGBA", (4, 4)))
        nw, nh = _measure(probe, self.target_str or "0", self.num_font)
        lw, lh = _measure(probe, self.label, self.label_font) if self.label else (0, 0)
        pad = 60
        self.card_w = max(nw, lw) + pad * 2
        self.card_h = nh + (lh + 24 if self.label else 0) + pad * 2
        self.canvas = (int(self.card_w + 80), int(self.card_h + 80))

    def render(self, t, dur):
        appear, alpha, count = _timeline(t, dur)
        card = _rounded((int(self.card_w), int(self.card_h)), self.radius, self.accent + (255,))
        d = ImageDraw.Draw(card)
        num_str = _fmt_count(self.target_int, self.target_str, count) if self.target_int else self.target_str
        nb = d.textbbox((0, 0), num_str, font=self.num_font)
        nw, nh = nb[2] - nb[0], nb[3] - nb[1]
        top = 60
        d.text(((self.card_w - nw) / 2 - nb[0], top - nb[1]), num_str, font=self.num_font, fill=self.num_color + (255,))
        if self.label:
            lb = d.textbbox((0, 0), self.label, font=self.label_font)
            lw = lb[2] - lb[0]
            d.text(((self.card_w - lw) / 2 - lb[0], top + nh + 24 - lb[1]), self.label, font=self.label_font, fill=self.label_color + (255,))
        return _compose(self.canvas, card, 0.85 + 0.15 * appear, (1 - appear) * 46, alpha)


class _StatementCard:
    """Rounded card with a short key phrase that pops in."""

    def __init__(self, spec, cfg, vw):
        m = cfg["motion"]
        self.font = _font(m["font_path"], int(m.get("card_font_size", 88)))
        self.bg = tuple(spec.get("color") or m.get("card_color", [22, 20, 30]))
        self.fg = tuple(m.get("card_text_color", [255, 255, 255]))
        self.bar = tuple(m.get("accent_color", [255, 214, 40]))
        self.radius = int(m.get("radius", 44))
        text = str(spec.get("text", "")).strip()
        if m.get("uppercase", True):
            text = text.upper()
        max_text_w = int(vw * 0.82) - 120
        probe = ImageDraw.Draw(Image.new("RGBA", (4, 4)))
        self.lines = _wrap(probe, text, self.font, max_text_w) or [text]
        self.line_h = _measure(probe, "Ag", self.font)[1] + 18
        widest = max(_measure(probe, ln, self.font)[0] for ln in self.lines)
        pad = 56
        self.card_w = widest + pad * 2 + 24
        self.card_h = self.line_h * len(self.lines) + pad * 2
        self.canvas = (int(self.card_w + 80), int(self.card_h + 80))

    def render(self, t, dur):
        appear, alpha, _ = _timeline(t, dur)
        card = _rounded((int(self.card_w), int(self.card_h)), self.radius, self.bg + (235,))
        d = ImageDraw.Draw(card)
        d.rounded_rectangle([28, 28, 44, self.card_h - 28], radius=8, fill=self.bar + (255,))
        y = 56
        for ln in self.lines:
            lb = d.textbbox((0, 0), ln, font=self.font)
            d.text((70 - lb[0], y - lb[1]), ln, font=self.font, fill=self.fg + (255,))
            y += self.line_h
        return _compose(self.canvas, card, 0.82 + 0.18 * appear, (1 - appear) * 40, alpha)


class _CompareBars:
    """Two labeled horizontal bars that grow, values counting up."""

    def __init__(self, spec, cfg, vw):
        m = cfg["motion"]
        self.label_font = _font(m["font_path"], int(m.get("label_font_size", 64)))
        self.val_font = _font(m["font_path"], int(m.get("label_font_size", 64)))
        self.bg = tuple(m.get("card_color", [22, 20, 30]))
        self.fg = tuple(m.get("card_text_color", [255, 255, 255]))
        self.radius = int(m.get("radius", 44))
        self.a_label = str(spec.get("a_label", "A")).upper()
        self.b_label = str(spec.get("b_label", "B")).upper()
        self.a_val, _ = _split_number(spec.get("a_value", 0))
        self.b_val, _ = _split_number(spec.get("b_value", 0))
        self.a_val = self.a_val or 0
        self.b_val = self.b_val or 0
        self.a_color = tuple(m.get("accent_color", [255, 214, 40]))
        self.b_color = tuple(m.get("accent_b_color", [83, 196, 255]))
        self.card_w = int(vw * 0.82)
        self.card_h = 420
        self.bar_max = self.card_w - 340
        self.peak = max(self.a_val, self.b_val, 1)
        self.canvas = (int(self.card_w + 80), int(self.card_h + 80))

    def _row(self, d, y, label, val, color, count):
        d.text((56, y), label, font=self.label_font, fill=self.fg + (255,))
        track_x, track_y, track_h = 56, y + 78, 46
        d.rounded_rectangle([track_x, track_y, track_x + self.bar_max, track_y + track_h], radius=23, fill=(255, 255, 255, 40))
        w = int(self.bar_max * (val / self.peak) * count)
        if w > 4:
            d.rounded_rectangle([track_x, track_y, track_x + w, track_y + track_h], radius=23, fill=color + (255,))
        cur = int(round(val * count))
        d.text((track_x + self.bar_max + 24, y + 66), str(cur), font=self.val_font, fill=color + (255,))

    def render(self, t, dur):
        appear, alpha, count = _timeline(t, dur)
        card = _rounded((int(self.card_w), int(self.card_h)), self.radius, self.bg + (235,))
        d = ImageDraw.Draw(card)
        self._row(d, 56, self.a_label, self.a_val, self.a_color, count)
        self._row(d, 240, self.b_label, self.b_val, self.b_color, count)
        return _compose(self.canvas, card, 0.86 + 0.14 * appear, (1 - appear) * 44, alpha)


class _TitleCard:
    """Intro hook: the video title stamped on screen for the first beat —
    slams in oversized, settles, pops out. Styled from the "hook" config block."""

    def __init__(self, spec, cfg, vw):
        m = cfg["motion"]
        h = cfg.get("hook") or {}
        self.font = _font(h.get("font_path") or m["font_path"], int(h.get("font_size", 116)))
        self.bg = tuple(h.get("color") or m.get("accent_color", [255, 214, 40]))
        self.fg = tuple(h.get("text_color") or m.get("stat_text_color", [18, 16, 8]))
        self.radius = int(m.get("radius", 44))
        # fonts can't render emoji; keep printable ASCII only
        text = re.sub(r"[^\x20-\x7E]", "", str(spec.get("text", ""))).strip()
        if h.get("uppercase", True):
            text = text.upper()
        max_text_w = int(vw * 0.86) - 112
        probe = ImageDraw.Draw(Image.new("RGBA", (4, 4)))
        self.lines = _wrap(probe, text, self.font, max_text_w) or [text]
        self.line_h = _measure(probe, "Ag", self.font)[1] + 20
        widest = max(_measure(probe, ln, self.font)[0] for ln in self.lines)
        pad = 56
        self.card_w = widest + pad * 2
        self.card_h = self.line_h * len(self.lines) + pad * 2
        # canvas must fit the 1.3x oversized stamp-in frame
        self.canvas = (int(self.card_w * 1.32) + 8, int(self.card_h * 1.32) + 8)

    def render(self, t, dur):
        # Snappier than _timeline: hard stamp in (~0.22s), quick pop out.
        appear = _ease_out(t / 0.22)
        alpha = min(appear, _ease_out((dur - t) / 0.22))
        card = _rounded((int(self.card_w), int(self.card_h)), self.radius, self.bg + (255,))
        d = ImageDraw.Draw(card)
        y = 56
        for ln in self.lines:
            lb = d.textbbox((0, 0), ln, font=self.font)
            lw = lb[2] - lb[0]
            d.text(((self.card_w - lw) / 2 - lb[0], y - lb[1]), ln, font=self.font, fill=self.fg + (255,))
            y += self.line_h
        # stamp: starts 30% oversized, shrinks onto the screen
        return _compose(self.canvas, card, 1.0 + 0.3 * (1.0 - appear), 0, alpha)


_COMPONENTS = {"stat": _StatCounter, "card": _StatementCard, "compare": _CompareBars, "title": _TitleCard}


def _make_component(spec, cfg, vw):
    cls = _COMPONENTS.get(spec.get("type"))
    if cls is None:
        return None
    try:
        return cls(spec, cfg, vw)
    except Exception as e:
        print(f"[motion] skipped {spec.get('type')} graphic: {e}")
        return None


# ------------------------------------------------------------- moviepy glue ---
def _to_alpha_clip(render_fn, duration, fps):
    """Wrap a render(local_t)->RGBA-image function in a MoviePy clip with alpha."""
    cache = {}

    def _img(t):
        key = round(float(t), 4)
        img = cache.get(key)
        if img is None:
            img = render_fn(min(float(t), duration))
            if len(cache) > 4:
                cache.clear()
            cache[key] = img
        return img

    def make_rgb(t):
        return np.asarray(_img(t).convert("RGB"))

    def make_alpha(t):
        return np.asarray(_img(t).split()[-1]).astype("float64") / 255.0

    clip = VideoClip(make_rgb, duration=duration)
    mask = VideoClip(make_alpha, duration=duration, is_mask=True)
    return clip.with_mask(mask).with_fps(fps)


def build_motion_clips(cfg, graphics, video_size, fps):
    """Turn timed graphic specs into positioned, animated MoviePy clips."""
    m = cfg.get("motion") or {}
    if not m.get("enabled", True) or not graphics:
        return []
    vw, vh = video_size
    clips = []
    for g in graphics:
        comp = _make_component(g, cfg, vw)
        if comp is None:
            continue
        # a spec may pin its own vertical spot (the title hook sits higher)
        y = int(vh * float(g.get("y_ratio", m.get("vertical_position", 0.34))))
        dur = max(0.4, float(g["end"]) - float(g["start"]))
        clip = _to_alpha_clip(lambda t, c=comp, d=dur: c.render(t, d), dur, fps)
        clip = clip.with_start(float(g["start"])).with_position(("center", y - comp.canvas[1] // 2))
        clips.append(clip)
    return clips


def build_progress_clip(cfg, duration, video_size, fps):
    """Thin retention progress bar that fills over the whole video. Returns a
    positioned MoviePy clip, or None when disabled."""
    p = cfg.get("progress_bar") or {}
    if not p.get("enabled", True) or duration <= 0:
        return None
    W, H = video_size
    h = max(2, int(p.get("height", 12)))
    color = tuple(p.get("color") or (cfg.get("motion") or {}).get("accent_color", [255, 214, 40]))
    fill_a = min(max(float(p.get("opacity", 0.85)), 0.0), 1.0)
    track_a = min(max(float(p.get("track_opacity", 0.18)), 0.0), 1.0)
    at_top = str(p.get("position", "bottom")) == "top"

    bar = np.zeros((h, W, 3), dtype=np.uint8)
    bar[:] = color

    def make_alpha(t):
        a = np.full((h, W), track_a, dtype="float64")
        w = int(W * min(max(t / duration, 0.0), 1.0))
        if w > 0:
            a[:, :w] = fill_a
        return a

    clip = VideoClip(lambda t: bar, duration=duration)
    clip = clip.with_mask(VideoClip(make_alpha, duration=duration, is_mask=True)).with_fps(fps)
    return clip.with_position((0, 0 if at_top else H - h))


def build_flash_clips(cfg, times, video_size, fps, video_duration):
    """Brief full-frame white flash at each reveal time — sells the beat when a
    graphic lands. Cheap: a few frames of constant white under a fading mask."""
    m = cfg.get("motion") or {}
    if not m.get("flash", True) or not times:
        return []
    dur = max(float(m.get("flash_seconds", 0.1)), 1.0 / max(fps, 1))
    peak = min(max(float(m.get("flash_opacity", 0.16)), 0.0), 1.0)
    if peak <= 0:
        return []
    W, H = video_size
    white = np.full((H, W, 3), 255, dtype=np.uint8)
    clips = []
    for t0 in times:
        t0 = float(t0)
        if t0 < 0 or t0 >= video_duration - dur:
            continue

        def make_alpha(t, d=dur):
            return np.full((H, W), peak * max(0.0, 1.0 - t / d), dtype="float64")

        clip = VideoClip(lambda t: white, duration=dur)
        clip = clip.with_mask(VideoClip(make_alpha, duration=dur, is_mask=True))
        clips.append(clip.with_fps(fps).with_start(t0))
    return clips
