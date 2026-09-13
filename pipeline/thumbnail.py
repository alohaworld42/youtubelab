"""
thumbnail.py — Generate a 1280x720 YouTube thumbnail from a finished video.

Picks the most colorful frame, blur-fills the 16:9 canvas, places the sharp
vertical frame right of center and paints the title big on the left. Colors
and font come from the caption style so thumbnails match the channel look.
"""
import os
import unicodedata

from PIL import Image, ImageDraw, ImageFilter, ImageFont

THUMB_W, THUMB_H = 1280, 720


def _strip_emoji(text):
    """Drop emoji/symbol chars — standard bold fonts render them as tofu boxes."""
    out = []
    for ch in text or "":
        if ord(ch) > 0xFFFF:  # astral plane = emoji/pictographs
            continue
        if unicodedata.category(ch) in ("So", "Sk", "Cs"):  # symbols, surrogates
            continue
        out.append(ch)
    return " ".join("".join(out).split())


def _colorfulness(arr):
    """Hasler–Süsstrunk colorfulness metric (simplified), on a numpy frame."""
    import numpy as np

    rg = arr[:, :, 0].astype("float32") - arr[:, :, 1]
    yb = 0.5 * (arr[:, :, 0].astype("float32") + arr[:, :, 1]) - arr[:, :, 2]
    return float(
        (rg.std() ** 2 + yb.std() ** 2) ** 0.5
        + 0.3 * ((rg.mean() ** 2 + yb.mean() ** 2) ** 0.5)
    )


def pick_vivid_frame(clip, samples=5):
    """Most colorful frame from evenly spaced sample times of a MoviePy clip."""
    best, best_score = None, -1.0
    dur = max(clip.duration or 1.0, 0.5)
    for i in range(1, samples + 1):
        t = dur * i / (samples + 1)
        try:
            frame = clip.get_frame(t)
        except Exception:
            continue
        score = _colorfulness(frame)
        if score > best_score:
            best, best_score = frame, score
    return best


def _wrap_title(draw, title, font, max_width):
    words = title.split()
    lines, cur = [], ""
    for w in words:
        cand = (cur + " " + w).strip()
        if draw.textlength(cand, font=font) <= max_width or not cur:
            cur = cand
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines[:4]


def compose_thumbnail(frame_img, title, cfg, out_path):
    """Compose the 1280x720 thumbnail from a PIL frame + title text."""
    c = cfg["captions"]
    accent = tuple((c.get("pill") or {}).get("color") or c.get("highlight_color") or [255, 222, 0])
    stroke = tuple(c.get("stroke_color", [0, 0, 0]))

    # blurred cover background
    bg = frame_img.copy()
    scale = max(THUMB_W / bg.width, THUMB_H / bg.height)
    bg = bg.resize((round(bg.width * scale), round(bg.height * scale)))
    bg = bg.crop((
        (bg.width - THUMB_W) // 2, (bg.height - THUMB_H) // 2,
        (bg.width - THUMB_W) // 2 + THUMB_W, (bg.height - THUMB_H) // 2 + THUMB_H,
    )).filter(ImageFilter.GaussianBlur(22))
    canvas = Image.new("RGB", (THUMB_W, THUMB_H))
    canvas.paste(bg, (0, 0))

    # sharp vertical frame on the right
    fg = frame_img.copy()
    fg_h = THUMB_H
    fg_w = round(fg.width * fg_h / fg.height)
    fg = fg.resize((fg_w, fg_h))
    fg_x = THUMB_W - fg_w - 48
    canvas.paste(fg, (fg_x, 0))

    # darken left panel for text contrast
    panel = Image.new("L", (THUMB_W, THUMB_H), 0)
    pd = ImageDraw.Draw(panel)
    pd.rectangle([0, 0, fg_x, THUMB_H], fill=110)
    canvas = Image.composite(Image.new("RGB", canvas.size, (10, 10, 14)), canvas, panel)

    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype(c["font_path"], 92)
    except Exception:
        font = ImageFont.load_default()

    max_text_w = max(fg_x - 96, 300)
    lines = _wrap_title(draw, _strip_emoji(title), font, max_text_w)
    line_h = 104
    total_h = line_h * len(lines)
    y = (THUMB_H - total_h) // 2
    for i, line in enumerate(lines):
        fill = accent if i == 0 else (255, 255, 255)
        draw.text((48, y), line, font=font, fill=fill,
                  stroke_width=6, stroke_fill=stroke)
        y += line_h

    canvas.save(out_path, "JPEG", quality=90)
    return out_path


def generate_thumbnail(video_path, title, cfg, out_path=None):
    """Full pipeline: open video, pick vivid frame, compose, save. Returns the
    thumbnail path or None on any failure (thumbnails are never fatal)."""
    from moviepy import VideoFileClip

    out_path = out_path or os.path.splitext(video_path)[0] + "-thumb.jpg"
    try:
        with VideoFileClip(video_path) as clip:
            frame = pick_vivid_frame(clip)
        if frame is None:
            return None
        compose_thumbnail(Image.fromarray(frame), title, cfg, out_path)
        return out_path
    except Exception:
        return None
