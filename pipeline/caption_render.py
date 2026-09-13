"""
caption_render.py — Render a caption (a word or short phrase) to a transparent PNG
using Pillow. We avoid MoviePy's TextClip on purpose because that needs ImageMagick,
which is a pain to install on Windows. Pillow is already a dependency and just works.
"""
import os
import tempfile

_FONT_CACHE = {}
_TMPDIR = tempfile.mkdtemp(prefix="brainrot_caps_")


def _get_font(font_path, size):
    from PIL import ImageFont

    key = (font_path, size)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    try:
        font = ImageFont.truetype(font_path, size)
    except Exception:
        # Last-resort fallback; looks plain but never crashes. Pass the size so
        # captions don't collapse to Pillow's tiny default bitmap font
        # (load_default honours `size` on Pillow >= 10, which we pin).
        try:
            font = ImageFont.load_default(size=size)
        except TypeError:
            font = ImageFont.load_default()
    _FONT_CACHE[key] = font
    return font


def render_group(text, cfg, idx):
    """Render `text` to a PNG and return its path.

    Optional rounded "pill" behind the text (captions.pill in config) for the
    friendlier kids/clean styles; the classic stroked look needs no pill.
    """
    from PIL import Image, ImageDraw

    c = cfg["captions"]
    if c.get("uppercase", True):
        text = text.upper()
    font = _get_font(c["font_path"], c["font_size"])
    fill = tuple(c["fill_color"])
    stroke = tuple(c["stroke_color"])
    stroke_w = int(c["stroke_width"])
    pill = c.get("pill") or {}
    pill_on = bool(pill.get("enabled"))

    # Measure with stroke included.
    tmp = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    d = ImageDraw.Draw(tmp)
    bbox = d.textbbox((0, 0), text, font=font, stroke_width=stroke_w)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    pad_x = int(pill.get("pad_x", 0)) if pill_on else 0
    pad_y = int(pill.get("pad_y", 0)) if pill_on else 0
    pad = stroke_w * 2 + 24
    W = tw + (pad + pad_x) * 2
    H = th + (pad + pad_y) * 2

    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    if pill_on:
        color = tuple(pill.get("color", [0, 0, 0]))
        opacity = int(pill.get("opacity", 220))
        radius = int(pill.get("radius", 30))
        inset = max(6, stroke_w)
        draw.rounded_rectangle(
            [inset, inset, W - inset, H - inset],
            radius=radius,
            fill=color + (opacity,),
        )

    x = pad + pad_x - bbox[0]
    y = pad + pad_y - bbox[1]
    draw.text(
        (x, y),
        text,
        font=font,
        fill=fill + (255,),
        stroke_width=stroke_w,
        stroke_fill=stroke + (255,),
    )

    out = os.path.join(_TMPDIR, f"cap_{idx:04d}.png")
    img.save(out)
    return out


def _structure_key(word):
    """(verse, line) for a word that knows where it sits in the lyrics.

    `alignment.align_to_lyrics` stamps every word with its lyric line index (and
    its verse index when the caller supplied verses). Raw-transcript words have
    neither, and return None here — those group purely by count, exactly as
    before.
    """
    line = word.get("line")
    verse = word.get("verse")
    if line is None and verse is None:
        return None
    return (verse, line)


def group_words(words, words_per_group):
    """Chunk the flat word list into caption groups, each with start/end/text.

    A group NEVER spans a lyric line or a verse boundary. Chunking purely by
    count glued the tail of one line to the head of the next and put nonsense on
    screen for a pre-reader to follow — a real shipped episode showed
    "NAME CLAP YOUR", the last word of "One for the boy who knows my name"
    joined to the first two of "Clap your hands and sing with me". So the count
    is a MAXIMUM within a line, not a fixed stride: a line shorter than
    `words_per_group` yields a short group rather than borrowing from the next
    line.
    """
    groups = []
    n = max(1, int(words_per_group))

    def flush(chunk):
        if chunk:
            groups.append(
                {
                    "text": " ".join(w["word"] for w in chunk),
                    "start": chunk[0]["start"],
                    "end": chunk[-1]["end"],
                }
            )

    chunk = []
    current = None
    for word in words or []:
        key = _structure_key(word)
        # Split when the group is full, or when this word starts a new lyric
        # line/verse. A word with no structure (raw transcript) never forces a
        # split, so a mixed list degrades to count-only chunking.
        if chunk and (len(chunk) >= n or (key is not None and current is not None and key != current)):
            flush(chunk)
            chunk = []
        if not chunk:
            current = key
        chunk.append(word)
    flush(chunk)

    # A group must not still be on screen when the next one appears: the
    # per-word MIN_WORD_SECONDS floor in alignment can push a word's end past
    # the following word's start, which would stack two pills.
    for group, nxt in zip(groups, groups[1:]):
        if group["end"] > nxt["start"] >= group["start"]:
            group["end"] = nxt["start"]
    return groups
