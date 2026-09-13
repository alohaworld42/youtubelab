"""
alignment.py — Anchor known lyrics onto ASR word timings.

For sung audio we already KNOW the text (the song's verses came from the LLM and
are persisted in output/<base>-song.json). Open transcription of singing is
unreliable: Whisper drops quiet verses, mangles words ("hands" -> "hand") and
emits junk tokens ("Music", "Ooh!"). But its *timings* on the words it does
recognise are good.

So instead of trusting the transcript, we align it: run a sequence match between
the recognised words and the known lyric words, treat every match as a time
anchor, and interpolate the timings of the lyric words in between. The output
text is always the real lyrics; only the clock comes from Whisper.

    align_to_lyrics(asr_words, lyric_lines) -> [{"word", "start", "end", "line"}]

Returns None when too little of the lyric text could be anchored, so callers can
fall back to the raw transcript rather than render confidently-wrong captions.
"""
import difflib
import re

# Below this fraction of lyric words anchored, the alignment is not trustworthy.
MIN_ANCHOR_RATIO = 0.25
# Nominal seconds per word, used only to extrapolate past the outermost anchors.
DEFAULT_WORD_SECONDS = 0.35
# No caption word may be shorter than this, or it flashes for zero frames.
MIN_WORD_SECONDS = 0.08


def _norm(word):
    """Comparison key: lowercase, letters/digits/apostrophes only.

    Hyphenated singing tokens ("Rub-a-dub-dub") lose their hyphens so they match
    whichever way the model chose to split them.

    German umlauts and ß are KEPT (the old ASCII-only class dropped them, so
    "für"->"fr" and German lyrics/ASR barely matched); English is unaffected
    because English tokens contain none of these characters.
    """
    return re.sub(r"[^a-z0-9äöüß']", "", (word or "").lower())


def lyric_tokens(lines):
    """Flatten lyric lines into [(display_word, norm_key, line_index), ...].

    `lines` may be plain strings, or (line_text, verse_index) pairs as produced
    by `verse_lines(..., with_verse=True)`. The verse index is not returned here
    — `align_to_lyrics` re-attaches it — but accepting the pair form means the
    caller does not have to strip it first.
    """
    out = []
    for idx, line in enumerate(lines or []):
        if isinstance(line, (tuple, list)) and len(line) == 2:
            line = line[0]
        for raw in str(line).split():
            key = _norm(raw)
            if key:
                out.append((raw.strip(), key, idx))
    return out


def line_verses(lines):
    """Verse index per line for the (line_text, verse_index) pair form.

    Returns None for the plain-string form, where verse structure is unknown.
    """
    out = []
    for line in lines or []:
        if isinstance(line, (tuple, list)) and len(line) == 2:
            out.append(line[1])
        else:
            out.append(None)
    return out


def verse_lines(verses, with_verse=False):
    """Flatten a song's verses ([{"lines": [...]}, ...]) to a list of lines.

    With `with_verse=True` each entry is a (line_text, verse_index) pair, so the
    verse structure survives the flattening and captions can be forbidden from
    spanning a verse boundary.
    """
    lines = []
    for v_idx, verse in enumerate(verses or []):
        for line in (verse.get("lines") or []) if isinstance(verse, dict) else []:
            lines.append((line, v_idx) if with_verse else line)
    return lines


def _interpolate(tokens, anchors, audio_end, verses=None):
    """Fill timings for every lyric token given sparse (index, start, end) anchors."""
    n = len(tokens)
    timed = [None] * n
    for idx, start, end in anchors:
        timed[idx] = (start, end)

    # Between consecutive anchors: spread the gap evenly over the unanchored run.
    for a, b in zip(anchors, anchors[1:]):
        lo, hi = a[0], b[0]
        gap = hi - lo
        if gap <= 1:
            continue
        t0, t1 = a[2], b[1]  # end of the left anchor -> start of the right one
        if t1 <= t0:
            t1 = t0 + DEFAULT_WORD_SECONDS * (gap - 1)
        step = (t1 - t0) / (gap - 1 + 1e-9)
        for k in range(1, gap):
            s = t0 + step * (k - 1)
            timed[lo + k] = (s, s + step)

    # Before the first anchor / after the last: lay the unanchored run out at the
    # nominal rate when there is room for it, otherwise COMPRESS it evenly into
    # the room there is.
    #
    # This used to walk outward one nominal word at a time and clamp at the
    # boundary, which piled every word that did not fit onto the same instant:
    # a song whose first anchor lands at 0.5s gave seven lyric words all timed
    # 0.000-0.080, rendering as one 2-frame flash of overlapping captions
    # instead of seven words. The tail had the mirror image whenever `audio_end`
    # sat at or before the last anchor's end. Evenly spread is not perfect
    # either — the words are guesses, there are no anchors — but distinct and
    # ordered beats identical and stacked.
    first, last = anchors[0][0], anchors[-1][0]

    if first > 0:
        head_end = timed[first][0]
        natural = first * DEFAULT_WORD_SECONDS
        step = DEFAULT_WORD_SECONDS if natural <= head_end else head_end / first
        base = max(0.0, head_end - step * first)
        for k in range(first):
            s = base + step * k
            timed[k] = (s, s + step)

    tail_count = n - 1 - last
    if tail_count > 0:
        tail_start = timed[last][1]
        natural = tail_count * DEFAULT_WORD_SECONDS
        room = None if audio_end is None else float(audio_end) - tail_start
        # `room <= 0` means the transcript already runs to (or past) the end of
        # the audio. Honouring it would stack the whole tail on one instant, so
        # fall back to the nominal rate and let the editor clip to duration.
        step = room / tail_count if (room is not None and 0 < room < natural) else DEFAULT_WORD_SECONDS
        for i, k in enumerate(range(last + 1, n)):
            s = tail_start + step * i
            timed[k] = (s, s + step)

    words = []
    for (display, _key, line), (start, end) in zip(tokens, timed):
        start = max(0.0, float(start))
        # Floor the duration so an interpolated word is never rendered for 0 frames.
        end = max(start + MIN_WORD_SECONDS, float(end))
        entry = {"word": display, "start": start, "end": end, "line": line}
        if verses:
            verse = verses[line] if line < len(verses) else None
            if verse is not None:
                entry["verse"] = verse
        words.append(entry)
    return words


def align_to_lyrics(asr_words, lines, audio_end=None):
    """Re-time the known lyrics using the ASR transcript as a clock.

    asr_words: [{"word", "start", "end"}, ...] from open transcription.
    lines:     the known lyric lines, in order.
    Returns the lyric words with timings, or None if too few could be anchored.
    """
    tokens = lyric_tokens(lines)
    asr_words = [w for w in (asr_words or []) if _norm(w.get("word"))]
    if not tokens or not asr_words:
        return None

    lyric_keys = [t[1] for t in tokens]
    asr_keys = [_norm(w.get("word")) for w in asr_words]

    matcher = difflib.SequenceMatcher(a=lyric_keys, b=asr_keys, autojunk=False)
    anchors = []
    for block in matcher.get_matching_blocks():
        for k in range(block.size):
            src = asr_words[block.b + k]
            anchors.append(
                (block.a + k, float(src.get("start", 0.0)), float(src.get("end", 0.0)))
            )

    if not anchors:
        return None
    # Anchors must advance in time; drop any that would run the clock backwards.
    clean = []
    for anchor in anchors:
        if not clean or anchor[1] >= clean[-1][2] - 1e-6:
            clean.append(anchor)
    anchors = clean

    ratio = len(anchors) / float(len(tokens))
    if ratio < MIN_ANCHOR_RATIO:
        return None

    print(
        f"[alignment] anchored {len(anchors)}/{len(tokens)} lyric words "
        f"({ratio:.0%}) on {len(asr_words)} transcribed words."
    )
    return _interpolate(tokens, anchors, audio_end, verses=line_verses(lines))
