"""
highlights.py — Pick out short "stat" moments from the Whisper word timeline
and turn them into on-screen highlight badges (e.g. "3 GB", "38", "0.2S"),
synced to the exact words that were said.

Fully deterministic: no LLM call, just a scan over the same word-level
timestamps captions.py already produces for the word-by-word captions.
"""
import re

_DIGIT = re.compile(r"\d")

# faster-whisper commonly spells small numbers out ("3" -> "THREE") instead of
# using numerals, so a digit-only regex misses most spoken stats. Map the
# common spelled-out forms to their numeral for both detection and display.
_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14",
    "fifteen": "15", "sixteen": "16", "seventeen": "17", "eighteen": "18",
    "nineteen": "19", "twenty": "20", "thirty": "30", "forty": "40",
    "fifty": "50", "sixty": "60", "seventy": "70", "eighty": "80",
    "ninety": "90", "hundred": "100", "thousand": "1,000",
    "million": "1,000,000", "billion": "1,000,000,000",
}


def _numeric_form(word):
    """Return a display numeral ('3') if `word` is a digit token or a
    spelled-out number, else None."""
    if _DIGIT.search(word):
        return word
    return _NUMBER_WORDS.get(word.lower())


def extract_highlights(words, cfg):
    """Return [{"text": "3 GB", "start": 4.0, "end": 4.6}, ...] for runs of
    words that resolve to a number, capped and spaced out per `highlights`
    config.
    """
    h = cfg.get("highlights") or {}
    if not h.get("enabled", True) or not words:
        return []

    max_count = int(h.get("max_count", 4))
    min_gap = float(h.get("min_gap_seconds", 3.0))
    hold = float(h.get("hold_seconds", 0.6))
    max_extra_words = int(h.get("max_extra_words", 1))

    picks = []
    i, n = 0, len(words)
    while i < n and len(picks) < max_count:
        num = _numeric_form(words[i]["word"])
        if num is None:
            i += 1
            continue
        j = i
        parts = [num]
        while (
            j + 1 < n
            and len(parts) <= max_extra_words
            and _numeric_form(words[j + 1]["word"]) is None
            and len(words[j + 1]["word"]) <= 6
            and words[j + 1]["start"] - words[j]["end"] < 0.35
        ):
            j += 1
            parts.append(words[j]["word"])
        start, end = words[i]["start"], words[j]["end"]
        if not picks or start - picks[-1]["end"] >= min_gap:
            picks.append({"text": " ".join(parts), "start": start, "end": end + hold})
        i = j + 1
    return picks


def to_stat_specs(highlights):
    """Convert extract_highlights() output into animated 'stat' graphic specs
    for the motion engine: the leading number is the value, the rest the label.
    """
    specs = []
    for h in highlights:
        parts = h["text"].split(" ", 1)
        specs.append(
            {
                "type": "stat",
                "value": parts[0],
                "label": parts[1] if len(parts) > 1 else "",
                "start": h["start"],
                "end": h["end"],
            }
        )
    return specs
