"""Caption groups must never cross a lyric line or a verse boundary.

A shipped episode rendered the pill "NAME CLAP YOUR" — the last word of
"One for the boy who knows my name" glued to the first two of "Clap your hands
and sing with me" — because `caption_render.group_words` chunked the flat word
list by count alone. On a sing-along channel for pre-readers the words on screen
ARE the product, so this is a correctness test, not a cosmetic one.

No model, no GPU, no network: the word list is built by hand or by
`alignment.align_to_lyrics` over a synthetic transcript.
"""
import pytest

from pipeline import alignment
from pipeline.caption_render import group_words


def _words(spec, per_line=None):
    """Build aligned-style words from [(text, line, verse), ...] at 1s/word."""
    out = []
    for i, (text, line, verse) in enumerate(spec):
        entry = {"word": text, "start": float(i), "end": float(i) + 0.9, "line": line}
        if verse is not None:
            entry["verse"] = verse
        out.append(entry)
    return out


def _texts(groups):
    return [g["text"] for g in groups]


# ------------------------------------------------------------------ lines ---
def test_group_never_spans_a_lyric_line():
    words = _words([
        ("WHO", 0, 0), ("KNOWS", 0, 0), ("MY", 0, 0), ("NAME", 0, 0),
        ("CLAP", 1, 0), ("YOUR", 1, 0), ("HANDS", 1, 0),
    ])
    assert _texts(group_words(words, 3)) == [
        "WHO KNOWS MY", "NAME", "CLAP YOUR HANDS",
    ]


def test_the_shipped_gibberish_group_is_gone():
    """The exact regression: "NAME CLAP YOUR" must never be emitted."""
    words = _words([
        ("ONE", 0, 1), ("FOR", 0, 1), ("THE", 0, 1), ("BOY", 0, 1),
        ("WHO", 0, 1), ("KNOWS", 0, 1), ("MY", 0, 1), ("NAME", 0, 1),
        ("CLAP", 1, 2), ("YOUR", 1, 2), ("HANDS", 1, 2),
    ])
    assert "NAME CLAP YOUR" not in _texts(group_words(words, 3))


# ----------------------------------------------------------------- verses ---
def test_group_never_spans_a_verse():
    """Even when two verses share a line index numbering, the verse splits."""
    words = _words([
        ("SING", 0, 0), ("WITH", 0, 0),
        ("ME", 0, 1), ("AGAIN", 0, 1),
    ])
    assert _texts(group_words(words, 4)) == ["SING WITH", "ME AGAIN"]


# ------------------------------------------------------------------ sizes ---
def test_single_word_groups_still_work():
    words = _words([("A", 0, 0), ("B", 0, 0), ("C", 1, 0)])
    assert _texts(group_words(words, 1)) == ["A", "B", "C"]


def test_short_line_is_not_padded_from_the_next_line():
    """A 2-word line under a group size of 3 stays 2 words long."""
    words = _words([
        ("UP", 0, 0), ("HIGH", 0, 0),
        ("DOWN", 1, 0), ("LOW", 1, 0), ("NOW", 1, 0),
    ])
    groups = group_words(words, 3)
    assert _texts(groups) == ["UP HIGH", "DOWN LOW NOW"]


def test_long_line_still_splits_by_count():
    words = _words([(str(i), 0, 0) for i in range(7)])
    assert _texts(group_words(words, 3)) == ["0 1 2", "3 4 5", "6"]


def test_words_per_group_is_a_maximum_not_a_stride():
    words = _words([
        ("A", 0, 0),
        ("B", 1, 0), ("C", 1, 0), ("D", 1, 0), ("E", 1, 0),
    ])
    assert _texts(group_words(words, 3)) == ["A", "B C D", "E"]


# ----------------------------------------------------------------- timing ---
def test_timing_is_monotonic_and_non_overlapping():
    words = _words([
        ("A", 0, 0), ("B", 0, 0), ("C", 1, 0), ("D", 1, 0), ("E", 2, 1),
    ])
    groups = group_words(words, 2)
    for g in groups:
        assert g["end"] >= g["start"]
    for a, b in zip(groups, groups[1:]):
        assert a["start"] <= b["start"]
        assert a["end"] <= b["start"] + 1e-9, f"{a['text']} outlasts {b['text']}"


def test_a_group_never_outlasts_the_next_groups_start():
    """alignment's MIN_WORD_SECONDS floor can push a word past its successor."""
    words = [
        {"word": "A", "start": 0.0, "end": 0.5, "line": 0},
        {"word": "B", "start": 0.4, "end": 0.9, "line": 1},
    ]
    groups = group_words(words, 3)
    assert len(groups) == 2
    assert groups[0]["end"] == pytest.approx(0.4)


def test_group_start_is_its_first_words_start():
    words = _words([("A", 0, 0), ("B", 0, 0), ("C", 1, 0)])
    groups = group_words(words, 2)
    assert groups[0]["start"] == pytest.approx(0.0)
    assert groups[-1]["start"] == pytest.approx(2.0)


# ------------------------------------------------------- no structure info ---
def test_raw_transcript_words_group_by_count_exactly_as_before():
    """Words with no line/verse (open transcription) keep the old behaviour."""
    words = [
        {"word": w, "start": float(i), "end": float(i) + 0.9}
        for i, w in enumerate(["A", "B", "C", "D", "E"])
    ]
    assert _texts(group_words(words, 2)) == ["A B", "C D", "E"]


def test_empty_input():
    assert group_words([], 3) == []
    assert group_words(None, 3) == []


# ------------------------------------------------- end-to-end via alignment ---
VERSES = [
    {"lines": ["One for the master, one for the dame",
               "One for the boy who knows my name"]},
    {"lines": ["Clap your hands and sing with me",
               "Baa, baa, black sheep, one, two, three"]},
]


def _fake_asr(verses):
    """A perfect transcript on a 0.5s/word clock — no model involved."""
    out, t = [], 0.0
    for line in alignment.verse_lines(verses):
        for word in line.split():
            out.append({"word": word, "start": t, "end": t + 0.4})
            t += 0.5
    return out


def test_alignment_carries_line_and_verse_through_to_groups():
    lines = alignment.verse_lines(VERSES, with_verse=True)
    words = alignment.align_to_lyrics(_fake_asr(VERSES), lines)
    assert words is not None
    assert {w["verse"] for w in words} == {0, 1}

    groups = group_words(words, 3)
    # Re-derive each group's membership and assert it is structurally uniform.
    idx = 0
    for g in groups:
        chunk = words[idx:idx + len(g["text"].split())]
        idx += len(chunk)
        assert len({(w["verse"], w["line"]) for w in chunk}) == 1, g["text"]
    assert idx == len(words)
    assert "name Clap your" not in " | ".join(_texts(groups))


def test_verse_lines_pair_form_matches_plain_form():
    plain = alignment.verse_lines(VERSES)
    pairs = alignment.verse_lines(VERSES, with_verse=True)
    assert [p[0] for p in pairs] == plain
    assert [p[1] for p in pairs] == [0, 0, 1, 1]
    # lyric_tokens accepts either form and yields identical tokens.
    assert alignment.lyric_tokens(pairs) == alignment.lyric_tokens(plain)


def test_align_to_lyrics_without_verses_still_sets_line_only():
    words = alignment.align_to_lyrics(_fake_asr(VERSES),
                                      alignment.verse_lines(VERSES))
    assert words is not None
    assert all("verse" not in w for w in words)
    assert {w["line"] for w in words} == {0, 1, 2, 3}
