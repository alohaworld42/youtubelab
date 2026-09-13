"""Verse windows derived from the sung transcript must never collapse.

`sing.verse_times_from_words` anchors each verse by fuzzy-matching its first
line in the Whisper transcript, falling back to a proportional split for
verses it cannot match. The fill loop enforced monotonicity with a +1ms nudge,
so ONE late false match — a repeated chorus line scoring higher in the outro
than in its own verse; the matcher takes the best overlap in the remaining
stream, not the earliest good one — squeezed every later verse whose
proportional fallback landed before that point into a 1ms window.

Measured on a 60s song: 48.000s / 0.001s / 0.001s / 11.998s. The director then
tiles no shots into those verses and the cut gate rejects the result.

CPU-only, no network.
"""
import logging

import pytest

from pipeline.kidsong import sing


def _words(spec):
    """[(text, start), ...] -> the word-dict stream the matcher consumes."""
    return [{"word": t, "start": s, "end": s + 0.4} for t, s in spec]


def _verses(*first_lines):
    return [{"lines": [line]} for line in first_lines]


def test_a_late_false_match_does_not_collapse_the_later_verses(caplog):
    verses = _verses("alpha alpha alpha", "bravo bravo bravo",
                     "charlie charlie charlie", "delta delta delta")
    words = _words(
        [("alpha", 0.5)] * 3
        + [("zzz", 5.0 + i) for i in range(45)]
        + [("bravo", 55.0)] * 3
    )

    with caplog.at_level(logging.WARNING, logger="kidsong.sing"):
        windows = sing.verse_times_from_words(words, verses, 60.0)

    assert all(b - a >= sing._MIN_VERSE_SECONDS for a, b in windows), windows
    assert any("degenerate layout" in r.message for r in caplog.records)
    # ...and the fallback is the proportional split, not a nudged version of the
    # broken anchors.
    assert windows == [(0.0, 15.0), (15.0, 30.0), (30.0, 45.0), (45.0, 60.0)]


def test_a_healthy_anchoring_is_left_alone():
    """The guard must not throw away good anchors — the whole point of matching
    is that it beats a proportional guess."""
    verses = _verses("one one", "two two", "three three")
    words = _words([("one", 0.0)] * 2 + [("two", 20.0)] * 2 + [("three", 40.0)] * 2)

    windows = sing.verse_times_from_words(words, verses, 60.0)

    assert windows == [(0.0, 20.0), (20.0, 40.0), (40.0, 60.0)]


def test_windows_are_contiguous_and_cover_the_whole_song():
    verses = _verses("aa", "bb", "cc")
    words = _words([("aa", 0.0), ("bb", 10.0), ("cc", 20.0)])

    windows = sing.verse_times_from_words(words, verses, 30.0)

    assert windows[0][0] == 0.0
    assert windows[-1][1] == 30.0
    for (_, prev_end), (next_start, _) in zip(windows, windows[1:]):
        assert prev_end == next_start


def test_no_transcript_at_all_falls_back_to_the_proportional_split():
    verses = _verses("aa", "bb")
    assert sing.verse_times_from_words([], verses, 40.0) == [(0.0, 20.0), (20.0, 40.0)]


def test_a_single_verse_is_the_whole_song():
    assert sing.verse_times_from_words([], _verses("aa"), 30.0) == [(0.0, 30.0)]


def test_no_verses_yields_no_windows():
    assert sing.verse_times_from_words([], [], 30.0) == []


@pytest.mark.parametrize("n", [2, 3, 4, 5, 8])
def test_a_degenerate_layout_of_any_length_is_rebuilt(n):
    verses = _verses(*[f"line{i} line{i}" for i in range(n)])
    # Every verse matches at nearly the same late instant.
    words = _words([(f"line{i}", 50.0 + i * 0.001) for i in range(n)])

    windows = sing.verse_times_from_words(words, verses, 60.0)

    assert all(b - a >= sing._MIN_VERSE_SECONDS for a, b in windows), windows
