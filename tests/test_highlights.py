from pipeline.highlights import extract_highlights


def _words(specs):
    """specs: list of (word, start, end)"""
    return [{"word": w, "start": s, "end": e} for w, s, e in specs]


def _cfg(**overrides):
    h = {
        "enabled": True,
        "max_count": 4,
        "min_gap_seconds": 3.0,
        "hold_seconds": 0.6,
        "max_extra_words": 1,
    }
    h.update(overrides)
    return {"highlights": h}


def test_no_digits_returns_empty():
    words = _words([("HELLO", 0.0, 0.5), ("WORLD", 0.5, 1.0)])
    assert extract_highlights(words, _cfg()) == []


def test_single_number_picked_up_with_trailing_word():
    words = _words(
        [
            ("OCTOPUSES", 0.0, 0.6),
            ("HAVE", 0.6, 0.9),
            ("3", 0.9, 1.1),
            ("HEARTS", 1.1, 1.5),
        ]
    )
    picks = extract_highlights(words, _cfg())
    assert len(picks) == 1
    assert picks[0]["text"] == "3 HEARTS"
    assert picks[0]["start"] == 0.9
    assert abs(picks[0]["end"] - 2.1) < 1e-9  # 1.5 + hold_seconds(0.6)


def test_spelled_out_number_converts_to_numeral():
    # faster-whisper commonly writes small numbers as words, not digits.
    words = _words(
        [
            ("OCTOPUSES", 0.0, 0.6),
            ("HAVE", 0.6, 0.9),
            ("THREE", 0.9, 1.3),
            ("HEARTS", 1.3, 1.7),
        ]
    )
    picks = extract_highlights(words, _cfg())
    assert len(picks) == 1
    assert picks[0]["text"] == "3 HEARTS"


def test_respects_max_count():
    words = _words([(str(n), float(n), float(n) + 0.3) for n in range(10)])
    picks = extract_highlights(words, _cfg(min_gap_seconds=0.0, max_count=3))
    assert len(picks) == 3


def test_min_gap_merges_close_numbers():
    words = _words([("5", 0.0, 0.3), ("6", 0.5, 0.8)])
    picks = extract_highlights(words, _cfg(min_gap_seconds=3.0, max_extra_words=0))
    assert len(picks) == 1
    assert picks[0]["text"] == "5"


def test_disabled_returns_empty():
    words = _words([("3", 0.0, 0.3)])
    assert extract_highlights(words, _cfg(enabled=False)) == []


def test_empty_words_returns_empty():
    assert extract_highlights([], _cfg()) == []
