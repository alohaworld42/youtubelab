"""Lyric-to-transcript alignment and the captions fallback chain (no model, no GPU)."""
import pytest

from pipeline import alignment, captions

LINES = ["We're washing our hands", "Rub-a-dub-dub with soap today"]


def _asr(pairs):
    return [{"word": w, "start": s, "end": s + 0.3} for w, s in pairs]


def _words(result):
    return [w["word"] for w in result]


# ------------------------------------------------------------------ alignment ---
def test_perfect_transcript_keeps_lyric_text_and_asr_timings():
    asr = _asr([("We're", 1.0), ("washing", 1.5), ("our", 2.0), ("hands", 2.5)])
    out = alignment.align_to_lyrics(asr, ["We're washing our hands"])
    assert _words(out) == ["We're", "washing", "our", "hands"]
    assert out[0]["start"] == 1.0
    assert out[3]["end"] == pytest.approx(2.8)


def test_misheard_words_are_replaced_by_the_known_lyrics():
    # Whisper hears "hand" and injects a junk token; output must be the real text.
    asr = _asr([("Music", 0.2), ("were", 1.0), ("washing", 1.5), ("our", 2.0), ("hand", 2.5)])
    out = alignment.align_to_lyrics(asr, ["We're washing our hands"])
    assert _words(out) == ["We're", "washing", "our", "hands"]
    assert "Music" not in _words(out)


def test_unanchored_words_are_interpolated_between_anchors():
    # "our" is missing from the transcript; it must still get a slot in between.
    asr = _asr([("washing", 1.0), ("hands", 3.0)])
    out = alignment.align_to_lyrics(asr, ["washing our hands"])
    assert _words(out) == ["washing", "our", "hands"]
    assert 1.3 <= out[1]["start"] <= 3.0
    assert out[0]["start"] <= out[1]["start"] <= out[2]["start"]


def test_timings_are_monotonic_and_never_zero_length():
    asr = _asr([("washing", 1.0), ("hands", 1.0), ("soap", 5.0)])
    out = alignment.align_to_lyrics(asr, LINES)
    for prev, cur in zip(out, out[1:]):
        assert cur["start"] >= prev["start"]
    for w in out:
        assert w["end"] - w["start"] >= alignment.MIN_WORD_SECONDS - 1e-9


def test_hyphenated_tokens_match_however_the_model_splits_them():
    asr = _asr([("rub", 1.0), ("a", 1.4), ("dub", 1.8), ("dub", 2.2), ("soap", 3.0)])
    out = alignment.align_to_lyrics(asr, ["Rub-a-dub-dub with soap today"])
    assert _words(out)[0] == "Rub-a-dub-dub"


def test_line_indices_are_carried_through():
    asr = _asr([("washing", 1.0), ("soap", 5.0)])
    out = alignment.align_to_lyrics(asr, LINES)
    assert {w["line"] for w in out} == {0, 1}


def test_audio_end_clamps_trailing_extrapolation():
    asr = _asr([("We're", 1.0)])
    out = alignment.align_to_lyrics(asr, ["We're washing our hands"], audio_end=2.0)
    assert out[-1]["end"] <= 2.0 + alignment.MIN_WORD_SECONDS


def test_unrelated_transcript_is_rejected_rather_than_forced():
    asr = _asr([("banana", 1.0), ("helicopter", 2.0), ("xylophone", 3.0)])
    assert alignment.align_to_lyrics(asr, LINES) is None


@pytest.mark.parametrize("asr,lines", [([], LINES), (_asr([("washing", 1.0)]), []), (None, LINES)])
def test_empty_inputs_return_none(asr, lines):
    assert alignment.align_to_lyrics(asr, lines) is None


def test_verse_lines_flattens_song_verses():
    verses = [{"lines": ["a", "b"]}, {"lines": ["c"]}]
    assert alignment.verse_lines(verses) == ["a", "b", "c"]


# ------------------------------------------------------- captions integration ---
CFG = {"whisper": {"model_size": "small", "device": "cpu", "compute_type": "int8"}}


@pytest.fixture
def fake_transcribe(monkeypatch):
    """Replace the model call so tests never touch faster-whisper, disk or GPU."""
    calls = []

    def _install(result):
        def fake(audio_path, cfg, vad=None, language="en", size=None):
            calls.append({"vad": vad, "size": size, "language": language})
            return list(result() if callable(result) else result)

        monkeypatch.setattr(captions, "transcribe_words", fake)
        return calls

    return _install


def test_vad_is_off_by_default_so_singing_is_not_discarded():
    # The shipped bug: vad_filter=True dropped 100% of sung audio.
    assert captions._vad_default(CFG) is False


def test_lyrics_supplied_yields_aligned_known_text(fake_transcribe):
    fake_transcribe(_asr([("were", 1.0), ("washing", 1.5), ("our", 2.0), ("hand", 2.5)]))
    out = captions.get_word_timestamps("a.wav", CFG, lyrics=["We're washing our hands"])
    assert _words(out) == ["WE'RE", "WASHING", "OUR", "HANDS"]


def test_lyrics_accepts_song_verses_directly(fake_transcribe):
    fake_transcribe(_asr([("washing", 1.0), ("our", 1.5), ("hands", 2.0)]))
    out = captions.get_word_timestamps(
        "a.wav", CFG, lyrics=[{"lines": ["We're washing our hands"]}], uppercase=False
    )
    assert _words(out) == ["We're", "washing", "our", "hands"]


def test_falls_back_to_raw_transcript_when_alignment_is_rejected(fake_transcribe):
    fake_transcribe(_asr([("banana", 1.0), ("helicopter", 2.0)]))
    out = captions.get_word_timestamps("a.wav", CFG, lyrics=LINES)
    assert _words(out) == ["BANANA", "HELICOPTER"]


def test_alignment_crash_degrades_to_transcript_instead_of_killing_the_render(
    fake_transcribe, monkeypatch
):
    fake_transcribe(_asr([("washing", 1.0)]))
    monkeypatch.setattr(
        alignment, "align_to_lyrics", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    out = captions.get_word_timestamps("a.wav", CFG, lyrics=LINES)
    assert _words(out) == ["WASHING"]


def test_empty_transcript_triggers_a_vad_off_retry(fake_transcribe):
    results = [[], _asr([("washing", 1.0)])]
    calls = fake_transcribe(lambda: results.pop(0))
    out = captions.get_word_timestamps("a.wav", CFG, lyrics=LINES)
    assert len(calls) == 2
    assert calls[1]["vad"] is False
    assert _words(out) == ["WASHING"]


def test_total_failure_returns_empty_without_raising(fake_transcribe):
    fake_transcribe([])
    assert captions.get_word_timestamps("a.wav", CFG, lyrics=LINES) == []


def test_song_path_uses_the_larger_model(fake_transcribe):
    calls = fake_transcribe(_asr([("washing", 1.0)]))
    captions.get_word_timestamps("a.wav", CFG, lyrics=LINES)
    assert calls[0]["size"] == "medium"


def test_speech_path_uses_the_configured_model(fake_transcribe):
    calls = fake_transcribe(_asr([("hello", 1.0)]))
    out = captions.get_word_timestamps("a.wav", CFG)
    assert calls[0]["size"] is None
    assert _words(out) == ["HELLO"]


# ------------------------------------------- unanchored head/tail (stacking) --
def test_an_unanchored_head_is_spread_not_piled_onto_one_instant():
    """Whisper regularly drops the quiet opening of a song. The head extrapolation
    used to walk backward one nominal word at a time and clamp at 0, so every
    word that did not fit landed on the SAME instant: twelve lyric words with the
    first anchor at 0.5s gave seven of them timed 0.000-0.080 — one 2-frame flash
    of overlapping captions instead of seven words."""
    lyrics = ["one two three four five six seven eight nine ten eleven twelve"]
    asr = _asr([("nine", 0.5), ("ten", 0.8), ("eleven", 1.1), ("twelve", 1.4)])

    out = alignment.align_to_lyrics(asr, lyrics)

    starts = [round(w["start"], 4) for w in out]
    assert len(set(starts)) == len(starts), f"words share a start time: {starts}"
    assert starts == sorted(starts)
    assert out[0]["start"] == 0.0
    # The head must hand off to the first anchor, not overshoot it.
    assert out[8]["start"] == pytest.approx(0.5)


def test_an_unanchored_head_keeps_natural_pacing_when_there_is_room():
    """Compression is for when the words do not fit; with room to spare they
    should sit at the nominal rate and end on the anchor."""
    lyrics = ["one two three four"]
    asr = _asr([("four", 10.0)])

    out = alignment.align_to_lyrics(asr, lyrics)

    gaps = [round(b["start"] - a["start"], 4) for a, b in zip(out, out[1:])][:3]
    assert all(g == pytest.approx(alignment.DEFAULT_WORD_SECONDS) for g in gaps), gaps
    assert out[3]["start"] == pytest.approx(10.0)


def test_an_unanchored_tail_is_spread_not_piled_onto_one_instant():
    lyrics = ["one two three four five six seven eight"]
    asr = _asr([("one", 0.0), ("two", 0.3)])

    out = alignment.align_to_lyrics(asr, lyrics, audio_end=1.2)

    starts = [round(w["start"], 4) for w in out]
    assert len(set(starts)) == len(starts), f"words share a start time: {starts}"
    assert starts == sorted(starts)
    assert out[-1]["end"] <= 1.2 + alignment.MIN_WORD_SECONDS


def test_a_tail_with_no_room_left_still_produces_distinct_times():
    """`audio_end` at or before the last anchor used to give the whole tail the
    same zero-length slot. Honouring an impossible bound is worse than
    overshooting it — the editor clips to the real duration anyway."""
    lyrics = ["one two three four five"]
    asr = _asr([("one", 0.0), ("two", 4.0)])

    out = alignment.align_to_lyrics(asr, lyrics, audio_end=1.0)  # already past it

    starts = [round(w["start"], 4) for w in out]
    assert len(set(starts)) == len(starts), f"words share a start time: {starts}"
    assert starts == sorted(starts)


def test_head_and_tail_together_stay_monotonic_and_non_zero_length():
    # Eight words with two anchors sits exactly on MIN_ANCHOR_RATIO (0.25), so
    # the alignment is accepted; ten would be rejected as untrustworthy instead.
    lyrics = ["a b c d e f g h"]
    asr = _asr([("e", 2.0), ("f", 2.4)])

    out = alignment.align_to_lyrics(asr, lyrics, audio_end=3.0)

    for prev, cur in zip(out, out[1:]):
        assert cur["start"] >= prev["start"] - 1e-9
    for w in out:
        assert w["end"] - w["start"] >= alignment.MIN_WORD_SECONDS - 1e-9
