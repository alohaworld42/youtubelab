"""Regression tests for the German-language craft gate in script_qc.review_script.

Context: `review_script` is the lyric craft gate that runs right after
`lyrics.generate_song`. Every craft heuristic it started with (meter, rhyme, the
action-verb reward, the capitalized-name cast detector, the clumsy-phrasing
function-word list) is English-specific, and false-fired on well-formed German
lyrics: German's regular "...mit mir mit" endings misread as clumsy repetition,
its syllable counts and rhyme table don't match English pronunciation tables,
and its blanket noun-capitalization would (were it not gated) misread every
noun as a cast name.

kidsong.language="de" ships only CURATED, pre-vetted library songs (see
prompts/pd_songs.de.json / lyrics.FALLBACK_SONG_DE), so the fix is not a real
German prosody model — it is to SKIP the English-only sub-checks for any
non-English language and keep only the language-agnostic structural/safety
checks (verse/line counts, repetition, hook, brand words) active. English
behavior must stay byte-identical: the Unicode tokenizer fix that underpins all
of this (`_words`/`_letters` keeping äöüß) is unconditional but only affects
non-ASCII input, so it cannot change English output either.
"""
import copy

import pytest

from pipeline.kidsong import lyrics, script_qc


@pytest.fixture()
def cfg_de():
    from pipeline.config import load_config

    c = copy.deepcopy(load_config())
    c.setdefault("kidsong", {})["language"] = "de"
    c["kidsong"].setdefault("review", {})["reviewer"] = "heuristic"
    return c


@pytest.fixture()
def cfg_en():
    from pipeline.config import load_config

    c = copy.deepcopy(load_config())
    c.setdefault("kidsong", {}).setdefault("review", {})["reviewer"] = "heuristic"
    return c


def _reasons(verdict):
    return " | ".join(verdict["reasons"])


_ENGLISH_ONLY_REASONS = (
    "character names overused",
    "no rhyme",
    "meter inconsistent",
    "not enough action verbs",
)


# --------------------------------------------------------------- German path --
@pytest.mark.parametrize("song_attr", ["FALLBACK_SONG_DE", "LEARNING_FALLBACK_SONG_DE"])
def test_german_fallback_songs_pass_the_craft_gate(cfg_de, song_attr):
    song = getattr(lyrics, song_attr)
    verdict = script_qc.review_script(song, cfg_de)
    assert verdict["accept"], f"{song_attr} rejected: {_reasons(verdict)}"


@pytest.mark.parametrize("song_attr", ["FALLBACK_SONG_DE", "LEARNING_FALLBACK_SONG_DE"])
def test_german_fallback_songs_never_hit_the_english_only_reasons(cfg_de, song_attr):
    """Even if some OTHER, unrelated reason ever crept in, the specifically
    English-only sub-checks this change targets must never fire for German."""
    song = getattr(lyrics, song_attr)
    verdict = script_qc.review_script(song, cfg_de)
    joined = _reasons(verdict)
    for bad_reason in _ENGLISH_ONLY_REASONS:
        assert bad_reason not in joined, f"{song_attr}: {joined}"


def test_german_language_is_permissive_not_strict():
    """language='de' does not tighten anything relative to 'en' — a German
    song with the SAME structural shape as the passing English fallback must
    also pass (sanity check that the gate isn't accidentally more demanding)."""
    from pipeline.config import load_config

    cfg = copy.deepcopy(load_config())
    cfg.setdefault("kidsong", {})["language"] = "de"
    cfg["kidsong"].setdefault("review", {})["reviewer"] = "heuristic"
    verdict = script_qc.review_script(lyrics.FALLBACK_SONG_DE, cfg)
    assert verdict["score"] == 1.0, _reasons(verdict)


# ------------------------------------------------------- English unaffected --
def test_english_fallback_song_still_passes_unchanged(cfg_en):
    """Sanity check: the German branch work must not disturb English at all."""
    song = lyrics.FALLBACK_SONG
    verdict = script_qc.review_script(song, cfg_en)
    assert verdict["accept"], _reasons(verdict)
    assert verdict["score"] == 1.0


def test_english_learning_fallback_song_still_passes_unchanged(cfg_en):
    song = lyrics.LEARNING_FALLBACK_SONG
    verdict = script_qc.review_script(song, cfg_en)
    assert verdict["accept"], _reasons(verdict)


def test_default_language_is_english_and_unaffected(cfg_en):
    """No kidsong.language key at all (today's default config) must behave
    exactly like explicit 'en' — nothing about this change should require a
    config migration for existing English channels."""
    cfg_en["kidsong"].pop("language", None)
    verdict = script_qc.review_script(lyrics.FALLBACK_SONG, cfg_en)
    assert verdict["accept"], _reasons(verdict)


def test_english_broken_song_still_rejected_for_the_same_reasons(cfg_en):
    """The gate must still catch a genuinely broken ENGLISH song — this change
    must not have accidentally loosened English enforcement."""
    broken_song = {
        "title": "Broken Test Song",
        "description": "test",
        "tags": ["a"],
        "characters": "A happy toddler plays outside all afternoon long today.",
        "verses": [
            {
                "lines": [
                    "Purple elephants juggle enormous mathematics textbooks beside sleepy mountain lighthouses",
                    "Silver dolphins compose intricate symphonies inside forgotten crystal libraries endlessly",
                ],
                "scene": "x",
            },
            {
                "lines": [
                    "Golden giraffes analyze peculiar geometry puzzles atop wandering desert caravans",
                    "Emerald tigers assemble curious mechanical contraptions within abandoned volcanic laboratories",
                ],
                "scene": "y",
            },
            {
                "lines": [
                    "Crimson falcons decipher ancient philosophical manuscripts across frozen arctic tundras",
                    "Turquoise octopi construct fantastical architectural monuments beneath turbulent oceanic trenches",
                ],
                "scene": "z",
            },
        ],
    }
    verdict = script_qc.review_script(broken_song, cfg_en)
    assert not verdict["accept"]
    joined = _reasons(verdict)
    assert "line too long" in joined
    assert "no repetition" in joined
    assert "not enough action verbs" in joined
    assert "characters sentence invalid" in joined


# ------------------------------------------------------- tokenizer fix scope --
def test_umlaut_tokenizer_fix_does_not_change_english_word_counts():
    """The unconditional `_words`/`_letters` Unicode fix must be a no-op on
    plain-ASCII English input — no English token contains äöüß, so counts,
    lengths and derived heuristics must be identical to the pre-fix behavior."""
    line = "Clap, clap, clap your hands with me today"
    assert script_qc._words(line) == [
        "Clap", "clap", "clap", "your", "hands", "with", "me", "today",
    ]
    assert script_qc._letters("Clap") == "clap"


def test_umlaut_tokenizer_fix_keeps_german_words_whole():
    assert script_qc._words("für die Hände") == ["für", "die", "Hände"]
    assert script_qc._letters("Füße") == "füße"
