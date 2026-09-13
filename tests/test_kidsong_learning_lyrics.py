"""Tests for the learning-mode lyric library (prompts/learning_songs.json) and
its selection path in pipeline.kidsong.lyrics.

CPU-only, no GPU, no network, no LLM calls. Mirrors tests/test_kidsong_pd_lyrics.py
(the public-domain library's own test file) closely, since kidsong.content_mode
="learning" reuses the exact same selection/rotation machinery
(`choose_pd_song`, generalized to take a `songs=` library) and the exact same
pipeline-shape conversion (`pd_song_to_dict`) as kidsong.lyrics_source=
"public_domain" — the only genuinely new things are the library file itself,
the per-verse `story_subject`/`learning_item` fields it carries, and the
content_mode dispatch in `generate_song`.
"""
import copy
import json

import pytest

from pipeline.kidsong import lyrics, script_qc


@pytest.fixture()
def cfg():
    from pipeline.config import load_config

    c = copy.deepcopy(load_config())
    c.setdefault("kidsong", {}).setdefault("review", {})["reviewer"] = "heuristic"
    return c


def _reasons(verdict):
    return " | ".join(verdict["reasons"])


# ----------------------------------------------------------------- library ---
def test_learning_library_loads_with_three_songs(cfg):
    songs = lyrics.load_learning_library(cfg)
    assert len(songs) == 3, f"library has {len(songs)} songs, want 3"
    assert {s["id"] for s in songs} == {
        "learning_abc_time", "learning_counting_time", "learning_shapes_time",
    }


def test_learning_library_is_independent_of_the_pd_library(cfg):
    """Loading one library must not disturb the other — they are cached by
    their own (distinct) file path."""
    pd_ids = {s["id"] for s in lyrics.load_pd_library(cfg)}
    learning_ids = {s["id"] for s in lyrics.load_learning_library(cfg)}
    assert pd_ids.isdisjoint(learning_ids)
    assert len(pd_ids) >= 8  # the pd library is unaffected by the new file


def test_every_learning_verse_carries_story_subject_and_learning_item(cfg):
    for song in lyrics.load_learning_library(cfg):
        assert 3 <= len(song["verses"]) <= 4, song["id"]
        for v in song["verses"]:
            assert str(v.get("story_subject") or "").strip(), (song["id"], v)
            assert str(v.get("learning_item") or "").strip(), (song["id"], v)
            assert str(v.get("scene_hint") or "").strip(), (song["id"], v)


def test_every_learning_song_passes_the_full_script_gate(cfg):
    for entry in lyrics.load_learning_library(cfg):
        song = lyrics._normalize_song(lyrics.pd_song_to_dict(entry))
        verdict = script_qc.review_script(song, cfg)
        assert verdict["accept"], f"{entry['id']} rejected: {_reasons(verdict)}"


def test_learning_fallback_song_passes_the_full_script_gate(cfg):
    song = json.loads(json.dumps(lyrics.LEARNING_FALLBACK_SONG))
    verdict = script_qc.review_script(lyrics._normalize_song(song), cfg)
    assert verdict["accept"], _reasons(verdict)


# --------------------------------------------------------- field preservation
def test_pd_song_to_dict_preserves_story_subject_and_learning_item():
    entry = next(
        s for s in lyrics.load_learning_library() if s["id"] == "learning_abc_time"
    )
    song = lyrics.pd_song_to_dict(entry)
    for v, src in zip(song["verses"], entry["verses"]):
        assert v["story_subject"] == src["story_subject"]
        assert v["learning_item"] == src["learning_item"]
        assert v["scene"] == src["scene_hint"]


def test_normalize_verses_preserves_story_subject_and_learning_item():
    raw = [
        {"lines": ["a a", "b b"], "scene": "s1", "story_subject": "a big red circle", "learning_item": "circle"},
        {"lines": ["c c", "d d"], "scene": "s2"},
        {"lines": ["e e", "f f"], "scene": "s3", "story_subject": "   ", "learning_item": ""},
    ]
    out = lyrics._normalize_verses(raw)
    assert out[0] == {
        "lines": ["a a", "b b"], "scene": "s1",
        "story_subject": "a big red circle", "learning_item": "circle",
    }
    # No fields supplied -> no fields added.
    assert out[1] == {"lines": ["c c", "d d"], "scene": "s2"}
    # Blank fields are dropped, not carried through as whitespace.
    assert out[2] == {"lines": ["e e", "f f"], "scene": "s3"}


def test_pd_songs_verses_are_byte_identical_no_new_keys(cfg):
    """The public-domain library's own verses must produce the EXACT same
    two-key {lines, scene} dict as before story_subject/learning_item existed
    — this feature is additive-only."""
    for entry in lyrics.load_pd_library(cfg):
        song = lyrics.pd_song_to_dict(entry)
        for v in song["verses"]:
            assert set(v.keys()) == {"lines", "scene"}, (entry["id"], v)


# ---------------------------------------------------------------- selection --
def test_choose_learning_song_topic_match(cfg):
    songs = lyrics.load_learning_library(cfg)
    for topic, expected in [
        ("learning letters", "learning_abc_time"),
        ("counting numbers", "learning_counting_time"),
        ("shapes and colors", "learning_shapes_time"),
    ]:
        entry, why = lyrics.choose_learning_song(topic, cfg)
        assert entry["id"] == expected, f"{topic!r} -> {entry['id']} ({why})"


def test_choose_learning_song_rotation_never_repeats_the_previous_episode(cfg):
    songs = lyrics.load_learning_library(cfg)
    recent = []
    for _ in range(len(songs) + 2):
        entry, _why = lyrics.choose_pd_song(None, cfg, songs=songs, recent_ids=recent)
        assert entry["id"] != (recent[0] if recent else None), "back-to-back repeat"
        recent.insert(0, entry["id"])
    assert len(set(recent[-len(songs):])) == len(songs)


def test_choose_learning_song_is_deterministic(cfg):
    songs = lyrics.load_learning_library(cfg)
    picks = {
        lyrics.choose_pd_song("letters", cfg, songs=songs, recent_ids=["learning_abc_time"])[0]["id"]
        for _ in range(5)
    }
    assert len(picks) == 1


# ------------------------------------------------------------ content_mode --
def test_content_mode_learning_returns_a_learning_library_song(cfg, monkeypatch):
    cfg["kidsong"]["content_mode"] = "learning"

    def boom(*a, **k):  # pragma: no cover - must never run
        raise AssertionError("learning content_mode must not touch the LLM")

    monkeypatch.setattr(lyrics, "_generate_llm_song", boom)
    ids = {s["id"] for s in lyrics.load_learning_library(cfg)}
    song = lyrics.generate_song(None, cfg)
    assert song.get("pd_song_id") in ids
    for v in song["verses"]:
        assert v.get("story_subject")
        assert v.get("learning_item")


def test_content_mode_learning_falls_back_to_the_builtin_song_when_library_empty(cfg, monkeypatch):
    cfg["kidsong"]["content_mode"] = "learning"
    monkeypatch.setattr(lyrics, "load_pd_library", lambda c=None, filename=None: [])

    def boom(*a, **k):  # pragma: no cover - must never run
        raise AssertionError("must not fall through to the LLM")

    monkeypatch.setattr(lyrics, "_generate_llm_song", boom)
    song = lyrics.generate_song(None, cfg)
    assert song["title"] == lyrics.LEARNING_FALLBACK_SONG["title"]
    assert "pd_song_id" not in song
    for v in song["verses"]:
        assert v.get("story_subject")


def test_content_mode_default_is_song_and_fully_unaffected_by_learning_feature(cfg, monkeypatch):
    """`kidsong.content_mode` absent entirely -> behaves exactly like before
    this feature existed: normal public-domain song selection, no
    story_subject/learning_item anywhere."""
    cfg["kidsong"].pop("content_mode", None)
    # This test is about the content_mode dispatch, not the lyrics_source one:
    # pin public_domain so "song mode picks a library song" is a clean assertion
    # independent of the (now default) topical-LLM auto path.
    cfg["kidsong"]["lyrics_source"] = "public_domain"

    def boom(*a, **k):  # pragma: no cover - must never run
        raise AssertionError("song mode must not touch the learning library")

    monkeypatch.setattr(lyrics, "generate_learning_song", boom)
    song = lyrics.generate_song("stars", cfg)
    pd_ids = {s["id"] for s in lyrics.load_pd_library(cfg)}
    assert song.get("pd_song_id") in pd_ids
    for v in song["verses"]:
        assert "story_subject" not in v
        assert "learning_item" not in v


def test_content_mode_song_explicit_matches_default(cfg, monkeypatch):
    cfg["kidsong"]["content_mode"] = "song"
    cfg["kidsong"]["lyrics_source"] = "public_domain"

    def boom(*a, **k):  # pragma: no cover - must never run
        raise AssertionError("song mode must not touch the learning library")

    monkeypatch.setattr(lyrics, "generate_learning_song", boom)
    song = lyrics.generate_song("stars", cfg)
    pd_ids = {s["id"] for s in lyrics.load_pd_library(cfg)}
    assert song.get("pd_song_id") in pd_ids
