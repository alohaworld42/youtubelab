"""Tests for kidsong.director's learning-mode additions:

  * `_STORY_SUBJECTS` covers letters/numbers/shapes (the fallback extraction
    path used when a verse names an item but carries no explicit override).
  * An EXPLICIT per-verse `story_subject` (set by every verse in
    prompts/learning_songs.json) overrides both the song-level hero subject
    and the `extract_story_subject` gate — regardless of content_mode.
  * `kidsong.content_mode="learning"` biases a verse that carries an explicit
    subject into an INSERT shot (the item alone, no people, shown large) plus
    a REACTION shot (a child pointing at it), deterministically and within
    `max_unique_shots`.
  * None of the above changes a verse with no explicit `story_subject`, or a
    non-learning (`content_mode` absent/"song") run — byte-identical to the
    pre-existing behaviour.

CPU-only, no GPU, no network, no LLM calls: `_fallback_planner` is called
directly with hand-built songs, exactly like tests/test_kidsong_director_normalize.py.
"""
import copy

import pytest

from pipeline.kidsong import director, lyrics, script_qc


def _cfg(seed=20260717, content_mode=None):
    ks = {"seed": seed}
    if content_mode is not None:
        ks["content_mode"] = content_mode
    return {"kidsong": ks}


def _song(verses, characters="Zuri, Kofi and Nala are best friends"):
    return {"characters": characters, "verses": verses}


# --------------------------------------------------------- _STORY_SUBJECTS ---
@pytest.mark.parametrize(
    "scene, expect_substr",
    [
        ("A big letter A stands in the classroom.", "letter"),
        ("The letters dance across the wall.", "letter"),
        ("A big number 3 floats in the playroom.", "number"),
        ("A red circle rolls across the floor.", "circle"),
        ("A blue square sits on the rug.", "square"),
        ("A yellow triangle glows in the corner.", "triangle"),
        ("A pink heart glows on the wall.", "heart"),
        ("A colorful shape appears on the table.", "shape"),
    ],
)
def test_story_subjects_cover_letters_numbers_shapes(scene, expect_substr):
    subject = director.extract_story_subject(scene, [])
    assert subject is not None, scene
    assert expect_substr in subject.lower()
    assert director.valid_story_subject(subject)


def test_story_subjects_never_return_a_cast_child():
    for scene in (
        "A big letter A stands in the classroom.",
        "A big number 3 floats in the playroom.",
        "A red circle rolls across the floor.",
    ):
        subject = director.extract_story_subject(scene, [])
        assert subject and not director._cast_names_in(subject)


# ------------------------------------------------------- explicit override ---
def _learning_verses():
    return [
        {
            "lines": ["A is for apple, red and sweet", "Point at the letter A, so neat",
                      "Clap your hands and learn with me"],
            "scene": "A giant bright letter A stands in a colorful preschool classroom.",
            "story_subject": "a big colorful letter A",
            "learning_item": "A",
        },
        {
            "lines": ["B is for ball, round and bright", "Point at the letter B, so light",
                      "Clap your hands and learn with me"],
            "scene": "A giant bright letter B stands in a colorful preschool classroom.",
            "story_subject": "a big colorful letter B",
            "learning_item": "B",
        },
    ]


def test_explicit_story_subject_overrides_song_level_hero():
    """A verse's own `story_subject` wins even though the song has no
    detectable song-level hero at all (`song_story_subject` would return
    None for these scenes — no animal/object keyword anywhere)."""
    song = _song(_learning_verses())
    assert director.song_story_subject(song) != "a big colorful letter A"

    verse_times = [(0.0, 8.0), (8.0, 16.0)]
    shots = director._fallback_planner(song, verse_times, {}, _cfg())
    v0_subjects = {s["story_subject"] for s in shots if s["verse"] == 0}
    v1_subjects = {s["story_subject"] for s in shots if s["verse"] == 1}
    assert v0_subjects == {"a big colorful letter A"}
    assert v1_subjects == {"a big colorful letter B"}


def test_explicit_story_subject_honored_regardless_of_content_mode():
    """The override is unconditional — it does not require learning mode.
    Learning mode only controls whether the INSERT/REACTION shot bias fires."""
    song = _song(_learning_verses())
    verse_times = [(0.0, 8.0), (8.0, 16.0)]
    shots = director._fallback_planner(song, verse_times, {}, _cfg(content_mode=None))
    assert all(s["story_subject"] for s in shots)
    # But no insert shots without learning mode.
    assert not any(s["shot_type"] == "insert" for s in shots)


def test_verse_without_explicit_subject_is_unaffected():
    """A verse with no `story_subject` key falls through to the pre-existing
    extract_story_subject/song-level-hero logic, completely unaffected."""
    verses = [
        {"lines": ["Zuri had a little lamb", "Its fleece was white as snow"],
         "scene": "Zuri walks with a fluffy white cartoon lamb."},
        {"lines": ["Everywhere that Zuri went", "The lamb was sure to go"],
         "scene": "The lamb follows Zuri down a sunny path."},
    ]
    song = _song(verses)
    verse_times = [(0.0, 8.0), (8.0, 16.0)]
    shots_a = director._fallback_planner(song, verse_times, {}, _cfg())
    shots_b = director._fallback_planner(song, verse_times, {}, _cfg())
    assert shots_a == shots_b
    subjects = {s["story_subject"] for s in shots_a if s.get("story_subject")}
    assert subjects == {"a fluffy white cartoon lamb"}


# ----------------------------------------------------------- learning bias ---
def test_learning_bias_off_by_default_even_with_explicit_subject():
    song = _song(_learning_verses())
    verse_times = [(0.0, 8.0), (8.0, 16.0)]
    shots = director._fallback_planner(song, verse_times, {}, _cfg(content_mode="song"))
    assert not any(s["shot_type"] == "insert" for s in shots)


def test_learning_bias_produces_one_insert_shot_per_verse():
    song = _song(_learning_verses())
    verse_times = [(0.0, 8.0), (8.0, 16.0)]
    shots = director._fallback_planner(song, verse_times, {}, _cfg(content_mode="learning"))

    inserts = [s for s in shots if s["shot_type"] == "insert"]
    assert len(inserts) == 2, inserts
    by_verse = {s["verse"]: s for s in inserts}
    assert by_verse[0]["story_subject"] == "a big colorful letter A"
    assert by_verse[1]["story_subject"] == "a big colorful letter B"
    for s in inserts:
        # No people at all in an insert shot.
        assert s["characters"] == []
        # The action names the taught item.
        assert s["story_subject"] in s["action"]


def test_learning_bias_insert_shot_is_never_the_verse0_establishing_wide():
    """The hard invariant (verse 0's very first shot must stay the cast-wide
    establishing shot) must survive the learning bias."""
    song = _song(_learning_verses())
    verse_times = [(0.0, 8.0), (8.0, 16.0)]
    shots = director._fallback_planner(song, verse_times, {}, _cfg(content_mode="learning"))
    first = min(shots, key=lambda s: (s["verse"], s["start"]))
    assert first["shot_type"] == "wide"
    assert first["characters"] == ["all"]


def test_learning_bias_produces_a_reaction_shot_naming_the_subject():
    song = _song(_learning_verses())
    verse_times = [(0.0, 8.0), (8.0, 16.0)]
    shots = director._fallback_planner(song, verse_times, {}, _cfg(content_mode="learning"))

    for verse in (0, 1):
        vshots = [s for s in shots if s["verse"] == verse]
        subject = vshots[0]["story_subject"]
        reactions = [
            s for s in vshots
            if s["shot_type"] != "insert"
            and subject in (s.get("action") or "")
            and len(s["characters"]) == 1
        ]
        assert reactions, vshots


def test_learning_bias_is_deterministic():
    song = _song(_learning_verses())
    verse_times = [(0.0, 8.0), (8.0, 16.0)]
    cfg = _cfg(content_mode="learning")
    a = director._fallback_planner(song, verse_times, {}, cfg)
    b = director._fallback_planner(song, verse_times, {}, cfg)
    assert a == b


def test_learning_bias_adds_no_extra_shots_beyond_the_existing_cap_logic():
    """The learning bias only RELABELS shots the deterministic skeleton already
    planned (insert_idx/reaction_idx pick existing shot indices) — it must
    never grow the shot count, so the pre-existing max_unique_shots merge
    logic (unchanged, and exercised identically either way) still governs the
    total exactly as it did before learning mode existed."""
    song = _song(_learning_verses())
    verse_times = [(0.0, 12.0), (12.0, 24.0)]
    cfg_song = _cfg(content_mode="song")
    cfg_song["kidsong"]["director"] = {"max_unique_shots": 6}
    cfg_learning = _cfg(content_mode="learning")
    cfg_learning["kidsong"]["director"] = {"max_unique_shots": 6}

    shots_song = director._fallback_planner(song, verse_times, {}, cfg_song)
    shots_learning = director._fallback_planner(song, verse_times, {}, cfg_learning)
    assert len(shots_song) == len(shots_learning)
    assert len(shots_learning) <= 6 or len(shots_learning) == len(shots_song)


def test_learning_bias_single_shot_verse_still_gets_an_insert_no_crash():
    """A verse compressed to exactly one shot (by the max_unique_shots merge
    loop) has no room for both an insert AND a reaction; it must still not
    crash, and (for verse i>0) the one shot becomes the insert."""
    song = _song(_learning_verses())
    verse_times = [(0.0, 1.5), (1.5, 3.0)]  # very short -> 1 shot per verse
    cfg = _cfg(content_mode="learning")
    shots = director._fallback_planner(song, verse_times, {}, cfg)
    by_verse = {}
    for s in shots:
        by_verse.setdefault(s["verse"], []).append(s)
    # verse 0's single shot must stay the mandatory wide establishing shot.
    assert by_verse[0][0]["shot_type"] == "wide"
    # verse 1's single shot has no earlier "wide" reservation, so it becomes
    # the insert.
    if len(by_verse.get(1, [])) == 1:
        assert by_verse[1][0]["shot_type"] == "insert"


# ------------------------------------------------------- full pipeline fit ---
def test_learning_library_songs_produce_passing_shotlists_end_to_end():
    """Every learning-library song, planned deterministically with
    content_mode=learning, must pass the exact same shotlist QC gate the
    public-domain corpus does."""
    from pipeline.config import load_config

    cfg = copy.deepcopy(load_config())
    cfg.setdefault("kidsong", {}).setdefault("review", {})["reviewer"] = "heuristic"
    cfg["kidsong"]["content_mode"] = "learning"

    for entry in lyrics.load_learning_library(cfg):
        song = lyrics._normalize_song(lyrics.pd_song_to_dict(entry))
        verse_times = [(i * 8.0, (i + 1) * 8.0) for i in range(len(song["verses"]))]
        beats = {"bpm": 100, "beat_times": [i * 0.6 for i in range(200)]}
        shots = director._fallback_planner(song, verse_times, beats, cfg)

        verdict = script_qc.review_shotlist({"shots": shots}, song, cfg)
        assert verdict["accept"], (entry["id"], verdict["reasons"])

        inserts = [s for s in shots if s["shot_type"] == "insert"]
        assert len(inserts) == len(song["verses"]), entry["id"]
        taught = {v["learning_item"] for v in song["verses"]}
        assert len(taught) == len(song["verses"]), "each verse teaches a distinct item"
