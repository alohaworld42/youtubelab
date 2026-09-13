"""Public-domain lyric library + the lyric-craft gates that replaced the LLM's
bad songwriting.

Two things are locked down here:
  1. The library in prompts/pd_songs.json is legally and structurally sound, and
     every entry passes the FULL script gate unchanged.
  2. The new lyric-craft gates reject the two real songs the LLM path actually
     shipped (both reproduced verbatim below from output/*-song.json) — that is
     the regression proof — while still accepting a library song.
"""
import copy
import json

import pytest

from pipeline.kidsong import lyrics, script_qc
from pipeline.kidsong.sing import build_lyrics, verse_times_from_words


@pytest.fixture()
def cfg():
    from pipeline.config import load_config

    c = copy.deepcopy(load_config())
    c.setdefault("kidsong", {}).setdefault("review", {})["reviewer"] = "heuristic"
    return c


# ---------------------------------------------------------------- fixtures ---
# Verbatim copies of the two songs the user flagged as unusable. Reproduced here
# rather than read from output/ so the regression stands on its own even after
# the generated media is archived.
BAD_SONG_WATERING = {
    "title": "Zuri's Watering Blooms Day",
    "description": "Join Zuri, Kofi, and Nala as they care for their beautiful flowers!",
    "tags": ["kids song", "nursery rhyme"],
    "characters": "Three adorable Black toddlers: Zuri, Kofi and Nala, in their established looks.",
    "verses": [
        {
            "lines": [
                "Water, water, everywhere!",
                "Zuri's flowers need it to grow with care.",
                "Kofi and Nala too can help us share!",
            ],
            "scene": "Zuri stands by the garden tub, pouring water on the flowers.",
        },
        {
            "lines": [
                "We'll use our cups to pour with fun!",
                "Every flower will get some water soon.",
                "Nala's helping now, everyone!",
            ],
            "scene": "Kofi and Zuri hand each other small watering cans.",
        },
        {
            "lines": [
                "We'll make our flowers happy too!",
                "Watering them is fun to do.",
                "Zuri's helping Kofi and Nala!",
            ],
            "scene": "All three toddlers are gathered around the garden, clapping.",
        },
    ],
}

BAD_SONG_BREEZE = {
    "title": "Zuri, Kofi & Nala Feel the Breeze",
    "description": "Join Zuri, Kofi and Nala outside on a sunny day!",
    "tags": ["kids song", "nursery rhyme"],
    "characters": "Three adorable Black toddlers: Zuri, Kofi and Nala, in their established looks.",
    "verses": [
        {
            "lines": ["Let's go outside, feel the breeze", "Blow bubbles high up high"],
            "scene": "Zuri holds a bubble wand, Kofi stomps in a puddle.",
        },
        {
            "lines": ["Wave your arms, stomp your feet", "Dance to the wind's sweet beat"],
            "scene": "All three children dance and play together on a green grassy field.",
        },
        {
            "lines": ["Clap hands high up in the air", "Twirl around without a single care"],
            "scene": "Zuri claps, Kofi twirls, Nala laughs as they spin around each other.",
        },
    ],
}


def _reasons(verdict):
    return " | ".join(verdict["reasons"])


# ----------------------------------------------------------------- library ---
def test_library_loads_with_enough_songs(cfg):
    songs = lyrics.load_pd_library(cfg)
    assert 8 <= len(songs) <= 12, f"library has {len(songs)} songs, want 8-12"
    assert len({s["id"] for s in songs}) == len(songs), "duplicate song ids"


def test_every_library_entry_documents_its_public_domain_status(cfg):
    for s in lyrics.load_pd_library(cfg):
        src = s.get("source") or {}
        for key in ("author", "year", "publication", "public_domain", "adaptation"):
            assert str(src.get(key) or "").strip(), f"{s['id']}: source.{key} is missing"
        # Nothing may be sourced to a publication we cannot show is out of
        # copyright in the US: the pre-1929 expired-term rule is the line.
        assert int(src["year"]) < 1929, f"{s['id']}: source year {src['year']} is not pre-1929"


def test_every_library_entry_has_the_episode_canon_keys(cfg):
    """EPISODE CANON (2026-07-24): every entry declares ONE fixed look for its
    non-child story subject (`subject_description`, null for a song with no
    single recurring subject — e.g. "skip to my lou") and ONE fixed setting
    for the whole episode (`canonical_setting`, never null/empty) — see
    `director.plan_verse_locations` and `generate._canonical_subject_clause`,
    which both rely on these being present for every song."""
    for s in lyrics.load_pd_library(cfg):
        assert "subject_description" in s, f"{s['id']}: missing subject_description"
        desc = s["subject_description"]
        assert desc is None or (isinstance(desc, str) and desc.strip()), (
            f"{s['id']}: subject_description must be null or a non-empty string"
        )
        setting = s.get("canonical_setting")
        assert isinstance(setting, str) and setting.strip(), (
            f"{s['id']}: canonical_setting must be a non-empty string"
        )


def test_german_library_entries_also_have_the_episode_canon_keys(cfg):
    """prompts/pd_songs.de.json is the German counterpart library and must
    carry the same two ENGLISH-language fields (they feed the English-trained
    image/video model, exactly like `scene_hint` — see the file's own note)."""
    de_cfg = copy.deepcopy(cfg)
    de_cfg.setdefault("kidsong", {})["language"] = "de"
    songs = lyrics.load_pd_library(de_cfg)
    assert len(songs) >= 8
    for s in songs:
        assert "subject_description" in s, f"{s['id']}: missing subject_description"
        desc = s["subject_description"]
        assert desc is None or (isinstance(desc, str) and desc.strip()), (
            f"{s['id']}: subject_description must be null or a non-empty string"
        )
        setting = s.get("canonical_setting")
        assert isinstance(setting, str) and setting.strip(), (
            f"{s['id']}: canonical_setting must be a non-empty string"
        )


def test_pd_song_to_dict_stamps_subject_description_and_canonical_setting(cfg):
    """`pd_song_to_dict` — the exact function that stamps `pd_song_id` — must
    also stamp both episode-canon fields onto the pipeline-shaped song dict,
    None-safe either way."""
    songs = lyrics.load_pd_library(cfg)
    sheep_entry = next(s for s in songs if s["id"] == "baa_baa_black_sheep")
    lou_entry = next(s for s in songs if s["id"] == "skip_to_my_lou")

    sheep_song = lyrics.pd_song_to_dict(sheep_entry)
    assert sheep_song["subject_description"] == sheep_entry["subject_description"]
    assert sheep_song["canonical_setting"] == sheep_entry["canonical_setting"]

    lou_song = lyrics.pd_song_to_dict(lou_entry)
    assert lou_song["subject_description"] is None
    assert lou_song["canonical_setting"] == lou_entry["canonical_setting"]

    # None-safe: an entry that predates these keys entirely (e.g. every
    # learning_songs.json entry) must not crash and must stamp None/absent
    # rather than raise.
    legacy_entry = {k: v for k, v in sheep_entry.items()
                    if k not in ("subject_description", "canonical_setting")}
    legacy_song = lyrics.pd_song_to_dict(legacy_entry)
    assert legacy_song["subject_description"] is None
    assert legacy_song["canonical_setting"] is None


def test_library_excludes_the_songs_that_cannot_be_dated(cfg):
    """"The Wheels on the Bus" (1939) and "If You're Happy and You Know It"
    (undated 20th century) must never be in the library."""
    blob = json.dumps(lyrics.load_pd_library(cfg)).lower()
    for banned in ("wheels on the bus", "if you're happy"):
        assert banned not in blob


def test_every_library_song_passes_the_full_script_gate(cfg):
    for entry in lyrics.load_pd_library(cfg):
        song = lyrics._normalize_song(lyrics.pd_song_to_dict(entry))
        verdict = script_qc.review_script(song, cfg)
        assert verdict["accept"], f"{entry['id']} rejected: {_reasons(verdict)}"


# -------------------------------------------------------------- pipeline fit -
def test_pd_song_has_the_shape_the_pipeline_expects(cfg):
    cfg["kidsong"]["lyrics_source"] = "public_domain"
    song = lyrics.generate_song("stars", cfg)
    assert set(("title", "description", "tags", "characters", "verses")) <= set(song)
    assert isinstance(song["tags"], list) and song["tags"]
    assert "Zuri" in song["characters"]
    assert 3 <= len(song["verses"]) <= 5
    for v in song["verses"]:
        assert 2 <= len(v["lines"]) <= 4
        assert all(isinstance(l, str) and l.strip() for l in v["lines"])
        assert v["scene"].strip(), "every verse needs a scene for the director"


def test_pd_song_drives_sing_and_verse_timing(cfg):
    cfg["kidsong"]["lyrics_source"] = "public_domain"
    song = lyrics.generate_song("rainy day puddle jumping", cfg)

    tagged = build_lyrics(song["verses"])
    assert "[Verse 1]" in tagged
    assert "[Chorus]" in tagged, "the repeated hook verse should be tagged as a chorus"

    words, t = [], 0.0
    for v in song["verses"]:
        for line in v["lines"]:
            for w in line.split():
                words.append({"word": w, "start": t, "end": t + 0.35})
                t += 0.35
    times = verse_times_from_words(words, song["verses"], t)
    assert len(times) == len(song["verses"])
    assert times[0][0] == 0.0 and times[-1][1] == pytest.approx(t)
    assert all(end > start for start, end in times)
    assert all(b[0] >= a[1] for a, b in zip(times, times[1:])), "verse windows must not overlap"


# ---------------------------------------------------------------- selection --
def test_topic_maps_to_the_obvious_library_song(cfg):
    songs = lyrics.load_pd_library(cfg)
    for topic, expected in [
        ("rainy day puddle jumping", "rain_rain_go_away"),
        ("washing hands before we eat", "here_we_go_round_the_mulberry_bush"),
        ("stars", "twinkle_twinkle_little_star"),
    ]:
        entry, why = lyrics.choose_pd_song(topic, cfg, songs=songs, recent_ids=[])
        assert entry["id"] == expected, f"{topic!r} -> {entry['id']} ({why})"
        assert topic in why, "the mapping has to be logged in plain words"


def test_rotation_never_repeats_the_previous_episode(cfg):
    songs = lyrics.load_pd_library(cfg)
    recent = []
    for _ in range(len(songs) + 2):
        entry, _why = lyrics.choose_pd_song(None, cfg, songs=songs, recent_ids=recent)
        assert entry["id"] != (recent[0] if recent else None), "back-to-back repeat"
        recent.insert(0, entry["id"])
    # Rotation is exhaustive: every song is used before any is used twice.
    assert len(set(recent[-len(songs):])) == len(songs)


def test_topic_match_yields_when_it_would_repeat_the_last_episode(cfg):
    songs = lyrics.load_pd_library(cfg)
    entry, why = lyrics.choose_pd_song(
        "farm animals", cfg, songs=songs, recent_ids=["mary_had_a_little_lamb"]
    )
    assert entry["id"] != "mary_had_a_little_lamb", why


def test_selection_is_deterministic(cfg):
    songs = lyrics.load_pd_library(cfg)
    picks = {
        lyrics.choose_pd_song("bedtime", cfg, songs=songs, recent_ids=["baa_baa_black_sheep"])[0]["id"]
        for _ in range(5)
    }
    assert len(picks) == 1


# ------------------------------------------------------------ source switch --
# A songwriter-quality, ON-TOPIC song that clears the FULL craft gate. Used to
# stand in for a "good" LLM reply so the auto/llm paths can be exercised without
# the network. Its topic (a balance bike) is deliberately NOTHING like any
# library nursery rhyme, so "the returned song is THIS one" proves the topical
# LLM song shipped rather than an unrelated library rhyme.
GOOD_TOPICAL_SONG = {
    "title": "Ride My Balance Bike - A Happy Sing-Along for Toddlers",
    "description": (
        "Zuri, Kofi and Nala roll along on their balance bikes! "
        "#kidssong #balancebike #toddlersongs #singalong #kidsmusic"
    ),
    "tags": [
        "kids song", "balance bike", "toddler song", "sing along", "learning song",
        "preschool", "kids music", "children song",
    ],
    "characters": lyrics._fallback_characters(),
    "verses": [
        {
            "lines": [
                "Roll along, my bike and me",
                "Push my feet and set them free",
                "Roll along and sing with me",
            ],
            "scene": "The three toddlers roll their balance bikes along a sunny park path.",
        },
        {
            "lines": [
                "Wave hello as friends go by",
                "Big bright smiles up to the sky",
                "Roll along and sing with me",
            ],
            "scene": "The three toddlers wave to each other while riding around the park.",
        },
        {
            "lines": [
                "See the wheels go spinning fast",
                "Down the path we roll at last",
                "Roll along and sing with me",
            ],
            "scene": "The three toddlers roll their bikes down a gentle grassy slope.",
        },
    ],
}


def _passing_llm():
    """Build a stub `_generate_llm_song` that returns GOOD_TOPICAL_SONG and
    records every call (with the feedback it received)."""
    calls = []

    def fake(topic, c, feedback=None):
        calls.append({"topic": topic, "feedback": feedback})
        return copy.deepcopy(GOOD_TOPICAL_SONG)

    return fake, calls


def test_good_topical_song_actually_clears_the_gate(cfg):
    """Precondition for the auto/llm proofs below: the stand-in "good" LLM reply
    is genuinely gate-passing, so a test that ships it is exercising the real
    accept path, not a hole in the gate."""
    verdict = script_qc.review_script(GOOD_TOPICAL_SONG, cfg)
    assert verdict["accept"] is True, _reasons(verdict)


def test_lyrics_source_llm_still_calls_the_llm_path(cfg, monkeypatch):
    cfg["kidsong"]["lyrics_source"] = "llm"
    fake, calls = _passing_llm()
    monkeypatch.setattr(lyrics, "_generate_llm_song", fake)
    song = lyrics.generate_song("counting", cfg)
    assert calls and calls[0]["topic"] == "counting"
    assert song["title"] == GOOD_TOPICAL_SONG["title"]
    assert "pd_song_id" not in song


def test_lyrics_source_llm_ships_best_effort_and_never_falls_back_to_library(cfg, monkeypatch):
    """`llm` has NO library net: even when every attempt fails the gate, the
    best LLM attempt ships (pd_song_id absent) rather than a library rhyme."""
    cfg["kidsong"]["lyrics_source"] = "llm"

    def always_bad(topic, c, feedback=None):
        return copy.deepcopy(BAD_SONG_BREEZE)

    monkeypatch.setattr(lyrics, "_generate_llm_song", always_bad)

    def boom(*a, **k):  # pragma: no cover - must never run
        raise AssertionError("lyrics_source=llm must not consult the library")

    monkeypatch.setattr(lyrics, "generate_pd_song", boom)
    song = lyrics.generate_song("feeling the breeze", cfg)
    assert song["title"] == BAD_SONG_BREEZE["title"]
    assert "pd_song_id" not in song


def test_lyrics_source_public_domain_never_calls_the_llm(cfg, monkeypatch):
    cfg["kidsong"]["lyrics_source"] = "public_domain"

    def boom(*a, **k):  # pragma: no cover - must never run
        raise AssertionError("the public-domain path must not touch the LLM")

    monkeypatch.setattr(lyrics, "_generate_llm_song", boom)
    # Which song comes back depends on the episode history on disk; that it is a
    # library song at all is the point.
    ids = {s["id"] for s in lyrics.load_pd_library(cfg)}
    assert lyrics.generate_song("stars", cfg)["pd_song_id"] in ids


# ------------------------------------------------------------- auto dispatch --
def test_lyrics_source_auto_ships_the_topical_llm_song_when_it_passes(cfg, monkeypatch):
    """The whole point of the coherence fix: a topical, gate-passing LLM song
    ships as-is — the library is never consulted."""
    cfg["kidsong"]["lyrics_source"] = "auto"
    fake, calls = _passing_llm()
    monkeypatch.setattr(lyrics, "_generate_llm_song", fake)

    def boom(*a, **k):  # pragma: no cover - must never run
        raise AssertionError("a passing LLM song must not trigger the library fallback")

    monkeypatch.setattr(lyrics, "generate_pd_song", boom)
    song = lyrics.generate_song("learning to ride a balance bike", cfg)
    assert song["title"] == GOOD_TOPICAL_SONG["title"]
    assert "pd_song_id" not in song
    assert len(calls) == 1  # passed on the first attempt, no needless retries


def test_lyrics_source_auto_falls_back_to_the_library_when_the_llm_stays_bad(cfg, monkeypatch):
    """When the LLM cannot produce a passing song, the best library MATCH ships
    instead — a bad song never reaches the render, and the episode still runs."""
    cfg["kidsong"]["lyrics_source"] = "auto"
    cfg["kidsong"]["lyrics_llm_attempts"] = 2
    attempts = []

    def always_bad(topic, c, feedback=None):
        attempts.append(feedback)
        return copy.deepcopy(BAD_SONG_WATERING)

    monkeypatch.setattr(lyrics, "_generate_llm_song", always_bad)
    song = lyrics.generate_song("rainy day puddle jumping", cfg)
    # The library caught it: a real pd song, not the rejected LLM lyrics.
    ids = {s["id"] for s in lyrics.load_pd_library(cfg)}
    assert song.get("pd_song_id") in ids
    assert song["title"] != BAD_SONG_WATERING["title"]
    # It really did retry, and fed the rejection reasons back on the 2nd attempt.
    assert len(attempts) == 2
    assert attempts[0] is None and attempts[1], "the 2nd attempt must carry feedback"


def test_lyrics_source_auto_retries_with_feedback_then_ships_the_fixed_song(cfg, monkeypatch):
    """A first bad attempt, then a good one: the good song ships and the retry
    was actually driven by the gate's rejection reasons."""
    cfg["kidsong"]["lyrics_source"] = "auto"
    seen = []
    replies = [copy.deepcopy(BAD_SONG_BREEZE), copy.deepcopy(GOOD_TOPICAL_SONG)]

    def improving(topic, c, feedback=None):
        seen.append(feedback)
        return replies[min(len(seen) - 1, len(replies) - 1)]

    monkeypatch.setattr(lyrics, "_generate_llm_song", improving)

    def boom(*a, **k):  # pragma: no cover - must never run
        raise AssertionError("a passing retry must not fall back to the library")

    monkeypatch.setattr(lyrics, "generate_pd_song", boom)
    song = lyrics.generate_song("feeling happy like a butterfly", cfg)
    assert song["title"] == GOOD_TOPICAL_SONG["title"]
    assert len(seen) == 2
    assert seen[0] is None and seen[1], "the 2nd attempt must carry the rejection reasons"


def test_lyrics_source_auto_ships_the_llm_song_when_the_library_is_empty(cfg, monkeypatch):
    """Auto always tries the topical LLM first; if it passes there is nothing to
    fall back to and no library is needed anyway."""
    cfg["kidsong"]["lyrics_source"] = "auto"
    monkeypatch.setattr(lyrics, "load_pd_library", lambda c=None: [])
    fake, _calls = _passing_llm()
    monkeypatch.setattr(lyrics, "_generate_llm_song", fake)
    song = lyrics.generate_song(None, cfg)
    assert song["title"] == GOOD_TOPICAL_SONG["title"]
    assert "pd_song_id" not in song


def test_public_domain_mode_uses_the_builtin_song_when_the_library_is_empty(cfg, monkeypatch):
    """Explicitly asking for public-domain words must not silently switch to the
    very LLM output this mode exists to avoid."""
    cfg["kidsong"]["lyrics_source"] = "public_domain"
    monkeypatch.setattr(lyrics, "load_pd_library", lambda c=None: [])

    def boom(*a, **k):  # pragma: no cover - must never run
        raise AssertionError("must not fall through to the LLM")

    monkeypatch.setattr(lyrics, "_generate_llm_song", boom)
    assert lyrics.generate_song(None, cfg)["title"] == lyrics.FALLBACK_SONG["title"]


def test_default_source_is_auto(cfg, monkeypatch):
    """With no `lyrics_source` set, generate_song uses the auto path: a passing
    topical LLM song ships (the old default would have ignored the LLM entirely
    and returned a library rhyme)."""
    cfg["kidsong"].pop("lyrics_source", None)
    fake, calls = _passing_llm()
    monkeypatch.setattr(lyrics, "_generate_llm_song", fake)
    song = lyrics.generate_song("stars", cfg)
    assert calls, "the default path must reach the LLM"
    assert song["title"] == GOOD_TOPICAL_SONG["title"]
    assert "pd_song_id" not in song


def test_module_default_constant_is_auto():
    assert lyrics.DEFAULT_LYRICS_SOURCE == lyrics.LYRICS_SOURCE_AUTO == "auto"


# ---------------------------------------------------- topical floor ----------
# When no attempt fully passes the gate, auto-mode ships the best topical attempt
# (rather than an off-topic library rhyme) if it clears `lyrics_topical_floor`.
# These isolate that ROUTING from script_qc's scoring by stubbing the verdict.
def _topical_llm(verdict):
    """Stub `_generate_llm_song` (a topical song, title distinct from any library
    rhyme) plus a controlled `_craft_verdict`, so the auto path's floor routing
    is exercised independently of the real craft scoring."""
    song = copy.deepcopy(GOOD_TOPICAL_SONG)
    song["title"] = "A Topical Song About The Actual Topic"

    def fake_llm(topic, c, feedback=None):
        return copy.deepcopy(song)

    def fake_verdict(s, c):
        return dict(verdict)

    return song, fake_llm, fake_verdict


def test_auto_ships_topical_song_when_it_clears_the_floor(cfg, monkeypatch):
    """No attempt fully passes, but the best topical attempt scores >= the floor:
    it ships in preference to an off-topic library song (coherence first)."""
    cfg["kidsong"]["lyrics_source"] = "auto"
    cfg["kidsong"]["lyrics_llm_attempts"] = 1
    cfg["kidsong"]["lyrics_topical_floor"] = 0.6
    song, fake_llm, fake_verdict = _topical_llm(
        {"accept": False, "score": 0.7, "reasons": ["one verse doesn't rhyme"]}
    )
    monkeypatch.setattr(lyrics, "_generate_llm_song", fake_llm)
    monkeypatch.setattr(lyrics, "_craft_verdict", fake_verdict)

    def boom(*a, **k):  # pragma: no cover - must never run
        raise AssertionError("a topical song above the floor must not hit the library")

    monkeypatch.setattr(lyrics, "generate_pd_song", boom)
    out = lyrics.generate_song("sunflower family tree", cfg)
    assert out["title"] == song["title"]
    assert "pd_song_id" not in out
    # the private routing flags must never leak into the shipped song
    assert "_backend_dead" not in out and "_generic_fallback" not in out


def test_auto_falls_to_library_when_best_topical_song_is_below_the_floor(cfg, monkeypatch):
    """A genuinely unsingable attempt (below the floor) still falls to the
    topic-matched library — the floor is not a licence to ship garbage."""
    cfg["kidsong"]["lyrics_source"] = "auto"
    cfg["kidsong"]["lyrics_llm_attempts"] = 1
    cfg["kidsong"]["lyrics_topical_floor"] = 0.6
    song, fake_llm, fake_verdict = _topical_llm(
        {"accept": False, "score": 0.5, "reasons": ["no hook", "no rhyme"]}
    )
    monkeypatch.setattr(lyrics, "_generate_llm_song", fake_llm)
    monkeypatch.setattr(lyrics, "_craft_verdict", fake_verdict)
    out = lyrics.generate_song("rainy day puddle jumping", cfg)
    ids = {s["id"] for s in lyrics.load_pd_library(cfg)}
    assert out.get("pd_song_id") in ids
    assert out["title"] != song["title"]


def test_topical_floor_of_one_restores_the_strict_gate(cfg, monkeypatch):
    """Floor >= 1.0 disables the floor: only a fully-passing song ships, else the
    library — the pre-floor behaviour, opt-in via config."""
    cfg["kidsong"]["lyrics_source"] = "auto"
    cfg["kidsong"]["lyrics_llm_attempts"] = 1
    cfg["kidsong"]["lyrics_topical_floor"] = 1.0
    song, fake_llm, fake_verdict = _topical_llm(
        {"accept": False, "score": 0.7, "reasons": ["one verse doesn't rhyme"]}
    )
    monkeypatch.setattr(lyrics, "_generate_llm_song", fake_llm)
    monkeypatch.setattr(lyrics, "_craft_verdict", fake_verdict)
    out = lyrics.generate_song("counting stars", cfg)
    ids = {s["id"] for s in lyrics.load_pd_library(cfg)}
    assert out.get("pd_song_id") in ids  # strict: the 0.7 song did NOT ship
    assert out["title"] != song["title"]


def test_topical_floor_default_and_clamp():
    assert lyrics._topical_floor({}) == lyrics.DEFAULT_TOPICAL_FLOOR
    assert lyrics._topical_floor({"kidsong": {"lyrics_topical_floor": 2.0}}) == 1.0
    assert lyrics._topical_floor({"kidsong": {"lyrics_topical_floor": -1}}) == 0.0
    assert (
        lyrics._topical_floor({"kidsong": {"lyrics_topical_floor": "junk"}})
        == lyrics.DEFAULT_TOPICAL_FLOOR
    )


# ------------------------------------------------- lyric-craft regressions ---
def test_gate_rejects_the_real_watering_song(cfg):
    """output/20260720-110129-kidsong-zuris-watering-blooms-day-song.json"""
    verdict = script_qc.review_script(BAD_SONG_WATERING, cfg)
    assert verdict["accept"] is False
    assert verdict["retry_hints"]["regenerate"] is True
    reasons = _reasons(verdict)
    # 8 / 11 / 8 syllables in "We'll use our cups to pour with fun!" ...
    assert "meter inconsistent" in reasons
    # ... "fun" / "soon" / "everyone" do not rhyme ...
    assert "no rhyme" in reasons
    # ... nothing repeats across verses ...
    assert "no hook" in reasons
    # ... and 4 of its 9 lines name a child.
    assert "character names overused" in reasons


def test_gate_rejects_the_real_breeze_song(cfg):
    """output/20260720-170942-kidsong-zuri-kofi-nala-feel-the-breeze-song.json"""
    verdict = script_qc.review_script(BAD_SONG_BREEZE, cfg)
    assert verdict["accept"] is False
    assert verdict["retry_hints"]["regenerate"] is True
    reasons = _reasons(verdict)
    assert "no rhyme" in reasons          # "breeze" / "high"
    assert "clumsy phrasing" in reasons   # "Blow bubbles high up high"
    assert "no hook" in reasons


def test_gate_accepts_a_library_song(cfg):
    cfg["kidsong"]["lyrics_source"] = "public_domain"
    song = lyrics.generate_song("stars", cfg)
    verdict = script_qc.review_script(song, cfg)
    assert verdict["accept"] is True, _reasons(verdict)
    assert verdict["score"] == 1.0


def test_gate_still_accepts_the_builtin_fallback_song(cfg):
    song = json.loads(json.dumps(lyrics.FALLBACK_SONG))
    verdict = script_qc.review_script(song, cfg)
    assert verdict["accept"] is True, _reasons(verdict)


# --------------------------------------------------- lyric-craft primitives --
@pytest.mark.parametrize(
    "line, expected",
    [
        ("Twinkle, twinkle, little star", 7),
        ("How I wonder what you are", 7),
        ("He followed her to school one day", 8),   # "followed" is 2, not 3
        ("But still he lingered near", 6),
    ],
)
def test_syllable_counter(line, expected):
    assert script_qc._line_syllables(line) == expected


@pytest.mark.parametrize(
    "a, b, rhymes",
    [
        ("little star", "what you are", True),
        ("world so high", "in the sky", True),      # <gh> vs <y>
        ("down the stream", "but a dream", True),
        ("any wool", "three bags full", True),      # <oo>+l == <u>+ll
        ("as white as snow", "sure to go", True),   # silent <w>
        ("feel the breeze", "high up high", False),
        ("pour with fun", "water soon", False),
    ],
)
def test_rhyme_detection(a, b, rhymes):
    assert script_qc._verse_has_rhyme([a, b]) is rhymes


@pytest.mark.parametrize(
    "line, offender",
    [
        ("Blow bubbles high up high", "high"),
        ("Clap, clap, clap your hands", None),           # adjacent reduplication
        ("Merrily, merrily, merrily, merrily", None),
        ("Yes, sir, yes, sir, three bags full", None),   # repeated 2-word phrase
        ("Brother John, Brother John", None),
        ("Round and round and clap with me", None),
        ("Bake me a cake as fast as you can", None),     # function word
        ("Skip to my Lou, my darling", None),
    ],
)
def test_clumsy_phrasing_detection(line, offender):
    assert script_qc._clumsy_repetition(line) == offender


def test_chant_lines_are_exempt_from_meter_and_rhyme():
    """"Merrily, merrily, merrily, merrily" is a real nursery-rhyme device: it
    is deliberately off-meter and unrhymed and must not sink the verse."""
    song = {
        "title": "Row",
        "description": "d",
        "tags": ["t"],
        "characters": lyrics._fallback_characters(),
        "verses": [
            {
                "lines": ["Row, row, row your boat", "Gently down the stream",
                          "Merrily, merrily, merrily, merrily", "Life is but a dream"],
                "scene": "s",
            }
        ],
    }
    reasons = " | ".join(script_qc._check_script(song))
    assert "meter inconsistent" not in reasons
    assert "no rhyme" not in reasons


def test_hook_must_span_two_different_verses():
    """A verse bookended by its own opening line is not a hook."""
    verses = [
        {"lines": ["Clap your hands with me", "Jump up high", "Clap your hands with me"]},
        {"lines": ["Stomp your feet", "Wave hello"]},
    ]
    assert script_qc._hook_line(verses) is None
    verses[1]["lines"].append("Clap your hands with me")
    assert script_qc._hook_line(verses) == "clap your hands with me"


def test_existing_gates_are_not_weakened(cfg):
    """The pre-existing checks must still fire on the module's own broken song."""
    broken = {
        "title": "Broken Test Song",
        "description": "test",
        "tags": ["a"],
        "characters": "A happy toddler plays outside all afternoon long today.",
        "verses": [
            {"lines": ["Purple elephants juggle enormous mathematics textbooks beside sleepy lighthouses",
                       "Silver dolphins compose intricate symphonies inside forgotten crystal libraries"],
             "scene": "x"},
            {"lines": ["Golden giraffes analyze peculiar geometry puzzles atop wandering desert caravans",
                       "Emerald tigers assemble curious mechanical contraptions within abandoned laboratories"],
             "scene": "y"},
            {"lines": ["Crimson falcons decipher ancient philosophical manuscripts across frozen tundras",
                       "Turquoise octopi construct fantastical architectural monuments beneath oceanic trenches"],
             "scene": "z"},
        ],
    }
    reasons = _reasons(script_qc.review_script(broken, cfg))
    for expected in ("line too long", "no repetition", "not enough action verbs",
                     "characters sentence invalid"):
        assert expected in reasons


def test_brand_words_still_rejected(cfg):
    song = json.loads(json.dumps(lyrics.FALLBACK_SONG))
    song["title"] = "A Cocomelon Style Song"
    verdict = script_qc.review_script(song, cfg)
    assert verdict["accept"] is False
    assert "brand words present" in _reasons(verdict)


def test_topic_match_never_repeats_the_previous_episode_even_as_sole_match(cfg, monkeypatch):
    """Jobs 25+26 shipped baa_baa twice in a row: the farm topic's ONLY match
    was also the previous episode's song, and the old guard only skipped a
    repeat when a SECOND match existed. A sole-match repeat now falls through
    to rotation — an off-topic-but-fresh song programs the channel better than
    an on-topic rerun."""
    songs = lyrics.load_pd_library(cfg)
    baa = next(s for s in songs if "sheep" in s.get("id", "") or "baa" in s.get("id", ""))
    entry, why = lyrics.choose_pd_song(
        "big red tractor on the farm", cfg, songs=songs, recent_ids=[baa["id"]]
    )
    assert entry is not None
    assert entry["id"] != baa["id"], f"repeated {baa['id']} back-to-back: {why}"
