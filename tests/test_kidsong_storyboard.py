"""
kidsong.storyboard — the STORYBOARD BRAIN that turns a song into concrete
per-verse action beats with gaze targets, so shots can be tied to purposeful
actions instead of canned filler ("sings along with a happy face, moving to
the beat").

Locked down here:
  1. Schema coercion (`_normalize_storyboard`) never trusts an LLM's types,
     presence or bounds — malformed input degrades to an empty/safe shape.
  2. The deterministic gate (`_storyboard_verdict`) rejects every fault it
     claims to catch: verb-less beats, martial phrasing, camera overbudget,
     a missing "{name}" placeholder, a story_subject beat naming a child, a
     concrete cast name standing in for the placeholder, and a scattered
     per-verse activity progression — while accepting a well-formed board.
  3. `build_storyboard`'s orchestration: the accept path ships the LLM board
     verbatim-normalized; a rejected attempt's reasons are fed back into the
     next attempt's prompt and a later success ships; when every attempt
     fails (or the backend is unreachable), `fallback_storyboard` ships.
  4. `fallback_storyboard` itself always passes its own gate — for a song
     with a detectable story subject AND for one without — is deterministic,
     and never emits the filler phrases this module exists to replace.
"""
import copy
import json

import pytest

from pipeline.kidsong import director, storyboard


@pytest.fixture()
def cfg():
    from pipeline.config import load_config

    c = copy.deepcopy(load_config())
    c.setdefault("kidsong", {})["storyboard"] = {"attempts": 3}
    return c


def _reasons(verdict):
    return " | ".join(verdict["reasons"])


# ---------------------------------------------------------------- fixtures ---
SONG = {
    "title": "Ride My Balance Bike - A Happy Sing-Along for Toddlers",
    "description": "Zuri, Kofi and Nala roll along on their balance bikes!",
    "tags": ["kids song", "balance bike"],
    "characters": "Three adorable Black toddlers: Zuri, Kofi and Nala.",
    "verses": [
        {"lines": ["Roll along, my bike and me"], "scene": "A sunny park path with balance bikes."},
        {"lines": ["Wave hello as friends go by"], "scene": "The kids ride their balance bikes around the park."},
        {"lines": ["See the wheels go spinning fast"], "scene": "The kids roll their bikes down a gentle grassy slope."},
    ],
}

# A hand-authored, already-clean, gate-passing storyboard for SONG (3 verses).
# Deliberately includes ONE camera_address=True beat (proving "at most one" is
# a real budget, not "zero") and a mix of child/all performer beats.
GOOD_STORYBOARD = {
    "verses": [
        {
            "verse": 0,
            "beat_goal": "The kids notice the balance bikes for the first time.",
            "beats": [
                {"action": "{name} spots the shiny balance bike and points, eyes wide",
                 "gaze": "wide-eyed at the balance bike", "performer": "child", "camera_address": False},
                {"action": "{name} crouches down to touch the bike's handlebars",
                 "gaze": "down at the handlebars", "performer": "child", "camera_address": False},
                {"action": "The kids gather around the bike, chattering happily",
                 "gaze": "at the balance bike", "performer": "all", "camera_address": False},
                {"action": "{name} grins with a big happy face right at the viewer",
                 "gaze": "warmly at the camera", "performer": "child", "camera_address": True,
                 "prop": "a shiny red balance bike"},
            ],
        },
        {
            "verse": 1,
            "beat_goal": "The kids try riding the balance bike.",
            "beats": [
                {"action": "{name} climbs onto the balance bike seat",
                 "gaze": "down at the seat", "performer": "child", "camera_address": False,
                 "prop": "a shiny red balance bike"},
                {"action": "{name} pushes off with both feet on the path",
                 "gaze": "ahead on the path", "performer": "child", "camera_address": False},
                {"action": "The kids roll slowly along the sunny park path",
                 "gaze": "ahead on the path", "performer": "all", "camera_address": False},
                {"action": "{name} wobbles a little and giggles at the wobble",
                 "gaze": "down at the handlebars", "performer": "child", "camera_address": False},
                {"action": "The kids cheer for each other along the path",
                 "gaze": "at each other", "performer": "all", "camera_address": False},
            ],
        },
        {
            "verse": 2,
            "beat_goal": "The kids celebrate rolling their bikes together.",
            "beats": [
                {"action": "The kids roll their bikes down the grassy slope together",
                 "gaze": "ahead down the slope", "performer": "all", "camera_address": False},
                {"action": "{name} lifts both feet off the pedals, coasting freely",
                 "gaze": "down at the pedals", "performer": "child", "camera_address": False},
                {"action": "The kids ring their bike bells in triumph",
                 "gaze": "down at the bell", "performer": "all", "camera_address": False},
                {"action": "{name} hops off the bike, beaming with pride",
                 "gaze": "at the balance bike", "performer": "child", "camera_address": False},
            ],
        },
    ]
}


def _mutate_beat(board, verse_idx, beat_idx, **updates):
    """Deep-copy `board` and apply `updates` to one beat, for an isolated
    single-fault gate test."""
    out = copy.deepcopy(board)
    out["verses"][verse_idx]["beats"][beat_idx].update(updates)
    return out


# ------------------------------------------------------- schema coercion -----
def test_good_storyboard_fixture_actually_passes_the_gate():
    """Precondition for every gate-rejection test below: the fixture other
    tests mutate is genuinely gate-passing on its own."""
    verdict = storyboard._storyboard_verdict(GOOD_STORYBOARD, SONG)
    assert verdict["accept"] is True, _reasons(verdict)


def test_normalize_handles_non_dict_and_none_data():
    for garbage in (None, "a string", 42, ["a", "list"]):
        out = storyboard._normalize_storyboard(garbage, SONG)
        assert out == {"verses": [{"verse": i, "beat_goal": "", "beats": []} for i in range(3)]}


def test_normalize_without_song_uses_the_raw_data_length():
    raw = {"verses": [{"verse": 0, "beats": [{"action": "{name} claps"}]}]}
    out = storyboard._normalize_storyboard(raw, song=None)
    assert len(out["verses"]) == 1
    assert out["verses"][0]["beats"][0]["action"] == "{name} claps"


def test_normalize_coerces_wrong_types_defensively():
    song = {"verses": [{}, {}]}
    raw = {
        "verses": [
            {
                "verse": "not-an-int",
                "beat_goal": 12345,
                "beats": {"not": "a list at all"},
            },
            {
                "verse": 1,
                "beat_goal": "ok",
                "beats": [
                    {"action": "  {name}   claps  ", "gaze": None, "performer": 42,
                     "camera_address": "true", "prop": 7},
                    {"action": "", "gaze": "x", "performer": "child"},  # empty action -> dropped
                    "not a dict at all",                                # skipped entirely
                    {"action": "{name} waves", "performer": "ALL"},      # case-insensitive performer
                ],
            },
        ]
    }
    out = storyboard._normalize_storyboard(raw, song)
    assert len(out["verses"]) == 2

    v0 = out["verses"][0]
    assert v0["verse"] == 0            # invalid "verse" falls back to position
    assert v0["beat_goal"] == "12345"  # non-string coerced to string
    assert v0["beats"] == []           # a dict where a list was expected -> no beats survive

    v1 = out["verses"][1]
    assert v1["verse"] == 1
    assert len(v1["beats"]) == 2       # empty-action + non-dict entries dropped

    b0 = v1["beats"][0]
    assert b0["action"] == "{name} claps"   # whitespace collapsed and stripped
    assert b0["gaze"] == ""                 # None -> ""
    assert b0["performer"] == "child"       # invalid type -> default
    assert b0["camera_address"] is True     # "true" string -> real bool
    assert b0["prop"] == "7"                # numeric prop coerced to string

    b1 = v1["beats"][1]
    assert b1["performer"] == "all"         # "ALL" lowercased, still valid


def test_normalize_clamps_beats_to_six_per_verse():
    raw = {"verses": [{"verse": 0, "beats": [{"action": f"{{name}} claps beat {i}"} for i in range(9)]}]}
    out = storyboard._normalize_storyboard(raw, {"verses": [{}]})
    assert len(out["verses"][0]["beats"]) == 6


def test_normalize_is_idempotent_on_already_clean_input():
    out = storyboard._normalize_storyboard(copy.deepcopy(GOOD_STORYBOARD), SONG)
    assert out == GOOD_STORYBOARD


# --------------------------------------------------------------- the gate ----
def test_gate_rejects_a_verb_less_beat():
    bad = _mutate_beat(GOOD_STORYBOARD, 0, 0, action="{name} stay calm and quiet beside a red ball")
    verdict = storyboard._storyboard_verdict(bad, SONG)
    assert verdict["accept"] is False
    assert "not verb-ish" in _reasons(verdict)


def test_gate_rejects_martial_phrasing():
    bad = _mutate_beat(GOOD_STORYBOARD, 0, 0, action="{name} marches around the room in a straight line")
    verdict = storyboard._storyboard_verdict(bad, SONG)
    assert verdict["accept"] is False
    assert "martial staging" in _reasons(verdict)


def test_gate_rejects_frightening_content():
    bad = _mutate_beat(GOOD_STORYBOARD, 0, 0, action="{name} points at the scary monster and screams")
    verdict = storyboard._storyboard_verdict(bad, SONG)
    assert verdict["accept"] is False
    assert "frightening content" in _reasons(verdict)


def test_gate_rejects_camera_overbudget():
    # GOOD_STORYBOARD already has one camera_address=True beat; add a second.
    bad = _mutate_beat(GOOD_STORYBOARD, 1, 0, camera_address=True)
    verdict = storyboard._storyboard_verdict(bad, SONG)
    assert verdict["accept"] is False
    assert "camera_address overbudget" in _reasons(verdict)


def test_gate_accepts_exactly_one_camera_address_beat():
    """Positive control for the "at most one" rule: GOOD_STORYBOARD's single
    camera_address=True beat (with a gaze that addresses the camera) must NOT
    by itself be a rejection."""
    verdict = storyboard._storyboard_verdict(GOOD_STORYBOARD, SONG)
    assert verdict["accept"] is True, _reasons(verdict)


def test_gate_rejects_gaze_at_camera_without_camera_address_true():
    bad = _mutate_beat(GOOD_STORYBOARD, 1, 0, gaze="looking at the camera")
    verdict = storyboard._storyboard_verdict(bad, SONG)
    assert verdict["accept"] is False
    assert "gaze at camera" in _reasons(verdict)


def test_gate_rejects_missing_name_placeholder_on_a_child_beat():
    bad = _mutate_beat(GOOD_STORYBOARD, 0, 0, action="Kofi spots the shiny balance bike and points")
    verdict = storyboard._storyboard_verdict(bad, SONG)
    assert verdict["accept"] is False
    reasons = _reasons(verdict)
    assert "missing name placeholder" in reasons
    assert "concrete cast name" in reasons  # also caught by the blanket cast-name rule


def test_gate_rejects_story_subject_beat_using_the_name_placeholder():
    bad = _mutate_beat(
        GOOD_STORYBOARD, 0, 1,
        action="{name} watches the balance bike gleam in the sun",
        performer="story_subject",
    )
    verdict = storyboard._storyboard_verdict(bad, SONG)
    assert verdict["accept"] is False
    assert "story_subject uses placeholder" in _reasons(verdict)


def test_gate_rejects_story_subject_beat_naming_a_cast_child():
    bad = _mutate_beat(
        GOOD_STORYBOARD, 0, 1,
        action="the bike gleams while Nala watches nearby",
        performer="story_subject",
    )
    verdict = storyboard._storyboard_verdict(bad, SONG)
    assert verdict["accept"] is False
    assert "concrete cast name" in _reasons(verdict)


def test_gate_rejects_beat_count_out_of_range():
    bad = copy.deepcopy(GOOD_STORYBOARD)
    bad["verses"][0]["beats"] = bad["verses"][0]["beats"][:2]  # only 2 beats
    verdict = storyboard._storyboard_verdict(bad, SONG)
    assert verdict["accept"] is False
    assert "beat count" in _reasons(verdict)


def test_gate_rejects_missing_verse_coverage():
    bad = copy.deepcopy(GOOD_STORYBOARD)
    bad["verses"] = bad["verses"][:2]  # SONG has 3 verses
    verdict = storyboard._storyboard_verdict(bad, SONG)
    assert verdict["accept"] is False
    assert "verse coverage" in _reasons(verdict)


def test_gate_rejects_missing_gaze():
    bad = _mutate_beat(GOOD_STORYBOARD, 0, 0, gaze="")
    verdict = storyboard._storyboard_verdict(bad, SONG)
    assert verdict["accept"] is False
    assert "gaze missing" in _reasons(verdict)


def test_gate_rejects_scattered_activity_progression():
    """Five beats in one verse, each a DIFFERENT recognized activity verb (no
    repeats) — over the <=4-distinct-verbs budget `verse_activity_progression`
    allows for a legitimate narrative arc, so it reads as scattered rather
    than a single continuous moment."""
    scattered = copy.deepcopy(GOOD_STORYBOARD)
    scattered["verses"][1]["beats"] = [
        {"action": "{name} claps happily", "gaze": "at the bike", "performer": "child", "camera_address": False},
        {"action": "{name} stomps happily", "gaze": "at the bike", "performer": "child", "camera_address": False},
        {"action": "{name} waves happily", "gaze": "at the bike", "performer": "child", "camera_address": False},
        {"action": "{name} jumps happily", "gaze": "at the bike", "performer": "child", "camera_address": False},
        {"action": "{name} washes happily", "gaze": "at the bike", "performer": "child", "camera_address": False},
    ]
    # Precondition: this fixture isn't accidentally failing some OTHER rule
    # (each beat individually is verb-ish, safe, has {name}+gaze, in range).
    verdict = storyboard._storyboard_verdict(scattered, SONG)
    assert verdict["accept"] is False
    assert "scattered progression" in _reasons(verdict)


# --------------------------------------------------------- LLM stub helpers --
def _stub(*replies):
    """A `_generate_llm_storyboard`-shaped stub that returns each of `replies`
    in turn (repeating the last one if called more often), recording the
    `feedback` it was called with each time."""
    calls = []

    def fake(song, cfg, feedback=None):
        calls.append(feedback)
        idx = min(len(calls) - 1, len(replies) - 1)
        return copy.deepcopy(replies[idx])

    return fake, calls


# ------------------------------------------------------ build_storyboard -----
def test_build_storyboard_accept_path_returns_the_board_verbatim_normalized(cfg, monkeypatch):
    fake, calls = _stub(GOOD_STORYBOARD)
    monkeypatch.setattr(storyboard, "_generate_llm_storyboard", fake)
    out = storyboard.build_storyboard(SONG, cfg)
    assert out == GOOD_STORYBOARD
    assert len(calls) == 1
    assert calls[0] is None  # first attempt carries no feedback


def test_build_storyboard_retries_with_feedback_and_third_attempt_wins(cfg, monkeypatch):
    bad_missing_name = _mutate_beat(GOOD_STORYBOARD, 0, 0, action="Kofi spots the bike and points")
    bad_camera_budget = _mutate_beat(GOOD_STORYBOARD, 1, 0, camera_address=True)
    fake, calls = _stub(bad_missing_name, bad_camera_budget, GOOD_STORYBOARD)
    monkeypatch.setattr(storyboard, "_generate_llm_storyboard", fake)

    out = storyboard.build_storyboard(SONG, cfg)

    assert out == GOOD_STORYBOARD
    assert len(calls) == 3
    assert calls[0] is None
    assert calls[1] and any("missing name placeholder" in r or "concrete cast name" in r for r in calls[1])
    assert calls[2] and any("camera_address overbudget" in r for r in calls[2])


def test_build_storyboard_falls_back_when_every_attempt_fails(cfg, monkeypatch):
    cfg["kidsong"]["storyboard"]["attempts"] = 2
    always_bad = _mutate_beat(GOOD_STORYBOARD, 0, 0, action="Kofi spots the bike and points")
    fake, calls = _stub(always_bad)
    monkeypatch.setattr(storyboard, "_generate_llm_storyboard", fake)

    out = storyboard.build_storyboard(SONG, cfg)

    assert len(calls) == 2  # it really did retry before giving up
    assert out == storyboard.fallback_storyboard(SONG)


def test_build_storyboard_stops_immediately_when_the_backend_is_dead(cfg, monkeypatch):
    """`_backend_dead` short-circuits the retry loop instead of burning every
    remaining attempt on a backend that is definitely not coming back."""
    calls = []

    def dead(song, c, feedback=None):
        calls.append(feedback)
        board = storyboard._normalize_storyboard(None, song)
        board["_backend_dead"] = True
        return board

    cfg["kidsong"]["storyboard"]["attempts"] = 5
    monkeypatch.setattr(storyboard, "_generate_llm_storyboard", dead)
    out = storyboard.build_storyboard(SONG, cfg)
    assert len(calls) == 1
    assert out == storyboard.fallback_storyboard(SONG)


def test_build_storyboard_never_raises_when_the_stub_itself_raises(cfg, monkeypatch):
    def boom(song, c, feedback=None):
        raise RuntimeError("network exploded")

    monkeypatch.setattr(storyboard, "_generate_llm_storyboard", boom)
    out = storyboard.build_storyboard(SONG, cfg)
    assert out == storyboard.fallback_storyboard(SONG)


def test_build_storyboard_never_raises_on_garbage_song_or_cfg():
    assert storyboard.build_storyboard(None, {}) is not None
    assert storyboard.build_storyboard({}, None) is not None
    assert storyboard.build_storyboard("not a song", {"kidsong": {}}) is not None


def test_build_storyboard_respects_the_attempts_config(cfg, monkeypatch):
    cfg["kidsong"]["storyboard"]["attempts"] = 1
    always_bad = _mutate_beat(GOOD_STORYBOARD, 0, 0, action="Kofi spots the bike and points")
    fake, calls = _stub(always_bad, GOOD_STORYBOARD)  # 2nd reply would pass, but only 1 attempt allowed
    monkeypatch.setattr(storyboard, "_generate_llm_storyboard", fake)
    out = storyboard.build_storyboard(SONG, cfg)
    assert len(calls) == 1
    assert out == storyboard.fallback_storyboard(SONG)


def test_module_default_attempts_constant():
    assert storyboard.DEFAULT_STORYBOARD_ATTEMPTS == 3


def test_generate_llm_storyboard_flags_backend_dead_when_every_backend_fails(cfg, monkeypatch):
    """Exercises the REAL backend cascade (and, transitively, that
    prompts/kidsong_storyboard.txt actually loads) with every backend forced
    to fail, proving `_backend_dead` gets set without a live LLM."""
    cfg["llm"]["backend"] = "not-a-real-backend"  # step 1 raises immediately
    monkeypatch.setattr(director, "_post_ollama_chat", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no local ollama")))
    monkeypatch.setattr(director, "_post_openai_compatible_chat", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no local server")))

    out = storyboard._generate_llm_storyboard(SONG, cfg)
    assert out.get("_backend_dead") is True
    assert out["verses"] == [{"verse": i, "beat_goal": "", "beats": []} for i in range(3)]


# ------------------------------------------------------- fallback_storyboard -
def test_fallback_storyboard_passes_the_gate_for_a_story_subject_song(monkeypatch):
    """A song director.song_story_subject resolves to a real, non-child
    subject (a sunflower) — the has_subject=True branch of the arc."""
    monkeypatch.setattr(director, "song_story_subject", lambda song: "a bright yellow sunflower")
    song = {
        "title": "Zuri's Sunflower Garden Day",
        "description": "test",
        "tags": ["kids song"],
        "characters": "Three adorable Black toddlers: Zuri, Kofi and Nala.",
        "verses": [
            {"lines": ["A little seed goes in the ground"], "scene": "A sunny garden with a seedling."},
            {"lines": ["Water it each sunny day"], "scene": "The kids water the seedling."},
            {"lines": ["Up it grows so tall and bright"], "scene": "The sunflower has grown taller."},
            {"lines": ["Sing and dance around the sunflower"], "scene": "The kids dance around the bloom."},
        ],
    }
    board = storyboard.fallback_storyboard(song)
    assert len(board["verses"]) == 4
    verdict = storyboard._storyboard_verdict(board, song)
    assert verdict["accept"] is True, _reasons(verdict)
    # at least one beat is genuinely about the subject, not a child
    performers = {b["performer"] for v in board["verses"] for b in v["beats"]}
    assert "story_subject" in performers


def test_fallback_storyboard_passes_the_gate_for_a_subject_less_song(monkeypatch):
    """No non-child story subject at all — the generic shared-play arc built
    around the topic noun pulled from the title."""
    monkeypatch.setattr(director, "song_story_subject", lambda song: None)
    song = {
        "title": "Ride My Balance Bike - A Happy Sing-Along for Toddlers",
        "description": "test",
        "tags": ["kids song"],
        "characters": "Three adorable Black toddlers: Zuri, Kofi and Nala.",
        "verses": [
            {"lines": ["Roll along, my bike and me"], "scene": "s0"},
            {"lines": ["Wave hello as friends go by"], "scene": "s1"},
            {"lines": ["See the wheels go spinning fast"], "scene": "s2"},
        ],
    }
    board = storyboard.fallback_storyboard(song)
    assert len(board["verses"]) == 3
    verdict = storyboard._storyboard_verdict(board, song)
    assert verdict["accept"] is True, _reasons(verdict)
    performers = {b["performer"] for v in board["verses"] for b in v["beats"]}
    assert "story_subject" not in performers  # nothing to fabricate a subject beat from


@pytest.mark.parametrize("n_verses", [1, 2, 3, 4, 5])
def test_fallback_storyboard_passes_the_gate_across_verse_counts(monkeypatch, n_verses):
    monkeypatch.setattr(director, "song_story_subject", lambda song: None)
    song = {
        "title": "A Song About Something Wonderful",
        "verses": [{"lines": [f"line {i}"], "scene": f"scene {i}"} for i in range(n_verses)],
    }
    board = storyboard.fallback_storyboard(song)
    assert len(board["verses"]) == n_verses
    verdict = storyboard._storyboard_verdict(board, song)
    assert verdict["accept"] is True, _reasons(verdict)


def test_fallback_storyboard_is_deterministic():
    b1 = storyboard.fallback_storyboard(SONG)
    b2 = storyboard.fallback_storyboard(SONG)
    assert b1 == b2


def test_fallback_storyboard_contains_no_filler_or_camera_phrases():
    board = storyboard.fallback_storyboard(SONG)
    blob = json.dumps(board).lower()
    for banned in ("sings along", "moves to the beat", "smiles at the camera",
                   "at the camera", "into the lens"):
        assert banned not in blob, f"found banned phrase {banned!r} in the fallback board"


def test_fallback_storyboard_beats_always_use_the_name_placeholder_not_cast_names():
    board = storyboard.fallback_storyboard(SONG)
    blob = json.dumps(board)
    for cast_name in ("Zuri", "Kofi", "Nala"):
        assert cast_name not in blob
    assert "{name}" in blob


def test_gate_rejects_filler_actions():
    """Measured live (plan-A/B, sunflower song): an LLM board shipped
    "{name} sings along with a happy face, moving to the beat" straight through
    the gate — verb-ish, non-martial, gaze present — into the plan. The gate
    must hard-reject generic filler; prompt-level bans alone don't hold an 8B."""
    board = storyboard.fallback_storyboard(SONG)
    # corrupt one beat with the live-measured filler
    board["verses"][0]["beats"][0]["action"] = (
        "{name} sings along with a happy face, moving to the beat"
    )
    verdict = storyboard._storyboard_verdict(board, SONG)
    assert verdict["accept"] is False
    assert any("filler action" in r for r in verdict["reasons"])


def test_fallback_prop_is_topic_matched():
    """The TRY stage's held prop follows the topic via the curated table —
    a sunflower song explores a watering can, not "a favorite toy" (measured:
    the generic prop made the fallback arc read as random rather than topical).
    Unmatched topics keep the universally-coherent generic toy."""
    assert storyboard._fallback_prop("sunflower family tree") == "a small green watering can"
    assert storyboard._fallback_prop("rain rain go away") == "a bright yellow umbrella"
    assert storyboard._fallback_prop("bubble winding fun") == "a bubble wand"
    assert storyboard._fallback_prop("colorful breakfast table") == "a little mixing spoon"
    assert storyboard._fallback_prop("zorbulous quibblefratz") == storyboard._GENERIC_FALLBACK_PROP
    assert storyboard._fallback_prop("") == storyboard._GENERIC_FALLBACK_PROP
    assert storyboard._fallback_prop(None) == storyboard._GENERIC_FALLBACK_PROP


def test_fallback_storyboard_try_verse_uses_the_topical_prop():
    """End-to-end: a sunflower song's fallback board embeds the watering can in
    its try-stage beats (and still passes its own gate — asserted by the
    existing gate tests above)."""
    board = storyboard.fallback_storyboard({
        "title": "Sunflower Family Tree",
        "verses": [
            {"lines": ["a", "b"], "scene": "the kids find a sunflower"},
            {"lines": ["c", "d"], "scene": "the kids water the sunflower"},
            {"lines": ["e", "f"], "scene": "the kids cheer"},
        ],
    })
    blob = json.dumps(board).lower()
    assert "watering can" in blob
    assert storyboard._GENERIC_FALLBACK_PROP not in blob


def test_fallback_storyboard_follows_the_discover_try_grow_celebrate_arc():
    """For a 4-verse song this is literally discover -> try -> grow -> celebrate."""
    four_verse_song = {
        "title": "Test", "verses": [{"lines": ["x"], "scene": "x"} for _ in range(4)],
    }
    board = storyboard.fallback_storyboard(four_verse_song)
    goals = [v["beat_goal"] for v in board["verses"]]
    assert "notice" in goals[0]
    assert "try" in goals[1]
    assert "change" in goals[2] or "progress" in goals[2]
    assert "celebrate" in goals[3]


def test_fallback_storyboard_never_raises_on_garbage_song():
    for garbage in (None, {}, {"verses": []}, {"verses": "not a list"}, "not even a dict"):
        board = storyboard.fallback_storyboard(garbage)
        assert isinstance(board, dict) and "verses" in board


# ===================================================== story-driven casting ===
_STORY_CFG = {"kidsong": {"staging": "story"}}


def _story_song(n=5):
    return {
        "title": "A Song About Something Wonderful",
        "characters": "Three adorable Black toddlers: Zuri, Kofi and Nala.",
        "verses": [{"lines": [f"line {i}"], "scene": f"scene {i}"} for i in range(n)],
    }


def test_story_mode_middle_verses_have_no_group_beats_except_share(monkeypatch):
    monkeypatch.setattr(director, "song_story_subject", lambda song: None)
    song = _story_song(6)  # n=6 covers try/grow/explore/share as middles
    board = storyboard.fallback_storyboard(song, _STORY_CFG)
    n = len(board["verses"])
    for v in board["verses"]:
        stage = storyboard._stage_for_verse(v["verse"], n)
        all_beats = [b for b in v["beats"] if b.get("performer") == "all"]
        if stage in ("try", "grow", "explore"):
            assert not all_beats, f"story-mode {stage} verse has group beats: {all_beats}"
        if stage == "share":
            assert len(all_beats) == 1, "share keeps its single shared moment"


def test_story_mode_try_verse_has_exactly_one_partner_beat_with_both_placeholders(monkeypatch):
    monkeypatch.setattr(director, "song_story_subject", lambda song: None)
    board = storyboard.fallback_storyboard(_story_song(5), _STORY_CFG)
    partners = [b for v in board["verses"] for b in v["beats"] if b.get("role") == "partner"]
    assert len(partners) == 1
    beat = partners[0]
    assert "{name}" in beat["action"] and "{partner}" in beat["action"]
    assert beat.get("prop"), "the partner moment keeps the prop"
    assert beat["performer"] == "child"


@pytest.mark.parametrize("n_verses", [1, 2, 3, 4, 5, 6, 7])
@pytest.mark.parametrize("subject", ["a bright yellow sunflower", None])
def test_story_mode_board_passes_the_gate_across_verse_counts(monkeypatch, n_verses, subject):
    monkeypatch.setattr(director, "song_story_subject", lambda song: subject)
    song = _story_song(n_verses)
    board = storyboard.fallback_storyboard(song, _STORY_CFG)
    verdict = storyboard._storyboard_verdict(board, song)
    assert verdict["accept"] is True, _reasons(verdict)


def test_ensemble_board_is_byte_identical_with_and_without_cfg(monkeypatch):
    """Load-bearing default: cfg=None, staging absent, and staging=ensemble all
    produce the exact same board — and never a single role key."""
    monkeypatch.setattr(director, "song_story_subject", lambda song: None)
    song = _story_song(5)
    b_none = storyboard.fallback_storyboard(song)
    b_absent = storyboard.fallback_storyboard(song, {"kidsong": {}})
    b_ens = storyboard.fallback_storyboard(song, {"kidsong": {"staging": "ensemble"}})
    assert b_none == b_absent == b_ens
    assert not any("role" in b for v in b_none["verses"] for b in v["beats"])


def test_gate_rejects_partner_beat_missing_the_partner_placeholder(monkeypatch):
    monkeypatch.setattr(director, "song_story_subject", lambda song: None)
    song = _story_song(5)
    board = storyboard.fallback_storyboard(song, _STORY_CFG)
    for v in board["verses"]:
        for b in v["beats"]:
            if b.get("role") == "partner":
                b["action"] = "{name} passes a favorite toy gently to a castmate"
    verdict = storyboard._storyboard_verdict(board, song)
    assert verdict["accept"] is False
    assert "partner beat placeholders" in _reasons(verdict)


def test_gate_rejects_role_performer_mismatch(monkeypatch):
    monkeypatch.setattr(director, "song_story_subject", lambda song: None)
    song = _story_song(5)
    board = storyboard.fallback_storyboard(song, _STORY_CFG)
    board["verses"][0]["beats"][0]["role"] = "all"  # a solo beat claiming "all"
    verdict = storyboard._storyboard_verdict(board, song)
    assert verdict["accept"] is False
    assert "role/performer mismatch" in _reasons(verdict)


def test_gate_ignores_roles_when_absent():
    """A role-less board (ensemble, legacy LLM) sees ZERO new requirements —
    the original fixture still passes untouched."""
    verdict = storyboard._storyboard_verdict(GOOD_STORYBOARD, SONG)
    assert verdict["accept"] is True, _reasons(verdict)


def test_coerce_beat_passes_valid_roles_and_drops_garbage():
    good = storyboard._coerce_beat({"action": "{name} claps", "role": "Partner"})
    assert good["role"] == "partner"
    bad = storyboard._coerce_beat({"action": "{name} claps", "role": "director"})
    assert "role" not in bad


# ------------------------------------------------------- hostile input fuzz --
_HOSTILE = [
    None, {}, [], "", 0, False, "a string",
    {"verses": None}, {"verses": "not a list"}, {"verses": [None, 1, "x"]},
    {"verses": [{"lines": None}]}, {"verses": [{"lines": [None, 1, {}]}]},
    {"verses": [{"beats": [None, "x", {"action": None}]}]},
    {"verses": [{"verse": "NaN", "beats": [{"action": "clap"}]}]},
    {"title": {"nested": "dict"}, "verses": [{"lines": ["x"]}]},
]


@pytest.mark.parametrize("bad", _HOSTILE)
def test_normalize_survives_any_shape_of_llm_output(bad):
    """"Never trust an LLM's types, presence, or bounds" is this module's own
    stated rule. One line broke it: `song` was assumed to be a dict, so a
    non-dict raised AttributeError while every other input was coerced."""
    out = storyboard._normalize_storyboard(bad)
    assert isinstance(out, dict) and isinstance(out["verses"], list)


@pytest.mark.parametrize("bad", _HOSTILE)
def test_normalize_survives_a_hostile_song_argument(bad):
    out = storyboard._normalize_storyboard({"verses": [{"beats": [{"action": "clap"}]}]}, bad)
    assert isinstance(out, dict) and isinstance(out["verses"], list)


@pytest.mark.parametrize("bad", _HOSTILE)
def test_the_verdict_gate_survives_any_shape(bad):
    """The gate decides whether an attempt is retried; it must never be the
    thing that raises."""
    verdict = storyboard._storyboard_verdict(
        storyboard._normalize_storyboard(bad), bad if isinstance(bad, dict) else {}
    )
    assert isinstance(verdict, dict) and "accept" in verdict
