"""Tests for the scene-coherence fixes: one verse / one location / one activity,
no forbidden places, no martial or frightening staging.

Every fixture here is taken VERBATIM from a shot list or song that actually
shipped on 2026-07-20 and produced a visibly broken render, so a regression in
any of these rules re-breaks a known-bad episode:
  * output/20260720-170942-...-feel-the-breeze-song.json verse 0 gave three
    children three different activities.
  * ...-feel-the-breeze-shots.json planned verse 0 in a playroom, verse 1 in a
    backyard and verse 2 back in the playroom, and paired a "grassy field"
    action with a "playroom" setting (rendered as a room with a grass floor).
  * output/20260720-110129-...-watering-blooms-day-shots.json set a garden song
    inside "a bright bathroom with a bubbly tub" (rendered as a real bathtub).
"""
import pytest

from pipeline.kidsong import director
from pipeline.kidsong.script_qc import _check_shotlist, _check_verse_scenes

# The real multi-activity verse scene that started this.
_MIXED_SCENE = (
    "Zuri holds a bubble wand, Kofi stomps in a puddle, "
    "Nala dances with arms outstretched"
)
_TUB_SETTING = (
    "a bright bathroom with a bubbly tub behind them, a rubber duck on the "
    "tub edge, striped towels on a rail and a little step stool"
)
_PLAYROOM = "a bright cheerful playroom with colorful toys and a soft round rug"
_BACKYARD = (
    "a sunny backyard with green grass, a wooden fence, bright flowerbeds, a "
    "red tricycle and a leafy tree"
)


def _shot(sid, verse, **kw):
    shot = {
        "id": sid,
        "verse": verse,
        "start": 0.0,
        "end": 3.0,
        "shot_type": "medium",
        "characters": ["Zuri"],
        "action": "Zuri claps their hands to the beat",
        "camera": "static",
        "setting": _BACKYARD,
        "reuse_of": None,
    }
    shot.update(kw)
    return shot


def _categories(reasons):
    return {r.split(":", 1)[0].strip() for r in reasons}


# ------------------------------------------------- one verse, one activity ---
def test_multi_activity_scene_is_detected():
    assert director.verse_mixes_activities(_MIXED_SCENE)


def test_multi_activity_scene_reduces_to_one_shared_activity():
    reduced = director.reduce_scene_to_shared_activity(_MIXED_SCENE)
    # One activity, shared by the whole cast, built around the scene's own prop.
    assert "bubble" in reduced.lower()
    assert not director.verse_mixes_activities(reduced)
    # No child is singled out with their own separate activity any more.
    for name in ("Zuri", "Kofi", "Nala"):
        assert name not in reduced


def test_scene_with_a_single_shared_activity_is_left_alone():
    scene = "All three children dance and play together on a green grassy field"
    assert not director.verse_mixes_activities(scene)
    assert director.reduce_scene_to_shared_activity(scene) == scene


def test_one_verb_per_shot_one_location_sequence_is_a_narrative_not_a_mix():
    """Regression note / behaviour change: this exact fixture — three
    different children, one single-clause activity apiece, one shared
    location — used to be rejected by a flat "<=2 distinct verbs per verse"
    cap (this test used to be named `test_verse_mixing_activities_is_rejected`
    and asserted the opposite). That cap was an over-correction: a verse may
    legitimately progress through a setup -> event -> reaction arc, and
    `director.verse_activity_progression` now classifies "one shared
    location, at most one activity verb per shot, <=4 verbs total" as
    'narrative' rather than 'scattered'. The shot-list gate no longer rejects
    it. The SIMULTANEOUS-mix fault this fixture originally stood in for
    (different children handed different activities in the same beat) is
    caught earlier, on the LYRICS scene text, by `_check_verse_scenes` — see
    `test_mixed_scene_is_reported_by_the_script_gate_not_the_shotlist_gate`
    and `test_simultaneous_mix_scene_still_trips_the_scene_gate` below."""
    shots = [
        _shot("s00", 0, start=0.0, end=3.0, shot_type="wide",
              action="Zuri blows bubbles high in the air"),
        _shot("s01", 0, start=3.0, end=6.0, action="Kofi stomps in a puddle"),
        _shot("s02", 0, start=6.0, end=9.0, action="Nala dances with her arms outstretched"),
    ]
    assert "verse mixes activities" not in _categories(_check_shotlist(shots))


def test_narrative_arc_verse_is_not_a_mixed_verse():
    """A verse may legitimately progress through a temporal arc — setup,
    event, reaction — as long as its shots share one location and each shot
    carries at most one activity verb: the clock strikes, a mouse runs down
    it, a child notices and points."""
    clock_room = "a sunny playroom with a tall grandfather clock in the corner"
    shots = [
        _shot("s00", 0, start=0.0, end=3.0, shot_type="wide", characters=[],
              action="the clock strikes one", setting=clock_room),
        _shot("s01", 0, start=3.0, end=6.0, characters=[],
              action="a tiny round cartoon mouse scampers down the clock case",
              setting=clock_room),
        _shot("s02", 0, start=6.0, end=9.0, characters=["Zuri"],
              action="Zuri points after the mouse and giggles", setting=clock_room),
    ]
    assert "verse mixes activities" not in _categories(_check_shotlist(shots))


def test_hook_verse_of_all_clapping_shots_is_not_a_mixed_verse():
    shots = [
        _shot("s00", 0, start=0.0, end=3.0, shot_type="wide",
              action="Zuri claps their hands to the beat"),
        _shot("s01", 0, start=3.0, end=6.0, action="Kofi claps their hands to the beat"),
        _shot("s02", 0, start=6.0, end=9.0, action="Nala claps their hands to the beat"),
    ]
    assert "verse mixes activities" not in _categories(_check_shotlist(shots))


def test_genuinely_scattered_verse_is_still_rejected():
    """Differing settings AND enough distinct verbs that no shared through-line
    is plausible — no shared location, more than one verb per shot — is
    'scattered' and must still be rejected."""
    shots = [
        _shot("s00", 0, start=0.0, end=3.0, shot_type="wide",
              action="Zuri blows bubbles high in the air", setting=_BACKYARD),
        _shot("s01", 0, start=3.0, end=6.0,
              action="Kofi stomps in a puddle and splashes water", setting=_PLAYROOM),
        _shot("s02", 0, start=6.0, end=9.0,
              action="Nala paints a picture and builds a tower", setting=_BACKYARD),
    ]
    assert "verse mixes activities" in _categories(_check_shotlist(shots))


def test_same_activity_with_different_performers_is_not_a_mixed_verse():
    """The roll-call pattern is a repetition fault, never an activity fault —
    swapping the performer must not read as a new activity."""
    shots = [
        _shot("s00", 0, shot_type="wide", action="Zuri claps their hands to the beat"),
        _shot("s01", 0, action="Kofi claps their hands to the beat"),
        _shot("s02", 0, action="Nala claps their hands to the beat"),
    ]
    assert "verse mixes activities" not in _categories(_check_shotlist(shots))


def test_appended_performance_beats_do_not_read_as_new_activities():
    """`vary_repeated_actions` appends a beat to break up a roll-call; the gate
    must not then reject its own output."""
    shots = [
        _shot("s00", 0, shot_type="wide", action="Zuri sings along with a happy face"),
        _shot(
            "s01", 0,
            action="Kofi sings along with a happy face, bouncing gently on the spot",
        ),
        _shot(
            "s02", 0,
            action="Nala sings along with a happy face, swaying from side to side",
        ),
    ]
    assert "verse mixes activities" not in _categories(_check_shotlist(shots))


def test_vary_repeated_actions_breaks_up_a_roll_call():
    shots = [
        _shot("s00", 0, characters=["Zuri"], action="Zuri sings along with a happy face"),
        _shot("s01", 0, characters=["Kofi"], action="Kofi sings along with a happy face"),
        _shot("s02", 0, characters=["Nala"], action="Nala sings along with a happy face"),
    ]
    director.vary_repeated_actions(shots)
    actions = [s["action"] for s in shots]
    assert len(set(actions)) == 3, actions
    # ...but every shot still depicts the SAME activity.
    assert len({director.activity_signature(a) for a in actions}) == 1, actions


# ------------------------------------------------------ forbidden locations ---
@pytest.mark.parametrize(
    "setting",
    [
        _TUB_SETTING,
        "a warm bathroom with a big bathtub",
        "a sunny garden with a paddling pool",
        "a tiled room with a shower",
        "a bathroom with a potty and a step stool",
    ],
)
def test_forbidden_locations_are_rejected(setting):
    reasons = _check_shotlist([_shot("s00", 0, shot_type="wide", setting=setting)])
    assert "forbidden location" in _categories(reasons)


def test_bathroom_sink_is_still_allowed():
    """The channel bans tubs and pools, not bathrooms — tooth-brushing episodes
    legitimately happen at a bathroom sink."""
    setting = (
        "a bright cheerful bathroom with a white pedestal sink, a round mirror "
        "with a colorful frame and colorful toiletry bottles"
    )
    assert director.forbidden_location_terms(setting) == []


def test_water_songs_no_longer_map_to_a_tub():
    """The measured root cause of the watering-blooms bathtub: the word "water"
    in a GARDEN song matched a bathroom-with-tub location."""
    location = director._location_from(
        "Zuri pours water on the flowers while Kofi and Nala watch"
    )
    assert director.forbidden_location_terms(location) == []
    assert "backyard" in location


# ------------------------------------------------------- martial and scary ---
@pytest.mark.parametrize(
    "action",
    [
        "The kids march in a row across the grass",
        "The children line up in formation",
        "The kids move in unison, single file",
        "Three toddlers in uniforms salute the camera",
    ],
)
def test_martial_staging_is_rejected(action):
    reasons = _check_shotlist([_shot("s00", 0, shot_type="wide", action=action)])
    assert "martial staging" in _categories(reasons)


@pytest.mark.parametrize(
    "action",
    [
        "Zuri looks scared as a monster appears",
        "Kofi is crying in the darkness",
        "Nala runs away from a chasing shadow",
    ],
)
def test_frightening_content_is_rejected(action):
    reasons = _check_shotlist([_shot("s00", 0, shot_type="wide", action=action)])
    assert "frightening content" in _categories(reasons)


def test_ordinary_playful_actions_are_not_flagged():
    for action in (
        "The kids clap their hands to the beat",
        "Zuri blows bubbles and watches them float up",
        "Kofi stacks colorful blocks into a tower",
    ):
        assert director.martial_terms(action) == []
        assert director.scary_terms(action) == []


def test_sanitize_staging_replaces_a_marching_action():
    cleaned = director.sanitize_staging("The kids march in a straight line")
    assert director.martial_terms(cleaned) == []


# --------------------------------------------------------------- continuity ---
def test_location_change_within_a_verse_is_rejected():
    shots = [
        _shot("s00", 0, shot_type="wide", setting=_BACKYARD),
        _shot("s01", 0, setting=_PLAYROOM),
    ]
    assert "verse location discontinuity" in _categories(_check_shotlist(shots))


def test_returning_to_an_abandoned_location_is_rejected():
    """The real feel-the-breeze sequence: playroom -> backyard -> playroom."""
    shots = [
        _shot("s00", 0, shot_type="wide", setting=_PLAYROOM),
        _shot("s01", 1, setting=_BACKYARD),
        _shot("s02", 2, setting=_PLAYROOM),
    ]
    assert "location ping-pong" in _categories(_check_shotlist(shots))


def test_moving_forward_through_locations_is_allowed():
    kitchen = "a sunny cheerful kitchen with a fruit bowl and checkered curtains"
    shots = [
        _shot("s00", 0, shot_type="wide", setting=_PLAYROOM),
        _shot("s01", 1, setting=kitchen),
        _shot("s02", 2, setting=_BACKYARD),
    ]
    assert "location ping-pong" not in _categories(_check_shotlist(shots))


def test_indoor_action_in_an_outdoor_setting_is_rejected():
    """feel-the-breeze s00: a "grassy field" action in a "playroom" setting,
    which rendered as an indoor room with a grass floor."""
    shots = [
        _shot(
            "s00", 0, shot_type="wide", characters=["all"],
            action="The kids play in a bright green grassy field with flowers",
            setting=_PLAYROOM,
        ),
    ]
    assert "indoor/outdoor contradiction" in _categories(_check_shotlist(shots))


def test_agreeing_action_and_setting_pass():
    shots = [
        _shot(
            "s00", 0, shot_type="wide", characters=["all"],
            action="The kids play in the green grass",
            setting=_BACKYARD,
        ),
    ]
    assert "indoor/outdoor contradiction" not in _categories(_check_shotlist(shots))


# ------------------------------------------------------ verse location plan ---
def test_a_silent_verse_inherits_rather_than_teleporting_indoors():
    """The whole feel-the-breeze fault in one assertion: an outdoor song whose
    middle verse names no location must not snap back to a playroom."""
    song = {
        "title": "Feel the Breeze",
        "verses": [
            {"scene": _MIXED_SCENE, "lines": ["Let's go outside, feel the breeze"]},
            {
                "scene": "All three children dance on a green grassy field",
                "lines": ["Wave your arms, stomp your feet"],
            },
            {
                "scene": "Zuri claps, Kofi twirls, Nala laughs",
                "lines": ["Clap hands high up in the air"],
            },
        ],
    }
    locations = director.plan_verse_locations(song)
    assert len(set(locations)) == 1, locations
    assert "backyard" in locations[0]


def test_setting_is_a_place_accepts_the_default_playroom():
    """Regression: the substring rule matched " play" inside "playROOM", so the
    planner's own default location was reported as "not a place" on every shot
    of every episode that used it."""
    assert director.setting_is_a_place(_PLAYROOM)


def test_setting_is_a_place_still_rejects_an_action_sentence():
    assert not director.setting_is_a_place(
        "Amira, Kofi and Zuri are rinsing their mouths at the sink"
    )


# ------------------------------------------- paraphrase is not a new activity ---
def test_one_activity_phrased_several_ways_is_not_a_mixed_verse():
    """Measured false positive: the fallback song's verse 0 is ONE clapping
    activity, which the LLM phrased four different ways. Counting phrasings
    instead of activity verbs rejected a perfectly coherent verse."""
    shots = [
        _shot("s00", 0, shot_type="wide", characters=["all"],
              action="The kids stand in a sunny backyard with big smiles"),
        _shot("s01", 0, characters=["all"], action="The kids clasp hands together and clap"),
        _shot("s02", 0, action="Zuri claps her hands with a big smile"),
        _shot("s03", 0, characters=["all"], action="The kids continue clapping and smiling"),
    ]
    assert "verse mixes activities" not in _categories(_check_shotlist(shots))


def test_filler_beats_do_not_count_as_activities():
    for filler in ("Zuri sings along with a happy face", "Zuri smiles at the camera",
                   "Zuri stands still and watches"):
        assert director.activity_verbs(filler) == set(), filler
    assert "clap" in director.activity_verbs("Zuri claps her hands")


def test_three_genuinely_different_activities_in_one_shot_still_rejected():
    """The canonical bad verse — three unrelated activities forced into
    shared shots with no location cover, four distinct verbs — must still
    fail. (This used to reuse the "Zuri blows bubbles / Kofi stomps / Nala
    dances" fixture, one verb per shot; under
    `director.verse_activity_progression` that specific shape is now
    'narrative', not 'scattered' — see
    `test_one_verb_per_shot_one_location_sequence_is_a_narrative_not_a_mix`.
    This test keeps the "still rejected" guarantee alive with a fixture that
    is genuinely scattered under the new contract: no shared setting and more
    than one verb in a shot.)"""
    shots = [
        _shot("s00", 0, start=0.0, end=3.0, shot_type="wide",
              action="Zuri blows bubbles and chases them across the yard",
              setting=_BACKYARD),
        _shot("s01", 0, start=3.0, end=6.0,
              action="Kofi stomps in a puddle and splashes water everywhere",
              setting=_PLAYROOM),
        _shot("s02", 0, start=6.0, end=9.0,
              action="Nala dances and twirls with her arms outstretched",
              setting=_BACKYARD),
    ]
    assert "verse mixes activities" in _categories(_check_shotlist(shots))


# ------------------------------------------ public-domain nursery-rhyme corpus ---
def test_one_child_with_an_animal_is_not_a_mixed_verse():
    """"Zuri had a little lamb, he followed her to school" is ONE shared
    activity with a companion in frame — the check must require two distinct
    CAST children, not merely two verbs."""
    assert not director.verse_mixes_activities(
        "Zuri had a little lamb and he followed her to school one day"
    )


@pytest.mark.parametrize(
    "lyrics, expected",
    [
        ("Row, row, row your boat gently down the stream", "riverbank"),
        ("Baa baa black sheep have you any wool, skipping round the farm", "farmyard"),
        # A lone Twinkle line has no bedtime cue, so it resolves to the night-sky
        # stargazing location. (The full PD song, which does carry bedtime cues,
        # resolves to the cozy bedroom — both are coherent; what matters is that
        # neither is the generic playroom.)
        ("Twinkle twinkle little star, up above the world so high", "starry"),
    ],
)
def test_nursery_rhymes_get_a_location_matching_their_lyrics(lyrics, expected):
    """The PD corpus carries NO per-verse scene, so the song-level fallback is
    the only thing choosing the place. Landing a boat song in a playroom would
    guarantee an indoor/outdoor contradiction the replan loop cannot fix."""
    song = {"title": "t", "verses": [{"scene": None, "lines": [lyrics]}] * 3}
    locations = director.plan_verse_locations(song)
    assert len(set(locations)) == 1, locations
    assert expected in locations[0], locations[0]
    assert "playroom" not in locations[0], locations[0]


def test_canonical_setting_is_the_default_when_no_verse_names_another_place():
    """`song["canonical_setting"]` (stamped by lyrics.pd_song_to_dict onto
    every library entry, see the Episode Canon feature) replaces the
    keyword-scanned song-text default AND a verse matching ONLY the generic
    backyard catch-all no longer counts as "naming a location" -- closing the
    loophole a farm scene that merely mentions incidental outdoor words
    (grass, sun) used to fall through, stealing it away from the farm."""
    song = {
        "title": "t",
        "canonical_setting": "a small sunny farmyard with a red barn and hay bales",
        "verses": [
            {"scene": "A round woolly black sheep grazes quietly nearby.", "lines": ["baa baa"]},
            {"scene": "Three fat sacks of wool sit on the grass in the sun.", "lines": ["x"]},
            {"scene": "The toddlers clap their hands and count to three.", "lines": ["x"]},
        ],
    }
    locations = director.plan_verse_locations(song)
    assert len(set(locations)) == 1, locations
    assert "tricycle" not in locations[0], locations[0]


def test_baa_baa_black_sheep_never_drifts_into_the_generic_backyard():
    """Measured live (output/20260724-181038-kidsong-baa-baa-...): the episode
    correctly opened on the farm (scene_hints name a barn, hay bales, a stone
    wall) but every verse from s05 onward silently reclassified into the
    generic "sunny backyard ... red tricycle" catch-all, because that row used
    to sit BEFORE the farm-specific keyword row in `director._LOCATION_KEYWORDS`
    and a farm scene mentioning incidental outdoor words ("wool... on the
    grass") got stolen by it. Exercises the real shipped library entry
    end to end (both the reordered keyword table and the new
    `canonical_setting` plumbing)."""
    from pipeline.kidsong import lyrics

    songs = lyrics.load_pd_library()
    entry = next(s for s in songs if s.get("id") == "baa_baa_black_sheep")
    song = lyrics.pd_song_to_dict(entry)
    assert song["canonical_setting"], "baa_baa_black_sheep must carry a canonical_setting"

    locations = director.plan_verse_locations(song)
    assert len(locations) == len(song["verses"])
    for loc in locations:
        assert "tricycle" not in loc, loc
        assert "farm" in loc or "barn" in loc, loc
    # One stable setting for the whole episode -- no mid-song teleport.
    assert len(set(locations)) == 1, locations


def test_location_keywords_do_not_match_inside_other_words():
    """"row" must not fire on "brown"/"arrow"/"tomorrow", and "star" must not
    fire on "start" — bare substring matching used to allow exactly that."""
    for text in ("we start the day", "a brown teddy bear", "see you tomorrow"):
        location = director._location_from(text)
        assert "riverbank" not in location, (text, location)
        assert "starry" not in location, (text, location)


def test_pd_style_song_scene_of_none_does_not_crash_the_scene_gate():
    song = {"title": "t", "verses": [{"scene": None, "lines": ["Row your boat"]}]}
    assert _check_verse_scenes(song) == []


# -------------------------------------------- the scene gate is script-side ---
def test_mixed_scene_is_reported_by_the_script_gate_not_the_shotlist_gate():
    """The scene sentence is a LYRICS artifact and the director already repairs
    it while planning, so rejecting the SHOT LIST for it would be
    non-actionable — "replan" cannot rewrite the song."""
    song = {
        "title": "t",
        "verses": [{"scene": _MIXED_SCENE, "lines": ["Let's go outside"]}],
    }
    reasons = _check_verse_scenes(song)
    assert any(r.startswith("verse mixes activities") for r in reasons)
    # ...and the reason names the shared activity the lyrics should use instead.
    assert "bubble" in reasons[0].lower()


# ------------------------------------ a story subject is not "different children" ---
# `_check_verse_scenes` hunts for a scene that hands MULTIPLE CHILDREN multiple
# activities. A non-child story subject (a lamb, a mouse) plus a single child's
# reaction must never trip it — these two scene hints are verbatim from the
# library of story scenes the director draws on.
def test_story_subject_scene_with_one_child_reaction_does_not_trip_scene_gate():
    scene = (
        "The lamb peeks around the doorway of a bright little schoolroom while "
        "Kofi and Nala laugh and point."
    )
    song = {"title": "t", "verses": [{"scene": scene, "lines": ["A lamb followed her to school"]}]}
    reasons = _check_verse_scenes(song)
    assert not any(r.startswith("verse mixes activities") for r in reasons)


def test_story_subject_alone_at_its_scene_does_not_trip_scene_gate():
    scene = (
        "A tall friendly grandfather clock stands in a sunny playroom with a "
        "tiny round cartoon mouse at its base."
    )
    song = {"title": "t", "verses": [{"scene": scene, "lines": ["The clock struck one"]}]}
    reasons = _check_verse_scenes(song)
    assert not any(r.startswith("verse mixes activities") for r in reasons)


def test_simultaneous_mix_scene_still_trips_the_scene_gate():
    """Regression: the real simultaneous-mix scene that started this whole
    fix — three children, three different activities in the same beat — must
    still be rejected by the SCRIPT-side scene gate."""
    song = {
        "title": "t",
        "verses": [{"scene": _MIXED_SCENE, "lines": ["Let's go outside"]}],
    }
    assert any(
        r.startswith("verse mixes activities") for r in _check_verse_scenes(song)
    )


# ---------------------------------------- characters=[] + story_subject shots ---
def test_shot_with_no_characters_and_a_story_subject_passes_the_shotlist_gate():
    """A shot may legitimately have `characters == []` when only the story
    subject (never a child) is on screen — this must not be rejected as "a
    shot with no characters", and must not trip the head-count gate either."""
    clock_room = "a sunny playroom with a tall grandfather clock in the corner"
    shots = [
        _shot(
            "s00", 0, shot_type="wide", start=0.0, end=3.0, characters=[],
            action="a tiny round cartoon mouse scampers up the grandfather clock",
            setting=clock_room,
            story_subject="a tiny round cartoon mouse",
        ),
        _shot(
            "s01", 0, start=3.0, end=6.0, characters=["Zuri"],
            action="Zuri points after the mouse and giggles",
            setting=clock_room,
        ),
    ]
    reasons = _check_shotlist(shots)
    assert not any("no shots" in r for r in reasons)
    assert not any("head count contradiction" in r for r in reasons)
    assert not any("cluttered frame" in r for r in reasons)
