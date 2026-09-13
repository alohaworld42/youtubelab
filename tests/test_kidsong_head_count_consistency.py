"""Tests for the head-count CONTRADICTION repair — the defect two GPU A/B rounds
(output/_ab_cast_prompts/, output/_ab_cast_prompts_v2/) measured as the dominant
remaining one after the cast bible landed.

Those rounds found wardrobe fidelity improving 5/5 while head-count control
improved 0/5, with the failure mode merely changing shape: instead of inventing a
different extra child, the renderer began CLONING the correct child 2-3x (shot s10
rendered three identical Zuris in a circle). The cause was not the negatives —
adding "duplicate characters" to them did not stop it — but that a shot's own
action/setting prose contradicts the head-count sentence built from the SAME shot
a few words earlier in the SAME prompt. Verbatim from the shipped code at the time:

    "A closeup shot of exactly one child: Zuri, ... .
     The kids smile brightly at each other, in
     Amira, Kofi, and Zuri are rinsing their mouths ..."

One child and three children asserted in one breath. These tests pin the repair at
all three levels it was made at:
  - director._normalize_shots  — the source fix, so new shot lists are born
    consistent (settings reduced to places, single-character actions restated).
  - generate._shot_prompt      — the defensive fix, so the ~every-single-character
    shot in the shot lists ALREADY on disk still renders a consistent prompt.
  - script_qc                  — a structural gate, so a contradictory shot list is
    rejected rather than silently rendered.

CPU-only: no GPU, no network, no LLM, no ffmpeg.
"""
from pipeline.kidsong import director
from pipeline.kidsong.generate import _shot_prompt
from pipeline.kidsong.script_qc import _check_shotlist


# ------------------------------------------------------------------ fixtures ---
def _cfg():
    return {"kidsong": {"seed": 20260717, "shot": {"style_trigger": "P1x4r"}}}


def _song(characters="Zuri, Kofi and Nala are best friends"):
    return {
        "characters": characters,
        "verses": [{"scene": "a bathroom", "lines": ["brush your teeth", "b", "c"]}],
    }


def _shot(**kw):
    base = {
        "id": "s10", "verse": 0, "lyric_span": [0, 0], "start": 0.0, "end": 3.0,
        "shot_type": "closeup", "characters": ["Zuri"],
        "action": "Zuri smiles at the camera", "camera": "static",
        "setting": "a bright cheerful bathroom with a white pedestal sink",
        "reuse_of": None, "seed": 20260717, "status": "planned",
    }
    base.update(kw)
    return base


# ------------------------------------------------------- the primitives ---
def test_group_language_implies_two_children_even_when_it_names_nobody():
    # "The kids smile brightly at each other" names no cast member at all, which
    # is why the pre-existing closeup scrub (which only fired on a NAMED other
    # child) let it through onto a one-character shot.
    assert director.implied_child_count("The kids smile brightly at each other") >= 2
    assert director.implied_child_count("The three friends brush together") >= 2
    assert director.implied_child_count("Zuri smiles at the camera") == 1


def test_implied_count_counts_distinct_cast_names():
    assert director.implied_child_count("Zuri, Kofi, and Nala are rinsing") == 3
    assert director.implied_child_count("Zuri and Zuri") == 1


def test_implied_count_of_neutral_text_does_not_constrain_the_head_count():
    # 0 means "this text says nothing about how many children" — never "zero
    # children on screen".
    assert director.implied_child_count("a sunny backyard with a wooden fence") == 0


def test_setting_is_a_place_rejects_an_action_sentence_and_accepts_a_location():
    assert not director.setting_is_a_place(
        "Amira, Kofi, and Zuri are rinsing their mouths with water"
    )
    assert not director.setting_is_a_place(
        "The three friends are brushing their teeth together in the mirror"
    )
    assert director.setting_is_a_place(
        "a bright cheerful bathroom with a white pedestal sink and a round mirror"
    )
    assert director.setting_is_a_place("")


def test_sanitize_setting_keeps_the_room_while_removing_the_people():
    out = director.sanitize_setting(
        "The three friends are brushing their teeth together, smiling at each "
        "other in the mirror."
    )
    assert director.setting_is_a_place(out)
    # The shot must stay in the bathroom the director chose, not teleport.
    assert "bathroom" in out.lower()
    assert "three friends" not in out.lower()


def test_sanitize_setting_leaves_a_real_location_untouched():
    place = "a sunny backyard with green grass, a wooden fence and a leafy tree"
    assert director.sanitize_setting(place) == place


def test_every_canonical_location_survives_sanitising_unchanged():
    """`sanitize_setting` rewrites via `_location_from`, so if a canonical
    location did NOT pass `setting_is_a_place` the sanitizer would rewrite its
    own output and could silently teleport a shot to another room."""
    for _keywords, location in director._LOCATION_KEYWORDS:
        assert director.setting_is_a_place(location), location
        assert director.sanitize_setting(location) == location


def test_sanitize_setting_is_idempotent_on_a_people_setting():
    once = director.sanitize_setting("The three friends are brushing their teeth")
    assert director.sanitize_setting(once) == once


def test_sanitize_action_restates_group_prose_in_the_singular():
    out = director.sanitize_action(
        "The kids smile brightly at each other", 1, subject="Zuri"
    )
    assert director.implied_child_count(out) <= 1
    assert out.startswith("Zuri")


def test_sanitize_action_is_a_no_op_when_the_count_is_unconstrained_or_plural():
    action = "The kids brush their teeth with big smiles"
    assert director.sanitize_action(action, None, subject="Zuri") == action
    assert director.sanitize_action(action, 3, subject="Zuri") == action


def test_sanitize_action_leaves_an_already_singular_action_alone():
    action = "Zuri holds up the toothbrush and shows it to the camera"
    assert director.sanitize_action(action, 1, subject="Zuri") == action


# -------------------------------------------- the fix at the prompt level ---
def test_shot_prompt_no_longer_contradicts_itself_on_a_single_child_closeup():
    """The exact bennys s14 shot that rendered the correct child twice."""
    shot = _shot(
        id="s14", shot_type="closeup", characters=["Zuri"],
        action="The kids smile brightly at each other",
        setting="Amira, Kofi, and Zuri are rinsing their mouths with water.",
    )
    prompt = _shot_prompt(shot, _song(), _cfg())

    assert "exactly one child" in prompt
    # The contradicting prose is gone from the prompt entirely.
    assert "The kids" not in prompt
    assert "each other" not in prompt
    assert "Amira" not in prompt
    assert "Kofi" not in prompt


def test_shot_prompt_strips_group_language_from_a_single_child_wide_shot():
    """bennys s08: an unresolvable name narrows a wide shot to one child, while
    the prose still puts three in the room."""
    shot = _shot(
        id="s08", shot_type="wide", characters=["Amira"],
        action="The kids brush their teeth with big smiles",
        setting="The three friends are brushing their teeth together in the mirror.",
    )
    prompt = _shot_prompt(shot, _song(), _cfg())

    assert "Exactly one child is on screen" in prompt
    assert "three friends" not in prompt
    assert "The kids" not in prompt


def test_shot_prompt_rewrites_the_action_to_name_the_child_actually_described():
    """An unresolvable "Amira" renders as Nala, so a rewritten action must name
    Nala — naming the dropped child would reintroduce an off-cast name."""
    shot = _shot(
        id="s08", shot_type="wide", characters=["Amira"],
        action="The kids sing along with happy faces",
        setting="a bright cheerful bathroom with a white pedestal sink",
    )
    prompt = _shot_prompt(shot, _song(), _cfg())

    assert "Amira" not in prompt
    # Whatever the bible substituted is the one child both sentences talk about.
    subject = prompt.split("on screen: ", 1)[1].split(",", 1)[0]
    assert subject in ("Zuri", "Kofi", "Nala")
    assert prompt.count("A wide shot: %s " % subject) == 1


def test_shot_prompt_leaves_an_already_consistent_shot_byte_identical():
    """The natural control: a shot whose prose already agrees with its head count
    must render exactly as before, or the A/B is measuring collateral damage."""
    shot = _shot(
        shot_type="closeup", characters=["Zuri"],
        action="smiles at the camera",
        setting="a bright cheerful playroom with colorful toys",
    )
    prompt = _shot_prompt(shot, _song(), _cfg())

    assert "smiles at the camera, in a bright cheerful playroom with colorful toys" in prompt


def test_group_shots_keep_their_group_prose():
    """The repair must not singularise a genuine ensemble shot."""
    shot = _shot(
        id="s00", shot_type="wide", characters=["all"],
        action="The kids brush their teeth with big smiles",
        setting="a bright cheerful bathroom with a white pedestal sink",
    )
    prompt = _shot_prompt(shot, _song(), _cfg())

    assert "Exactly three children are on screen" in prompt
    assert "The kids brush their teeth" in prompt


# ------------------------------------------------ the fix at the source ---
def test_normalize_shots_repairs_a_single_character_shot_the_llm_wrote_group_prose_for():
    """The pre-existing closeup scrub only rewrote the action when it had to
    CHANGE `characters` first, so a closeup the LLM already gave exactly one name
    kept its group prose. That is the hole this closes."""
    song = _song()
    raw = {"shots": [
        {"verse": 0, "shot_type": "closeup", "characters": ["Zuri"],
         "action": "The kids smile brightly at each other", "camera": "static"},
    ]}
    shots = director._normalize_shots(raw, song, [(0.0, 8.0)], {}, _cfg())

    first = shots[0]
    assert first["characters"] == ["Zuri"]
    assert director.implied_child_count(first["action"]) <= 1
    assert director.setting_is_a_place(first["setting"])


def test_normalize_shots_output_always_passes_the_head_count_gate():
    song = _song()
    raw = {"shots": [
        {"verse": 0, "shot_type": "closeup", "characters": ["Zuri"],
         "action": "The three friends laugh together", "camera": "static"},
        {"verse": 0, "shot_type": "medium", "characters": ["Kofi"],
         "action": "Kofi and Nala wave at each other", "camera": "static"},
        {"verse": 0, "shot_type": "wide", "characters": ["all"],
         "action": "The kids dance and bounce to the music", "camera": "static"},
    ]}
    shots = director._normalize_shots(raw, song, [(0.0, 12.0)], {}, _cfg())

    offenders = [r for r in _check_shotlist(shots) if "head count contradiction" in r]
    assert offenders == [], offenders


def test_the_deterministic_fallback_planner_also_passes_the_head_count_gate():
    """The fallback planner runs whenever the LLM is unreachable, so its output
    must clear the same gate — otherwise an offline run builds a shot list its
    own QC stage rejects."""
    song = _song()
    song["verses"] = [
        {"scene": "a bathroom sink", "lines": ["brush your teeth", "wash up now", "sing along"]},
        {"scene": "a sunny backyard", "lines": ["clap your hands", "jump up high", "dance around"]},
    ]
    shots = director._fallback_planner(song, [(0.0, 12.0), (12.0, 24.0)], {}, _cfg())

    offenders = [
        r for r in _check_shotlist(shots)
        if "head count contradiction" in r or "setting is not a place" in r
    ]
    assert offenders == [], offenders


# ------------------------------------------------------- the QC gate ---
def test_qc_rejects_a_shot_whose_action_implies_more_children_than_it_lists():
    shots = [
        _shot(id="s00", shot_type="wide", characters=["all"], start=0.0, end=3.0,
              action="The kids wave at the camera"),
        _shot(id="s01", shot_type="closeup", characters=["Zuri"], start=3.0, end=6.0,
              action="The kids smile brightly at each other"),
    ]
    reasons = _check_shotlist(shots)

    hits = [r for r in reasons if "head count contradiction" in r]
    assert len(hits) == 1
    assert "s01" in hits[0]


def test_qc_rejects_a_setting_that_describes_people_instead_of_a_place():
    shots = [
        _shot(id="s00", shot_type="wide", characters=["all"], start=0.0, end=3.0,
              action="The kids wave at the camera",
              setting="Amira, Kofi, and Zuri are rinsing their mouths with water"),
    ]
    reasons = _check_shotlist(shots)

    assert any("setting is not a place" in r for r in reasons)


def test_qc_passes_a_consistent_shot_list():
    shots = [
        _shot(id="s00", shot_type="wide", characters=["all"], start=0.0, end=3.0,
              action="The kids wave at the camera"),
        _shot(id="s01", shot_type="closeup", characters=["Zuri"], start=3.0, end=6.0,
              action="Zuri smiles brightly at the camera"),
    ]
    reasons = _check_shotlist(shots)

    assert not any("head count contradiction" in r for r in reasons)
    assert not any("setting is not a place" in r for r in reasons)


# ---------------------------------------- story_subject never counts as a child ---
# `story_subject` (e.g. "a tiny round cartoon mouse", "a fluffy white lamb") names
# the non-child subject of a shot's little story. It is never a child and must
# never inflate an implied head count or make a shot with `characters == []` look
# like "a shot with no characters".
def test_story_subject_never_inflates_a_head_count():
    """A single-child shot whose action names a story subject (a mouse, alongside
    Zuri) must stay a clean 1-child shot — the mouse is not counted."""
    shots = [
        _shot(
            id="s00", shot_type="wide", characters=["Zuri"], start=0.0, end=3.0,
            action="Zuri watches as a tiny round cartoon mouse scampers past",
            story_subject="a tiny round cartoon mouse",
        ),
    ]
    reasons = _check_shotlist(shots)

    assert not any("head count contradiction" in r for r in reasons)


def test_shot_with_no_characters_and_a_story_subject_is_a_valid_shot():
    """`characters == []` is valid on its own when a non-null `story_subject` is
    the thing on screen — this must not be rejected as "a shot with no
    characters", and the action naming the story subject must pass cleanly."""
    shots = [
        _shot(
            id="s00", shot_type="wide", characters=[], start=0.0, end=3.0,
            action="a tiny round cartoon mouse scampers up the grandfather clock",
            setting="a sunny playroom with a tall grandfather clock in the corner",
            story_subject="a tiny round cartoon mouse",
        ),
        _shot(
            id="s01", shot_type="medium", characters=["Zuri"], start=3.0, end=6.0,
            action="Zuri points after the mouse and giggles",
            setting="a sunny playroom with a tall grandfather clock in the corner",
        ),
    ]
    reasons = _check_shotlist(shots)

    assert not any("head count contradiction" in r for r in reasons)
    assert not any("action missing/not verb-ish" in r for r in reasons)
    assert not any("setting is not a place" in r for r in reasons)
