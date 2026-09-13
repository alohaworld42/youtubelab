"""Tests for the de-creep gaze feature (kidsong.decreep.enabled, default False).

ROOT CAUSE (verified): every rendered toddler blankly stares into the camera
because generate.py's `_shot_prompt` appended "...face the camera as they
move" to EVERY child shot, director.py's `_PERFORMANCE_BEATS` pool included
"looking straight at the camera" as a repeated-action beat, and
`_ACTION_VERBS_SINGULAR["smile"]`/the "hold" entries in both verb tables said
"...at the camera"/"...to the camera". This file covers the fix:
  - `generate._gaze_sentence` — the per-shot replacement for the blanket
    camera sentence, gated by kidsong.decreep.enabled.
  - `generate._shot_prompt` decreep-on/off behaviour, including the
    byte-identical-when-off guarantee already pinned by
    tests/test_kidsong_shot_prompt.py and tests/test_kidsong_render_style.py.
  - `director._fallback_planner`'s budgeted hook-verse camera designation
    (the ONE shot per episode allowed to play to the lens on purpose).
  - `director._normalize_shots`'s validation of an LLM-proposed "gaze" field.
  - `director._PERFORMANCE_BEATS_V2` and the `_ACTION_VERBS`/
    `_ACTION_VERBS_SINGULAR` de-creep verb overlay.

CPU-only, no GPU, no network, no LLM calls -- these call `_fallback_planner`
and `_normalize_shots` directly with hand-built songs, the same way
tests/test_kidsong_director_normalize.py does.
"""
from pipeline.kidsong import director
from pipeline.kidsong.generate import _gaze_sentence, _shot_prompt

_MOUSE = "a tiny round cartoon mouse"


# =============================================================== _gaze_sentence ===
def test_gaze_sentence_explicit_gaze_plural():
    assert (
        _gaze_sentence({"gaze": "down at the blocks"}, None, 2)
        == "They look down at the blocks."
    )


def test_gaze_sentence_explicit_gaze_singular():
    assert (
        _gaze_sentence({"gaze": "down at the blocks"}, None, 1)
        == "The child looks down at the blocks."
    )


def test_gaze_sentence_camera_gaze_plural():
    assert _gaze_sentence({"gaze": "at the camera"}, None, 3) == (
        "They face the camera with warm natural smiles, blinking naturally."
    )


def test_gaze_sentence_camera_gaze_singular():
    assert _gaze_sentence({"gaze": "at the camera"}, None, 1) == (
        "The child faces the camera with a warm natural smile, blinking naturally."
    )


def test_gaze_sentence_story_subject_default_wins_over_count():
    """`story_subject` is the default regardless of count (no gaze field)."""
    assert (
        _gaze_sentence({}, _MOUSE, 2) == f"Their eyes are on {_MOUSE}."
    )
    assert (
        _gaze_sentence({}, _MOUSE, 1) == f"Their eyes are on {_MOUSE}."
    )


def test_gaze_sentence_each_other_default_for_no_subject_plural():
    assert _gaze_sentence({}, None, 2) == "The children look at each other as they play."
    # Count unknown (None) must not crash and must use the plural default too.
    assert _gaze_sentence({}, None, None) == "The children look at each other as they play."


def test_gaze_sentence_hands_default_for_no_subject_singular():
    assert _gaze_sentence({}, None, 1) == "The child looks at what their hands are doing."


def test_gaze_sentence_junk_values_fall_back_to_the_no_gaze_default():
    """A non-string, blank, absurdly long, or newline-carrying gaze must be
    treated exactly like an absent one -- never rendered verbatim."""
    for junk in (42, ["x"], {"a": 1}, "", "   ", "x" * 81, "line one\nline two"):
        assert (
            _gaze_sentence({"gaze": junk}, None, 1)
            == "The child looks at what their hands are doing."
        ), junk


def test_gaze_sentence_never_writes_a_negated_camera_phrase():
    """Belt-and-braces: no branch of the helper may ever emit "not ... camera"
    -- the text encoder is documented (see the helper's docstring) to read
    negation unreliably."""
    samples = [
        _gaze_sentence({"gaze": "at the camera"}, None, 1),
        _gaze_sentence({"gaze": "at the camera"}, None, 2),
        _gaze_sentence({"gaze": "down at the blocks"}, None, 1),
        _gaze_sentence({}, _MOUSE, 2),
        _gaze_sentence({}, None, 2),
        _gaze_sentence({}, None, 1),
    ]
    for text in samples:
        assert "not" not in text.lower()


# ================================================================ _shot_prompt ===
def _song():
    return {
        "characters": (
            "Three adorable Black toddlers: Zuri, a girl with afro puffs and a yellow "
            "shirt; Kofi, a boy with curly hair and a blue shirt; Nala, a girl with "
            "braids and a yellow dress."
        )
    }


def _kofi_closeup():
    return {
        "id": "s01", "shot_type": "closeup", "characters": ["Kofi"],
        "action": "The child brushes their teeth with a big smile",
        "setting": "a bright cheerful bathroom", "camera": "static",
    }


def _all_wide():
    return {
        "id": "s00", "shot_type": "wide", "characters": ["all"],
        "action": "The kids clap their hands together",
        "setting": "a sunny backyard", "camera": "static",
    }


def _prompt_cfg(decreep=True, hook_camera_budget=1):
    return {
        "kidsong": {
            "shot": {"style_trigger": "P1x4r"},
            "decreep": {"enabled": decreep, "hook_camera_budget": hook_camera_budget},
        }
    }


# NOTE: these baselines changed once, on 2026-07-31, and only here: the cast
# bible's `build` values were adjective phrases spliced after "with", so two of
# three characters described themselves in broken English ("a 4-year-old boy
# with noticeably tall and sturdy for a preschooler, with slightly broader
# shoulders"). QM-029/IMP-036 made them noun phrases. No descriptor was dropped
# and no prompt-composition code moved — if a future diff touches these strings
# for any other reason, that is a regression, not a rebaseline.
# The exact pre-existing (flag-off) output, pinned identically in
# tests/test_kidsong_shot_prompt.py and tests/test_kidsong_render_style.py.
_CLOSEUP_BASELINE = (
    "P1x4r, a high-quality 3D CGI toon animation in a preschool show "
    "style. A closeup shot of exactly one child: Kofi, a 4-year-old boy "
    "with a tall, sturdy build for a preschooler and slightly broader "
    "shoulders, medium-deep brown skin, short dark natural curly hair, "
    "wearing a blue t-shirt, grey jogger shorts and red sneakers. The "
    "child brushes their teeth with a big smile, in a bright cheerful "
    "bathroom. No other children are visible in frame. Every child on "
    "screen is one of these Black toddlers — no other children appear. The "
    "camera holds steady at the children's eye level. The children have "
    "soft rounded shapes and big expressive eyes and face the camera as "
    "they move. A warm key light from one side with a soft cool rim light "
    "along hair and shoulders, gentle soft-edged shadows. The set is "
    "lovingly dressed with two or three simple props that fit the "
    "location, kept clear around the children — bold happy colors, a "
    "joyful everyday toddler moment."
)

_WIDE_BASELINE = (
    "P1x4r, a high-quality 3D CGI toon animation in a preschool show "
    "style. Exactly three children are on screen: Zuri, a 3-year-old girl "
    "with an average toddler height and a soft, rounded little build, deep "
    "warm brown skin, round afro puffs tied with soft yellow fabric "
    "scrunchies, wearing a sunny yellow t-shirt, denim dungaree shorts and "
    "white sneakers; Kofi, a 4-year-old boy with a tall, sturdy build for "
    "a preschooler and slightly broader shoulders, medium-deep brown skin, "
    "short dark natural curly hair, wearing a blue t-shirt, grey jogger "
    "shorts and red sneakers; and Nala, a 3-year-old girl with a petite, "
    "delicate build, a little shorter and slighter, deep brown skin, neat "
    "cornrow braids, wearing a yellow pinafore dress over a white t-shirt, "
    "white tights and pink sneakers. A wide shot: The kids clap their "
    "hands together, in a sunny backyard. No other children appear "
    "anywhere in the frame, foreground or background. Every child on "
    "screen is one of these Black toddlers — no other children appear. The "
    "camera holds steady at the children's eye level. The children have "
    "soft rounded shapes and big expressive eyes and face the camera as "
    "they move. A warm key light from one side with a soft cool rim light "
    "along hair and shoulders, gentle soft-edged shadows. The set is "
    "lovingly dressed with two or three simple props that fit the "
    "location, kept clear around the children — bold happy colors, a "
    "joyful everyday toddler moment."
)


def test_decreep_on_closeup_prompt_has_gaze_sentence_not_legacy_camera_line():
    prompt = _shot_prompt(_kofi_closeup(), _song(), _prompt_cfg(decreep=True))
    assert "face the camera as they move" not in prompt
    assert "The children have soft rounded shapes and big expressive eyes." in prompt
    # No gaze/story_subject on this shot, count == 1 -> the "hands" default.
    assert "The child looks at what their hands are doing." in prompt


def test_decreep_on_wide_prompt_has_gaze_sentence_not_legacy_camera_line():
    prompt = _shot_prompt(_all_wide(), _song(), _prompt_cfg(decreep=True))
    assert "face the camera as they move" not in prompt
    assert "The children have soft rounded shapes and big expressive eyes." in prompt
    # No gaze/story_subject, count == 3 -> the "each other" default.
    assert "The children look at each other as they play." in prompt


def test_decreep_on_respects_an_explicit_shot_gaze_field():
    shot = _kofi_closeup()
    shot["gaze"] = "down at the toothbrush"
    prompt = _shot_prompt(shot, _song(), _prompt_cfg(decreep=True))
    assert "The child looks down at the toothbrush." in prompt
    assert "face the camera as they move" not in prompt


def test_decreep_off_is_byte_identical_to_the_pinned_baseline():
    """Flag off (explicit False, and absent entirely) must reproduce the
    exact pre-existing sentence -- the same guarantee pinned in
    tests/test_kidsong_shot_prompt.py and tests/test_kidsong_render_style.py."""
    assert _shot_prompt(_kofi_closeup(), _song(), _prompt_cfg(decreep=False)) == _CLOSEUP_BASELINE
    assert _shot_prompt(_all_wide(), _song(), _prompt_cfg(decreep=False)) == _WIDE_BASELINE

    minimal_cfg = {"kidsong": {"shot": {"style_trigger": "P1x4r"}}}
    assert _shot_prompt(_kofi_closeup(), _song(), minimal_cfg) == _CLOSEUP_BASELINE
    assert _shot_prompt(_all_wide(), _song(), minimal_cfg) == _WIDE_BASELINE


def test_decreep_off_ignores_an_explicit_shot_gaze_field():
    """The gaze field only ever affects rendering when the flag is on -- a
    shot that happens to carry `gaze` (e.g. left over from a decreep-on run)
    must not change the flag-off prompt at all."""
    shot = _kofi_closeup()
    shot["gaze"] = "down at the toothbrush"
    assert _shot_prompt(shot, _song(), _prompt_cfg(decreep=False)) == _CLOSEUP_BASELINE


# ========================================= _fallback_planner gaze designation ===
def _narrative_then_hook_song():
    """Verse 0 is a plain narrative verse (long, unique lines, no hook phrase
    or refrain cue); verse 1 is a hook/refrain verse (`_is_hook_verse` matches
    "clap along" and "sing with me")."""
    return {
        "characters": "Zuri, Kofi and Nala are best friends",
        "verses": [
            {"scene": "a backyard", "lines": [
                "The three friends walk slowly across the sunny yard",
                "They stop to look at a little brown rabbit",
                "Zuri waves and smiles at her happy friends",
            ]},
            {"scene": "a backyard", "lines": ["Clap along and sing with me"]},
        ],
    }


def _two_hook_verse_song():
    """Two DIFFERENT hook verses (different lyrics, so neither is a
    chorus-reuse of the other) -- both independently qualify via
    `_HOOK_PHRASES`."""
    return {
        "characters": "Zuri, Kofi and Nala are best friends",
        "verses": [
            {"scene": "a backyard", "lines": ["Clap your hands with me"]},
            {"scene": "a playroom", "lines": ["Come sing and count to three"]},
        ],
    }


def _decreep_director_cfg(enabled=True, hook_camera_budget=1, seed=20260717):
    return {
        "kidsong": {
            "seed": seed,
            "decreep": {"enabled": enabled, "hook_camera_budget": hook_camera_budget},
        }
    }


_VERSE_TIMES_2x8 = [(0.0, 8.0), (8.0, 16.0)]


def test_fallback_planner_decreep_on_designates_first_shot_of_first_hook_verse():
    song = _narrative_then_hook_song()
    skeleton = director._fallback_planner(song, _VERSE_TIMES_2x8, {}, _decreep_director_cfg())

    verse0_shots = [s for s in skeleton if s["verse"] == 0]
    verse1_shots = [s for s in skeleton if s["verse"] == 1]
    assert verse0_shots and verse1_shots  # sanity: both verses produced shots

    assert all("gaze" not in s for s in verse0_shots)
    assert verse1_shots[0].get("gaze") == "at the camera"
    assert all("gaze" not in s for s in verse1_shots[1:])


def test_fallback_planner_decreep_off_never_sets_a_gaze_key():
    song = _narrative_then_hook_song()
    skeleton = director._fallback_planner(
        song, _VERSE_TIMES_2x8, {}, _decreep_director_cfg(enabled=False)
    )
    assert all("gaze" not in s for s in skeleton)

    # A cfg that never mentions kidsong.decreep at all behaves identically.
    plain_cfg = {"kidsong": {"seed": 20260717}}
    skeleton2 = director._fallback_planner(song, _VERSE_TIMES_2x8, {}, plain_cfg)
    assert all("gaze" not in s for s in skeleton2)


def test_fallback_planner_decreep_zero_budget_designates_nothing():
    song = _narrative_then_hook_song()
    skeleton = director._fallback_planner(
        song, _VERSE_TIMES_2x8, {}, _decreep_director_cfg(hook_camera_budget=0)
    )
    assert all("gaze" not in s for s in skeleton)


def test_fallback_planner_decreep_never_designates_a_non_hook_verse():
    """A song with NO hook verse at all must never get a gaze designation --
    the mechanism must not fall back to just "the first verse"."""
    song = {
        "characters": "Zuri, Kofi and Nala are best friends",
        "verses": [
            {"scene": "a backyard", "lines": [
                "The three friends walk slowly across the sunny yard",
                "They stop to look at a little brown rabbit",
                "Zuri waves and smiles at her happy friends",
            ]},
            {"scene": "a playroom", "lines": [
                "Kofi builds a tower of colorful blocks",
                "Nala claps when the tower gets tall",
            ]},
        ],
    }
    skeleton = director._fallback_planner(song, _VERSE_TIMES_2x8, {}, _decreep_director_cfg())
    assert all("gaze" not in s for s in skeleton)


def test_fallback_planner_decreep_budget_one_only_designates_the_first_hook_verse():
    song = _two_hook_verse_song()
    skeleton = director._fallback_planner(
        song, _VERSE_TIMES_2x8, {}, _decreep_director_cfg(hook_camera_budget=1)
    )
    verse0_shots = [s for s in skeleton if s["verse"] == 0]
    verse1_shots = [s for s in skeleton if s["verse"] == 1]
    assert verse0_shots[0].get("gaze") == "at the camera"
    assert all("gaze" not in s for s in verse0_shots[1:])
    assert all("gaze" not in s for s in verse1_shots)


def test_fallback_planner_decreep_budget_two_designates_both_hook_verses():
    """Documents the budget mechanism generalizing beyond the default of 1 --
    "capped by hook_camera_budget" rather than hardcoded to a single verse."""
    song = _two_hook_verse_song()
    skeleton = director._fallback_planner(
        song, _VERSE_TIMES_2x8, {}, _decreep_director_cfg(hook_camera_budget=2)
    )
    verse0_shots = [s for s in skeleton if s["verse"] == 0]
    verse1_shots = [s for s in skeleton if s["verse"] == 1]
    assert verse0_shots[0].get("gaze") == "at the camera"
    assert verse1_shots[0].get("gaze") == "at the camera"


# ============================================ _normalize_shots gaze overlay ===
def test_normalize_shots_llm_gaze_overlay_accepted_when_not_camera():
    song = _narrative_then_hook_song()
    cfg = _decreep_director_cfg()
    raw = {"shots": [{"verse": 0, "gaze": "down at their hands"}]}
    result = director._normalize_shots(raw, song, _VERSE_TIMES_2x8, {}, cfg)
    s00 = next(s for s in result if s["id"] == "s00")
    assert s00["gaze"] == "down at their hands"


def test_normalize_shots_llm_gaze_placeholder_ellipsis_dropped():
    """The LLM echoes the response template's "gaze": "..." exactly the way it
    echoes "action": "..." (see director.is_placeholder_action). A literal
    ellipsis gaze must be treated as ABSENT — it otherwise renders the nonsense
    sentence "They look ...". Measured live in the plan-A/B demo (arm B shots
    s00/s05 carried gaze "...")."""
    song = _narrative_then_hook_song()
    cfg = _decreep_director_cfg()
    for junk in ("...", "…", " ... ", "-", "??"):
        raw = {"shots": [{"verse": 0, "gaze": junk}]}
        result = director._normalize_shots(raw, song, _VERSE_TIMES_2x8, {}, cfg)
        s00 = next(s for s in result if s["id"] == "s00")
        assert s00.get("gaze") != junk, f"placeholder gaze {junk!r} leaked through"


def test_normalize_shots_llm_gaze_camera_dropped_without_hook_designation():
    """Verse 0 is the narrative (non-hook) verse -- `_fallback_planner` never
    designates its first shot, so the LLM proposing "at the camera" here must
    be dropped rather than honored."""
    song = _narrative_then_hook_song()
    cfg = _decreep_director_cfg()
    raw = {"shots": [{"verse": 0, "gaze": "at the camera"}]}
    result = director._normalize_shots(raw, song, _VERSE_TIMES_2x8, {}, cfg)
    s00 = next(s for s in result if s["id"] == "s00")
    assert s00.get("gaze") != "at the camera"


def test_normalize_shots_llm_gaze_camera_drop_is_logged(caplog):
    song = _narrative_then_hook_song()
    cfg = _decreep_director_cfg()
    raw = {"shots": [{"verse": 0, "gaze": "at the camera"}]}
    with caplog.at_level("INFO", logger="kidsong.director"):
        director._normalize_shots(raw, song, _VERSE_TIMES_2x8, {}, cfg)
    assert any("dropped" in r.message.lower() for r in caplog.records)


def test_normalize_shots_llm_gaze_camera_kept_on_the_designated_hook_shot():
    song = _narrative_then_hook_song()
    cfg = _decreep_director_cfg()
    skeleton = director._fallback_planner(song, _VERSE_TIMES_2x8, {}, cfg)
    hook_shot = next(s for s in skeleton if s["verse"] == 1)
    assert hook_shot["gaze"] == "at the camera"  # sanity: step 5 already did this

    raw = {"shots": [{"verse": 1, "gaze": "at the camera"}]}
    result = director._normalize_shots(raw, song, _VERSE_TIMES_2x8, {}, cfg)
    result_shot = next(s for s in result if s["id"] == hook_shot["id"])
    assert result_shot["gaze"] == "at the camera"


def test_normalize_shots_llm_gaze_junk_values_are_ignored():
    song = _narrative_then_hook_song()
    cfg = _decreep_director_cfg()
    for junk in (42, ["x"], {"a": 1}, "", "   ", "x" * 81, "line one\nline two"):
        raw = {"shots": [{"verse": 0, "gaze": junk}]}
        result = director._normalize_shots(raw, song, _VERSE_TIMES_2x8, {}, cfg)
        s00 = next(s for s in result if s["id"] == "s00")
        assert "gaze" not in s00, junk


def test_normalize_shots_decreep_off_ignores_the_llm_gaze_field_entirely():
    song = _narrative_then_hook_song()
    cfg = _decreep_director_cfg(enabled=False)
    raw = {"shots": [{"verse": 0, "gaze": "down at their hands"}]}
    result = director._normalize_shots(raw, song, _VERSE_TIMES_2x8, {}, cfg)
    assert all("gaze" not in s for s in result)


def test_normalize_shots_with_no_llm_data_still_carries_the_hook_designation():
    """The realistic "LLM unreachable" path: `_normalize_shots` falls straight
    back to the skeleton, which must still carry step 5's designation."""
    song = _narrative_then_hook_song()
    cfg = _decreep_director_cfg()
    result = director._normalize_shots(None, song, _VERSE_TIMES_2x8, {}, cfg)
    hook_shots = [s for s in result if s["verse"] == 1]
    assert hook_shots[0]["gaze"] == "at the camera"


# ========================================================= de-creep verb pools ===
def test_performance_beats_v2_contains_no_camera_or_lens_language():
    for beat in director._PERFORMANCE_BEATS_V2:
        assert "camera" not in beat.lower()
        assert "lens" not in beat.lower()


def test_performance_beats_v2_is_used_only_when_decreep_is_on():
    """`vary_repeated_actions` must pick the legacy pool by default (so a
    caller that never passes `decreep` -- e.g. tests/test_scene_coherence.py
    -- is unaffected) and the V2 pool only when explicitly asked."""
    shots_off = [
        {"verse": 0, "characters": ["Zuri"], "action": "Zuri sings along with a happy face", "reuse_of": None},
        {"verse": 0, "characters": ["Kofi"], "action": "Kofi sings along with a happy face", "reuse_of": None},
    ]
    director.vary_repeated_actions(shots_off)
    assert "looking straight at the camera" in shots_off[1]["action"]

    shots_on = [
        {"verse": 0, "characters": ["Zuri"], "action": "Zuri sings along with a happy face", "reuse_of": None},
        {"verse": 0, "characters": ["Kofi"], "action": "Kofi sings along with a happy face", "reuse_of": None},
    ]
    director.vary_repeated_actions(shots_on, decreep=True)
    assert "camera" not in shots_on[1]["action"].lower()
    assert "leaning in to look closer" in shots_on[1]["action"]


def test_action_verb_v2_overlays_contain_no_camera_or_lens_language():
    for table in (director._ACTION_VERBS_SINGULAR_V2, director._ACTION_VERBS_V2):
        for phrase in table.values():
            assert "camera" not in phrase.lower()
            assert "lens" not in phrase.lower()


def test_action_verb_table_overlay_touches_only_the_camera_entries():
    base_singular = director._action_verb_table(singular=True, decreep=False)
    v2_singular = director._action_verb_table(singular=True, decreep=True)
    assert base_singular["smile"] == "smiles brightly at the camera"
    assert "camera" not in v2_singular["smile"]
    assert "camera" not in v2_singular["hold"]
    # Every OTHER entry is untouched by the overlay.
    for verb in base_singular:
        if verb in ("smile", "hold"):
            continue
        assert v2_singular[verb] == base_singular[verb]


def test_simplify_closeup_action_decreep_on_smile_has_no_camera_language():
    out = director._simplify_closeup_action(
        "the kids smile brightly at each other", "Zuri", decreep=True
    )
    assert out == "Zuri smiles brightly at their friends"


def test_simplify_closeup_action_decreep_off_keeps_the_legacy_camera_phrase():
    """Byte-identical regression pin for the flag-off path."""
    out = director._simplify_closeup_action("the kids smile brightly at each other", "Zuri")
    assert out == "Zuri smiles brightly at the camera"


def test_line_action_decreep_on_hold_has_no_camera_language():
    action = director._line_action(
        ["Kofi holds up the toy"], (0, 1), subject="Kofi", decreep=True
    )
    assert action == "Kofi holds up the hero prop proudly for the others to see"


def test_line_action_decreep_off_keeps_the_legacy_camera_phrase():
    action = director._line_action(["Kofi holds up the toy"], (0, 1), subject="Kofi")
    assert action == "Kofi holds up the hero prop and shows it to the camera"


def test_reduce_scene_to_shared_activity_decreep_on_hold_has_no_camera_language():
    mixed = (
        "Zuri holds a teddy bear, Kofi stomps in the grass, "
        "Nala dances with arms outstretched"
    )
    reduced = director.reduce_scene_to_shared_activity(mixed, decreep=True)
    assert reduced == "The kids hold up the hero prop proudly for the others to see"


def test_reduce_scene_to_shared_activity_decreep_off_keeps_the_legacy_camera_phrase():
    mixed = (
        "Zuri holds a teddy bear, Kofi stomps in the grass, "
        "Nala dances with arms outstretched"
    )
    reduced = director.reduce_scene_to_shared_activity(mixed)
    assert reduced == "The kids hold up the hero prop and show it to the camera"
