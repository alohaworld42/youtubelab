"""Tests for the character-consistency fix to the render-time prompt/negative/seed
machinery in pipeline.kidsong.generate and pipeline.kidsong.director.

Before this change `_cast_for` and `_shot_prompt` had ZERO test coverage, despite
being the biggest lever on the measured failure modes: identity/cast breaks (9/54
reviewed takes, 67% in wide/group shots — an invented extra child) and wardrobe
breaks (3/54, 100% in closeups — wrong shirt colour). This file covers the fix:
  - `_cast_for` now sources identity text from the cast bible (pipeline.kidsong.cast)
    instead of fragile free-text parsing, with a crash-proof legacy fallback.
  - `_shot_prompt` is now subject-first (identity clause before action/setting) and
    states an explicit head count (one child for closeups/mediums, exact N for
    wide/group), on top of the pre-existing "no other children" sentence.
  - per-shot NEGATIVE is now patched dynamically with character-specific terms.
  - seeds (planned and retry) are now keyed to the cast bible's per-character
    seed_offset, so a given child keeps the same seed family across shots,
    episodes, and retries.

CPU-only, no network, no GPU, no ffmpeg, no ComfyUI. The real cast bible
(prompts/cast_bible.json) is used for the positive-path tests since it already
exists on disk and is stdlib-only to load; the "cast bible unavailable"
tests force an ImportError via sys.modules, following the same pattern
tests/test_kidsong_review_cast.py established for this exact scenario.
"""
import sys

import pytest

from pipeline.kidsong.director import _char_hash, _seed_for_characters
from pipeline.kidsong.generate import (
    _baseline_negative,
    _cast_for,
    _cast_for_with_count,
    _retry_seed,
    _shot_negative,
    _shot_prompt,
)

_CAST_MODULE_NAME = "pipeline.kidsong.cast"


# ------------------------------------------------------------------ fixtures ---
@pytest.fixture
def no_cast_module(monkeypatch):
    """Force `from pipeline.kidsong import cast` to raise ImportError everywhere,
    regardless of whether the real cast.py exists on disk.

    Both halves are needed: the sys.modules entry (the documented way to make
    the import system treat a name as unimportable) AND the cached package
    attribute (a `from pipeline.kidsong import cast` that already ran once in
    this test session — e.g. any test above that imported the real module —
    would otherwise short-circuit straight to the cached attribute and never
    re-consult sys.modules at all).
    """
    import pipeline.kidsong as kidsong_pkg

    monkeypatch.delattr(kidsong_pkg, "cast", raising=False)
    monkeypatch.setitem(sys.modules, _CAST_MODULE_NAME, None)


def _song():
    return {
        "characters": (
            "Three adorable Black toddlers: Zuri, a girl with afro puffs and a yellow "
            "shirt; Kofi, a boy with curly hair and a blue shirt; Nala, a girl with "
            "braids and a yellow dress."
        )
    }


def _cfg():
    return {"kidsong": {"shot": {"style_trigger": "P1x4r"}}}


def _kofi_closeup():
    return {
        "id": "s01",
        "shot_type": "closeup",
        "characters": ["Kofi"],
        "action": "The child brushes their teeth with a big smile",
        "setting": "a bright cheerful bathroom",
        "camera": "static",
    }


def _all_wide():
    return {
        "id": "s00",
        "shot_type": "wide",
        "characters": ["all"],
        "action": "The kids clap their hands together",
        "setting": "a sunny backyard",
        "camera": "static",
    }


def _insert_shot():
    return {
        "id": "s05",
        "shot_type": "insert",
        "characters": [],
        "action": "a bright red toothbrush",
        "setting": "on the bathroom sink",
    }


# --------------------------------------------------------------- _cast_for ---
def test_cast_for_single_character_uses_the_cast_bible():
    result = _cast_for(_kofi_closeup(), _song())
    assert "Kofi" in result
    assert "Zuri" not in result
    assert "Nala" not in result


def test_cast_for_degrades_gracefully_without_raising_when_bible_unavailable(no_cast_module):
    shot = {"id": "s01", "characters": ["Kofi"]}
    result = _cast_for(shot, _song())  # must not raise
    assert "Kofi" in result
    assert "Zuri" not in result
    assert "Nala" not in result


# ------------------------------------------------------------- _shot_prompt ---
def test_closeup_single_character_names_only_that_child():
    """The crowding regression: a Kofi closeup must not mention Zuri or Nala."""
    prompt = _shot_prompt(_kofi_closeup(), _song(), _cfg())
    assert "Kofi" in prompt
    assert "Zuri" not in prompt
    assert "Nala" not in prompt


def test_closeup_single_character_states_one_child_count_and_clamp():
    prompt = _shot_prompt(_kofi_closeup(), _song(), _cfg())
    assert "exactly one child" in prompt.lower()
    assert "No other children are visible in frame." in prompt


def test_wide_all_states_exact_count_and_foreground_background_clamp():
    prompt = _shot_prompt(_all_wide(), _song(), _cfg())
    for name in ("Zuri", "Kofi", "Nala"):
        assert name in prompt
    assert "Exactly three children are on screen" in prompt
    assert "No other children appear anywhere in the frame, foreground or background." in prompt


def test_anti_regression_sentence_present_in_closeup_and_wide():
    literal = "Every child on screen is one of these Black toddlers — no other children appear."
    assert literal in _shot_prompt(_kofi_closeup(), _song(), _cfg())
    assert literal in _shot_prompt(_all_wide(), _song(), _cfg())


def test_identity_clause_appears_before_action_text_closeup():
    shot = _kofi_closeup()
    prompt = _shot_prompt(shot, _song(), _cfg())
    assert prompt.index("Kofi") < prompt.index("brushes their teeth")


def test_identity_clause_appears_before_action_text_wide():
    shot = _all_wide()
    prompt = _shot_prompt(shot, _song(), _cfg())
    assert prompt.index("Exactly three children are on screen") < prompt.index("clap their hands")


def test_insert_shot_still_produces_a_no_people_prompt():
    prompt = _shot_prompt(_insert_shot(), _song(), _cfg())
    assert "no people" in prompt.lower()
    assert "Kofi" not in prompt
    assert "Zuri" not in prompt
    assert "Nala" not in prompt


def test_shot_prompt_does_not_raise_without_cast_bible(no_cast_module):
    """Whole-prompt degrade path: a broken/missing bible must never crash a render."""
    prompt = _shot_prompt(_kofi_closeup(), _song(), _cfg())
    assert "Kofi" in prompt
    prompt_wide = _shot_prompt(_all_wide(), _song(), _cfg())
    assert isinstance(prompt_wide, str) and prompt_wide


# ------------------------------------------------------------ story_subject ---
# The song's actual protagonists (a mouse, a lamb, a sheep, a boat) never used
# to appear on screen — only the children did. `story_subject` is a new,
# optional shot field (str or None) naming the non-child subject visible in
# a shot; it is NEVER a child and NEVER counted in the head count. These tests
# cover the three shapes `_shot_prompt` must produce: no subject (byte-
# identical to the pre-existing behaviour, pinned above), children + subject,
# and subject-only (`characters == []`, only the mouse on screen).
_MOUSE = "a tiny round cartoon mouse"


def _kofi_closeup_with_mouse():
    shot = _kofi_closeup()
    shot["story_subject"] = _MOUSE
    return shot


def _mouse_only_shot():
    return {
        "id": "s09",
        "shot_type": "medium",
        "characters": [],
        "story_subject": _MOUSE,
        "action": "A tiny round cartoon mouse scampers up the tall grandfather clock",
        "setting": "a cozy nursery with a large wooden grandfather clock",
        "camera": "static",
    }


def test_no_story_subject_is_byte_identical_to_the_pinned_baseline():
    """Pin: a shot with no `story_subject` key at all must render exactly the
    same closeup/wide prompts the pre-existing (unmodified) implementation
    produced — captured from the real cast bible (prompts/cast_bible.json)
    before this change, and confirmed unchanged by re-running the pre-existing
    identity/count/anti-regression tests above after this change landed."""
    # Baseline updated 2026-07-22 for cast bible v1.2.0: every character now
    # carries a `build` (anatomy) fragment between age and skin — Kofi tall &
    # sturdy, Nala petite, Zuri average — so the three read as individuals of
    # different sizes instead of identically-proportioned clones. That fragment
    # is part of the canonical description everywhere by design (see
    # cast._describe_dict), so it belongs IN this pin, not around it.
    #
    # Baseline updated again 2026-07-31 for cast bible v1.2.1 (QM-029/IMP-036):
    # that `build` fragment is spliced in after "with", which takes a NOUN
    # phrase — and Kofi's and Nala's were written as ADJECTIVE phrases, so the
    # identity sentence read "a 4-year-old boy with noticeably tall and sturdy
    # for a preschooler, with slightly broader shoulders". No descriptor was
    # dropped and no composition code moved; only the bible's wording changed.
    closeup_baseline = (
        "P1x4r, a high-quality 3D CGI toon animation in a preschool show "
        "style. A closeup shot of exactly one child: Kofi, a 4-year-old "
        "boy with a tall, sturdy build for a preschooler and slightly "
        "broader shoulders, medium-deep brown skin, short dark natural "
        "curly hair, wearing a blue t-shirt, grey jogger shorts and red "
        "sneakers. The child brushes their teeth with a big smile, in a "
        "bright cheerful bathroom. No other children are visible in frame. "
        "Every child on screen is one of these Black toddlers — no other "
        "children appear. The camera holds steady at the children's eye "
        "level. The children have soft rounded shapes and big expressive "
        "eyes and face the camera as they move. A warm key light from one "
        "side with a soft cool rim light along hair and shoulders, gentle "
        "soft-edged shadows. The set is lovingly dressed with two or three "
        "simple props that fit the location, kept clear around the "
        "children — bold happy colors, a joyful everyday toddler moment."
    )
    assert _shot_prompt(_kofi_closeup(), _song(), _cfg()) == closeup_baseline

    wide_baseline = (
        "P1x4r, a high-quality 3D CGI toon animation in a preschool show "
        "style. Exactly three children are on screen: Zuri, a 3-year-old "
        "girl with an average toddler height and a soft, rounded little "
        "build, deep warm brown skin, round afro puffs tied with soft "
        "yellow fabric scrunchies, wearing a sunny yellow t-shirt, denim "
        "dungaree shorts and white sneakers; Kofi, a 4-year-old boy with a "
        "tall, sturdy build for a preschooler and slightly broader "
        "shoulders, medium-deep brown skin, short dark natural curly hair, "
        "wearing a blue t-shirt, grey jogger shorts and red sneakers; and "
        "Nala, a 3-year-old girl with a petite, delicate build, a little "
        "shorter and slighter, deep brown skin, neat cornrow braids, "
        "wearing a yellow pinafore dress over a white t-shirt, white "
        "tights and pink sneakers. A wide shot: The kids clap their hands "
        "together, in a sunny backyard. No other children appear anywhere "
        "in the frame, foreground or background. Every child on screen is "
        "one of these Black toddlers — no other children appear. The "
        "camera holds steady at the children's eye level. The children "
        "have soft rounded shapes and big expressive eyes and face the "
        "camera as they move. A warm key light from one side with a soft "
        "cool rim light along hair and shoulders, gentle soft-edged "
        "shadows. The set is lovingly dressed with two or three simple "
        "props that fit the location, kept clear around the children — "
        "bold happy colors, a joyful everyday toddler moment."
    )
    assert _shot_prompt(_all_wide(), _song(), _cfg()) == wide_baseline


def test_no_story_subject_none_or_missing_are_equivalent():
    """`story_subject: None` (explicit) and an absent key must render identically."""
    shot_missing = _kofi_closeup()
    shot_none = _kofi_closeup()
    shot_none["story_subject"] = None
    assert _shot_prompt(shot_missing, _song(), _cfg()) == _shot_prompt(shot_none, _song(), _cfg())


def test_children_plus_story_subject_keeps_the_head_count_sentence_verbatim():
    prompt = _shot_prompt(_kofi_closeup_with_mouse(), _song(), _cfg())
    assert "A closeup shot of exactly one child: Kofi, a 4-year-old boy" in prompt
    assert "No other children are visible in frame." in prompt
    assert (
        "Every child on screen is one of these Black toddlers — no other children appear."
        in prompt
    )


def test_children_plus_story_subject_adds_the_subject_sentence_before_the_clamp():
    prompt = _shot_prompt(_kofi_closeup_with_mouse(), _song(), _cfg())
    subject_sentence = "A tiny round cartoon mouse is also in the shot."
    assert subject_sentence in prompt
    # Subject sentence must land after the identity clause but before both the
    # "no other children" clamp and the verbatim anti-regression sentence, so
    # neither reads as forbidding the mouse.
    assert prompt.index("exactly one child: Kofi") < prompt.index(subject_sentence)
    assert prompt.index(subject_sentence) < prompt.index("No other children are visible in frame.")
    assert prompt.index(subject_sentence) < prompt.index(
        "Every child on screen is one of these Black toddlers — no other children appear."
    )


def test_children_plus_story_subject_does_not_change_the_head_count():
    with_mouse = _shot_prompt(_kofi_closeup_with_mouse(), _song(), _cfg())
    without_mouse = _shot_prompt(_kofi_closeup(), _song(), _cfg())
    assert "exactly one child" in with_mouse.lower()
    assert "exactly one child" in without_mouse.lower()
    # Only the added subject sentence differs; the rest of the prompt (in
    # particular the head-count language) is identical either way.
    assert with_mouse.replace(" A tiny round cartoon mouse is also in the shot.", "") == without_mouse


def test_cast_for_with_count_ignores_story_subject_entirely():
    """`_cast_for_with_count` must never treat `story_subject` as a child: the
    count for a shot is identical whether or not a story_subject is present."""
    _, count_with = _cast_for_with_count(_kofi_closeup_with_mouse(), _song())
    _, count_without = _cast_for_with_count(_kofi_closeup(), _song())
    assert count_with == count_without == 1

    wide_with_mouse = _all_wide()
    wide_with_mouse["story_subject"] = _MOUSE
    _, wide_count_with = _cast_for_with_count(wide_with_mouse, _song())
    _, wide_count_without = _cast_for_with_count(_all_wide(), _song())
    assert wide_count_with == wide_count_without == 3


def test_subject_only_shot_has_no_head_count_or_identity_sentence():
    prompt = _shot_prompt(_mouse_only_shot(), _song(), _cfg())
    assert "exactly" not in prompt.lower()
    assert "child" not in prompt.lower()
    assert "toddler" not in prompt.lower()
    assert "Kofi" not in prompt
    assert "Zuri" not in prompt
    assert "Nala" not in prompt


def test_subject_only_shot_states_the_subject_and_a_no_people_clamp():
    prompt = _shot_prompt(_mouse_only_shot(), _song(), _cfg())
    assert "A tiny round cartoon mouse is the subject of this medium shot" in prompt
    assert "scampers up the tall grandfather clock" in prompt
    assert "grandfather clock" in prompt  # setting survives too
    assert "no people" in prompt.lower()


def test_subject_only_shot_does_not_crash_with_characters_none():
    shot = _mouse_only_shot()
    shot["characters"] = None
    prompt = _shot_prompt(shot, _song(), _cfg())
    assert "mouse" in prompt.lower()


def test_subject_only_shot_respects_camera_field():
    shot = _mouse_only_shot()
    shot["camera"] = "slow push-in"
    prompt = _shot_prompt(shot, _song(), _cfg())
    assert "The camera moves in a slow push-in." in prompt
    assert "children's eye level" not in prompt


def test_story_subject_none_empty_or_non_string_are_ignored_safely():
    for bad in (None, "", "   ", 42, ["a mouse"], {"x": 1}):
        shot = _kofi_closeup()
        shot["story_subject"] = bad
        prompt = _shot_prompt(shot, _song(), _cfg())
        assert "is also in the shot" not in prompt
        assert "Kofi" in prompt


def test_story_subject_is_truncated_when_absurdly_long():
    shot = _kofi_closeup()
    shot["story_subject"] = "a mouse " + ("x" * 500)
    prompt = _shot_prompt(shot, _song(), _cfg())
    # Rendered subject text is capped at 120 chars, so the 500-char tail must
    # not appear verbatim in the prompt.
    assert "x" * 500 not in prompt
    assert "is also in the shot." in prompt


def test_insert_shot_unaffected_by_story_subject_field():
    """The insert branch must keep working exactly as before, even if a shot
    somehow carries a story_subject (insert shots are object closeups; the
    field is meaningless there and must be ignored, not rendered)."""
    shot = _insert_shot()
    shot["story_subject"] = _MOUSE
    prompt = _shot_prompt(shot, _song(), _cfg())
    assert "no people" in prompt.lower()
    assert "mouse" not in prompt.lower()
    assert "is also in the shot" not in prompt


# ---------------------------------------------- canonical subject_description ---
# EPISODE CANON (2026-07-24): prompts/pd_songs*.json now carries a per-song
# `subject_description` — a fixed, full description (colour/material/size
# relative to the children) of the song's non-child story subject, exactly the
# `identity_clause` idea applied to the sheep/lamb/star/mouse/boat instead of
# the cast. When present it REPLACES the bare `story_subject` noun phrase
# wherever the prompt introduces the subject, so it can never quietly drift
# from shot to shot the way a short noun phrase alone did (measured: a "black
# sheep" rendering pale/white in some keyframes of the same episode).
_SHEEP_DESCRIPTION = (
    "a small round woolly BLACK sheep, knee-high to the children, with a "
    "soft black face, black ears and stubby black legs"
)


def _song_with_subject_description():
    song = _song()
    song["subject_description"] = _SHEEP_DESCRIPTION
    return song


def test_children_plus_story_subject_uses_the_canonical_description_when_present():
    shot = _kofi_closeup_with_mouse()
    shot["story_subject"] = "a round woolly black sheep"
    prompt = _shot_prompt(shot, _song_with_subject_description(), _cfg())
    expected = f"{_SHEEP_DESCRIPTION[0].upper()}{_SHEEP_DESCRIPTION[1:]} is also in the shot."
    assert expected in prompt
    # the shot's own terser phrase is not what actually got rendered
    assert "A round woolly black sheep is also in the shot." not in prompt


def test_story_subject_without_subject_description_stays_byte_identical():
    """A song without the new field (every song before this feature, and any
    library entry the catalog marks as having no single recurring subject)
    must keep rendering the bare `story_subject` phrase unchanged."""
    prompt = _shot_prompt(_kofi_closeup_with_mouse(), _song(), _cfg())
    assert "A tiny round cartoon mouse is also in the shot." in prompt


def test_subject_description_none_is_treated_like_absent():
    song = _song()
    song["subject_description"] = None
    prompt = _shot_prompt(_kofi_closeup_with_mouse(), song, _cfg())
    assert "A tiny round cartoon mouse is also in the shot." in prompt


def test_subject_description_does_not_appear_twice():
    shot = _kofi_closeup_with_mouse()
    shot["story_subject"] = "a round woolly black sheep"
    prompt = _shot_prompt(shot, _song_with_subject_description(), _cfg())
    # The sentence capitalizes the description's leading letter, so compare
    # case-insensitively rather than expecting the verbatim (lowercase-first)
    # constant to appear.
    assert prompt.lower().count(_SHEEP_DESCRIPTION.lower()) == 1


def test_subject_only_shot_uses_the_canonical_description_too():
    shot = _mouse_only_shot()
    shot["story_subject"] = "a round woolly black sheep"
    prompt = _shot_prompt(shot, _song_with_subject_description(), _cfg())
    expected = f"{_SHEEP_DESCRIPTION[0].upper()}{_SHEEP_DESCRIPTION[1:]} is the subject of this medium shot"
    assert expected in prompt
    assert "A round woolly black sheep is the subject" not in prompt


def test_keyframe_prompt_children_plus_subject_uses_the_canonical_description():
    from pipeline.kidsong.generate import _keyframe_prompt

    shot = _kofi_closeup_with_mouse()
    shot["story_subject"] = "a round woolly black sheep"
    prompt = _keyframe_prompt(shot, _song_with_subject_description(), _cfg())
    expected = f"{_SHEEP_DESCRIPTION[0].upper()}{_SHEEP_DESCRIPTION[1:]} is also in the shot."
    assert expected in prompt


def test_keyframe_prompt_subject_only_uses_the_canonical_description_too():
    from pipeline.kidsong.generate import _keyframe_prompt

    shot = _mouse_only_shot()
    shot["story_subject"] = "a round woolly black sheep"
    prompt = _keyframe_prompt(shot, _song_with_subject_description(), _cfg())
    expected = f"{_SHEEP_DESCRIPTION[0].upper()}{_SHEEP_DESCRIPTION[1:]} is the subject of this medium shot"
    assert expected in prompt


def test_keyframe_prompt_story_subject_without_description_stays_byte_identical():
    from pipeline.kidsong.generate import _keyframe_prompt

    shot = _kofi_closeup_with_mouse()
    prompt = _keyframe_prompt(shot, _song(), _cfg())
    assert "A tiny round cartoon mouse is also in the shot." in prompt


# --------------------------------------------------------- per-shot NEGATIVE ---
_BASELINE_NEGATIVE = (
    "photorealistic, live action, watermark, logo, deformed, extra limbs, "
    "wrong shirt color, shirtless child, jewellery, earrings"
)


def test_shot_negative_adds_character_term_and_keeps_baseline():
    result = _shot_negative(_kofi_closeup(), _cfg(), _BASELINE_NEGATIVE)
    assert "yellow shirt" in result.lower()  # Kofi's must_not signature term
    assert "photorealistic" in result
    assert "watermark" in result
    assert "deformed" in result


def test_shot_negative_does_not_duplicate_terms_already_in_baseline():
    result = _shot_negative(_kofi_closeup(), _cfg(), _BASELINE_NEGATIVE)
    assert result.lower().count("shirtless child") == 1
    assert result.lower().count("jewellery") == 1


def test_shot_negative_degrades_to_plain_baseline_without_cast_bible(no_cast_module):
    result = _shot_negative(_kofi_closeup(), _cfg(), _BASELINE_NEGATIVE)
    assert result == _BASELINE_NEGATIVE


class _FakeClientWithNegative:
    def __init__(self, negative_text):
        self._negative_text = negative_text

    def load_workflow(self, workflow_name):
        return {
            "6": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": "a happy scene", "clip": ["4", 0]},
                "_meta": {"title": "PROMPT"},
            },
            "7": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": self._negative_text, "clip": ["4", 0]},
                "_meta": {"title": "NEGATIVE"},
            },
        }


class _BrokenClient:
    def load_workflow(self, workflow_name):
        raise FileNotFoundError(f"no such workflow: {workflow_name}")


def test_baseline_negative_reads_the_negative_node_text():
    client = _FakeClientWithNegative("photorealistic, watermark, wrong shirt color")
    assert _baseline_negative(client, "ltx23_t2v_toon") == "photorealistic, watermark, wrong shirt color"


def test_baseline_negative_degrades_to_empty_string_on_load_failure():
    assert _baseline_negative(_BrokenClient(), "missing") == ""


def test_baseline_negative_reads_the_real_workflow_file():
    """Confidence check against the actual repo workflow (local disk read only,
    no network/GPU): the hardened baseline another agent maintains must include
    the character-agnostic wardrobe term this feature is built to reinforce."""
    from pipeline.kidsong.comfy import ComfyClient

    client = ComfyClient({})
    baseline = _baseline_negative(client, "ltx23_t2v_toon")
    assert baseline
    assert "wrong shirt color" in baseline


# ------------------------------------------------------------------- seeds ---
def test_retry_seed_deterministic_across_calls():
    assert _retry_seed(["Kofi"], 20260717, 1) == _retry_seed(["Kofi"], 20260717, 1)


def test_retry_seed_differs_per_attempt_but_stays_in_the_character_family():
    """Old behaviour was `seed += 977`, which threw away the character keying.
    The new behaviour must vary per attempt while remaining anchored to the
    character's own seed_offset (cast.seed_for(names, base_seed))."""
    from pipeline.kidsong import cast

    base = 20260717
    family = cast.seed_for(["Kofi"], base)
    seed_attempt_1 = _retry_seed(["Kofi"], base, 1)
    seed_attempt_2 = _retry_seed(["Kofi"], base, 2)

    assert seed_attempt_1 != seed_attempt_2
    assert seed_attempt_1 - 977 * 1 == family
    assert seed_attempt_2 - 977 * 2 == family


def test_retry_seed_differs_per_character():
    assert _retry_seed(["Kofi"], 20260717, 1) != _retry_seed(["Zuri"], 20260717, 1)


def test_retry_seed_falls_back_without_cast_bible(no_cast_module):
    # Must not raise, and must still be deterministic + vary per attempt.
    a = _retry_seed(["Kofi"], 20260717, 1)
    b = _retry_seed(["Kofi"], 20260717, 2)
    assert a != b
    assert a == 20260717 + 977 * 1
    assert b == 20260717 + 977 * 2


# ------------------------------------------------------- director seed wiring ---
def test_seed_for_characters_matches_cast_seed_for():
    from pipeline.kidsong import cast

    assert _seed_for_characters(["Kofi"], 100) == cast.seed_for(["Kofi"], 100)
    assert _seed_for_characters(["Zuri"], 100) != _seed_for_characters(["Kofi"], 100)


def test_seed_for_characters_falls_back_to_char_hash_without_cast_bible(no_cast_module):
    assert _seed_for_characters(["Kofi"], 100) == 100 + _char_hash(["Kofi"])


def _plan_shots_offline(monkeypatch, song, verse_times, beats, cfg):
    """Call director.plan_shots with every network-touching LLM call forced to
    fail, so it always falls straight through to the deterministic fallback
    planner — never hits localhost:11434 or any other host."""
    from pipeline.kidsong import director

    def _boom(*a, **k):
        raise RuntimeError("network disabled in tests")

    monkeypatch.setattr(director, "_call_gemini", _boom)
    monkeypatch.setattr(director, "_call_groq", _boom)
    monkeypatch.setattr(director, "_call_ollama", _boom)
    monkeypatch.setattr(director, "_post_ollama_chat", _boom)
    monkeypatch.setattr(director, "_post_openai_compatible_chat", _boom)
    return director.plan_shots(song, verse_times, beats, cfg)


def test_director_planned_seeds_same_character_same_seed_different_character_differs(monkeypatch):
    from pipeline.config import load_config
    from pipeline.kidsong.lyrics import FALLBACK_SONG

    cfg = load_config()
    verse_times = [(0, 15), (15, 30), (30, 45), (45, 60)]
    beats = {"bpm": 96, "beat_times": [i * 0.625 for i in range(97)]}

    result = _plan_shots_offline(monkeypatch, FALLBACK_SONG, verse_times, beats, cfg)
    shots = result["shots"]

    by_single_character = {}
    for s in shots:
        chars = s.get("characters") or []
        if len(chars) == 1 and chars[0] != "all":
            by_single_character.setdefault(chars[0], set()).add(s["seed"])

    # Every shot naming the same single character shares exactly one seed.
    for name, seeds in by_single_character.items():
        assert len(seeds) == 1, f"{name} got multiple seeds across shots: {seeds}"

    # Different characters get different seeds.
    seed_by_name = {name: next(iter(seeds)) for name, seeds in by_single_character.items()}
    assert len(set(seed_by_name.values())) == len(seed_by_name)


def test_director_planned_seeds_deterministic_across_two_calls(monkeypatch):
    """Ports the `assert seeds1 == seeds2` selftest from director.py's
    `if __name__ == "__main__":` block (never runs under pytest) into a real test."""
    from pipeline.config import load_config
    from pipeline.kidsong.lyrics import FALLBACK_SONG

    cfg = load_config()
    verse_times = [(0, 15), (15, 30), (30, 45), (45, 60)]
    beats = {"bpm": 96, "beat_times": [i * 0.625 for i in range(97)]}

    result1 = _plan_shots_offline(monkeypatch, FALLBACK_SONG, verse_times, beats, cfg)
    result2 = _plan_shots_offline(monkeypatch, FALLBACK_SONG, verse_times, beats, cfg)

    seeds1 = [s["seed"] for s in result1["shots"]]
    seeds2 = [s["seed"] for s in result2["shots"]]
    assert seeds1 == seeds2


# --------------------------------------------------------- decreep (gaze) cases ---
# kidsong.decreep.enabled (default False): see tests/test_kidsong_gaze.py for the
# full matrix (the `_gaze_sentence` unit tests, `_fallback_planner`'s budgeted
# hook-verse camera designation, `_normalize_shots`'s LLM "gaze" overlay
# validation, and the `_PERFORMANCE_BEATS`/`_ACTION_VERBS` de-creep pools). These
# cases cover `_shot_prompt` itself, alongside the byte-identical-when-off
# baselines already pinned above and in tests/test_kidsong_render_style.py.
def _cfg_decreep():
    return {"kidsong": {"shot": {"style_trigger": "P1x4r"}, "decreep": {"enabled": True}}}


def test_decreep_on_closeup_drops_the_legacy_camera_sentence():
    prompt = _shot_prompt(_kofi_closeup(), _song(), _cfg_decreep())
    assert "face the camera as they move" not in prompt
    assert "The children have soft rounded shapes and big expressive eyes." in prompt


def test_decreep_on_wide_drops_the_legacy_camera_sentence():
    prompt = _shot_prompt(_all_wide(), _song(), _cfg_decreep())
    assert "face the camera as they move" not in prompt
    assert "The children have soft rounded shapes and big expressive eyes." in prompt


def test_decreep_on_with_story_subject_gazes_at_the_subject():
    prompt = _shot_prompt(_kofi_closeup_with_mouse(), _song(), _cfg_decreep())
    assert f"Their eyes are on {_MOUSE}." in prompt
    assert "face the camera as they move" not in prompt


def test_decreep_on_still_keeps_the_pinned_identity_and_count_sentences():
    """The gaze-sentence swap must not disturb any of the OTHER sentences this
    file already pins for the flag-off path."""
    prompt = _shot_prompt(_kofi_closeup(), _song(), _cfg_decreep())
    assert "exactly one child" in prompt.lower()
    assert "No other children are visible in frame." in prompt
    assert (
        "Every child on screen is one of these Black toddlers — no other children appear."
        in prompt
    )


def test_decreep_off_explicit_matches_the_omitted_default():
    """`{"enabled": False}` and no `decreep` key at all must render identically
    -- both fall back to the same pinned baseline."""
    off_explicit = {
        "kidsong": {"shot": {"style_trigger": "P1x4r"}, "decreep": {"enabled": False}},
    }
    off_absent = _cfg()
    assert (
        _shot_prompt(_kofi_closeup(), _song(), off_explicit)
        == _shot_prompt(_kofi_closeup(), _song(), off_absent)
    )


# ------------------------------------------------- style skeleton pin (4c) ---
def test_keyframe_prompt_starts_with_the_active_style_skeleton():
    """The keyframe still prompt must OPEN with the active render style's
    prompt_skeleton — that skeleton is the only style conditioning a keyframe
    gets (no LoRA/trigger on the still models), so losing it silently would
    reintroduce the within-video style mix."""
    from pipeline.kidsong.generate import _keyframe_prompt

    skeleton = "a testable skeleton sentence for style pinning."
    cfg = {"kidsong": {
        "render_style": "teststyle",
        "render_styles": {"teststyle": {"prompt_skeleton": skeleton,
                                        "style_trigger": "TSTYL"}},
    }}
    prompt = _keyframe_prompt(_kofi_closeup(), _song(), cfg)
    assert prompt.startswith(skeleton)
    # the LTX trigger token stays out of still prompts
    assert "TSTYL" not in prompt
