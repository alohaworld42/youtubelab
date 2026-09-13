"""The lighting/set tail is a style knob, so prompt length is a config flip.

Measured across the shipped corpus (docs/quality/SOURCES.md): composed t2v
prompts run min 166 / median 184 / p90 231 / max 270 words, 12% of them over
the ~200-word cap the RunDiffusion guide recommends. The words are dominated by
the three full wardrobe descriptions in the identity sentence — the part that is
most load-bearing and least safe to cut — while the fixed 48-word lighting/set
tail is what gets crowded out. Shortening the tail is therefore the candidate
experiment, and it was hardcoded in `generate._shot_prompt`, so running it meant
editing code and re-pinning every prompt baseline.

It is now `render_style`'s `prompt_tail`, with `pixar_toon`'s value the verbatim
sentence pair the old code emitted. What this file protects:

  1. the default path is byte-identical (the regression guarantee the whole
     registry rests on — also pinned in tests/test_kidsong_shot_prompt.py and
     tests/test_kidsong_render_style.py);
  2. a style that shortens the tail changes ONLY the tail;
  3. an empty/absent/malformed tail degrades cleanly instead of leaving a
     dangling space or crashing a queued render;
  4. the shipped `pixar_toon_concise` entry is a true A/B arm — same graph,
     LoRA, trigger, skeleton and identity clause, shorter tail.

CPU-only, no GPU, no network.
"""
import pytest

from pipeline.kidsong.generate import _shot_prompt
from pipeline.kidsong.render_style import _PIXAR_TOON_DEFAULTS, resolve_style

_DEFAULT_TAIL = (
    "A warm key light from one side with a soft cool rim light along hair and "
    "shoulders, gentle soft-edged shadows. The set is lovingly dressed with two "
    "or three simple props that fit the location, kept clear around the "
    "children — bold happy colors, a joyful everyday toddler moment."
)


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


def _cfg(tail=None):
    """A registry whose active style differs from pixar_toon ONLY in the tail."""
    entry = {} if tail is None else {"prompt_tail": tail}
    return {"kidsong": {"render_style": "experiment",
                        "render_styles": {"experiment": entry}}}


_SHOTS = pytest.mark.parametrize("shot", [_kofi_closeup(), _all_wide()],
                                 ids=["closeup", "wide"])


# ------------------------------------------------- 1. the default is unchanged --
def test_the_default_tail_is_the_sentence_pair_the_old_code_hardcoded():
    assert _PIXAR_TOON_DEFAULTS["prompt_tail"] == _DEFAULT_TAIL


@_SHOTS
def test_every_child_prompt_still_ends_with_the_default_tail(shot):
    assert _shot_prompt(shot, _song(), {}).endswith(" " + _DEFAULT_TAIL)


@_SHOTS
def test_an_entry_that_omits_the_tail_inherits_the_default(shot):
    """A partial style entry must not silently lose the lighting/set direction."""
    assert _shot_prompt(shot, _song(), _cfg()) == _shot_prompt(shot, _song(), {})


# --------------------------------------- 2. a shorter tail changes only the tail --
_CONCISE = (
    "A warm key light with a soft cool rim light, gentle soft-edged shadows. "
    "Two or three simple props fit the location, kept clear around the "
    "children — bold happy colors."
)


@_SHOTS
def test_a_shorter_tail_replaces_the_default_and_nothing_else(shot):
    default = _shot_prompt(shot, _song(), {})
    concise = _shot_prompt(shot, _song(), _cfg(_CONCISE))

    assert concise.endswith(" " + _CONCISE)
    assert _DEFAULT_TAIL not in concise
    # everything before the tail is byte-identical
    assert concise[:-len(_CONCISE)] == default[:-len(_DEFAULT_TAIL)]


@_SHOTS
def test_the_shorter_tail_actually_buys_back_words(shot):
    """The whole point of the knob: fewer words reach the text encoder."""
    default_words = len(_shot_prompt(shot, _song(), {}).split())
    concise_words = len(_shot_prompt(shot, _song(), _cfg(_CONCISE)).split())
    assert default_words - concise_words == 19


# ----------------------------------------------- 3. degenerate values degrade --
@_SHOTS
def test_an_empty_tail_leaves_no_dangling_space(shot):
    """"" is a real value (not "unset"), and the join must not leave the prompt
    ending in whitespace or carrying a double space where the tail used to be."""
    prompt = _shot_prompt(shot, _song(), _cfg(""))
    assert prompt == prompt.strip()
    assert "  " not in prompt
    assert prompt.endswith("as they move.")
    assert _DEFAULT_TAIL not in prompt


@pytest.mark.parametrize("bad", [False, 0, [], {}, 12, ["a", "b"], object()],
                         ids=["false", "zero", "list", "dict", "int", "words", "obj"])
def test_a_malformed_tail_never_crashes_a_queued_render(bad):
    """config is operator-edited by hand; a wrong type here must not take down
    a render that has already reached the ComfyUI queue."""
    prompt = _shot_prompt(_kofi_closeup(), _song(), _cfg(bad))
    assert prompt.startswith(_PIXAR_TOON_DEFAULTS["style_trigger"])
    assert _PIXAR_TOON_DEFAULTS["identity_clause"] in prompt


# ------------------------------------ the branches that carry their own tail ---
def test_insert_and_subject_only_shots_are_untouched_by_the_knob():
    """Both early-return branches compose their own (already short, no-children)
    closing sentence. They are deliberately NOT driven by `prompt_tail` — their
    text names the object/subject rather than the children — so a style that
    empties the tail must leave them exactly as they are."""
    insert = {"id": "s02", "shot_type": "insert", "characters": [],
              "action": "a colorful toothbrush", "setting": "a bathroom shelf"}
    subject = {"id": "s03", "shot_type": "medium", "characters": [],
               "story_subject": "the little mouse", "camera": "static",
               "action": "climbs the tall clock", "setting": "a cozy hallway"}

    for shot in (insert, subject):
        assert _shot_prompt(shot, _song(), _cfg("")) == _shot_prompt(shot, _song(), {})


# ------------------------------------------- 4. the shipped A/B arm is valid ---
def test_the_shipped_concise_style_is_a_pure_tail_variant():
    """`pixar_toon_concise` exists so the experiment is a one-line config flip.
    It must differ from pixar_toon in the tail and in NOTHING else — otherwise
    the A/B measures two changes at once and cannot attribute the result."""
    from pipeline.config import load_config

    cfg = load_config()
    styles = cfg["kidsong"]["render_styles"]
    assert "pixar_toon_concise" in styles

    cfg["kidsong"]["render_style"] = "pixar_toon_concise"
    concise = resolve_style(cfg)
    cfg["kidsong"]["render_style"] = "pixar_toon"
    base = resolve_style(cfg)

    differing = [k for k in _PIXAR_TOON_DEFAULTS if concise[k] != base[k]]
    assert differing == ["prompt_tail"]
    assert len(concise["prompt_tail"].split()) < len(base["prompt_tail"].split())


def test_the_shipped_concise_tail_keeps_the_load_bearing_direction():
    """Shorter, but it must still say what the light does and that the set is
    dressed but clear — the two things the tail is there for. Guards against a
    future edit that trims it into a no-op and reports the A/B as "no effect"."""
    from pipeline.config import load_config

    cfg = load_config()
    cfg["kidsong"]["render_style"] = "pixar_toon_concise"
    tail = resolve_style(cfg)["prompt_tail"].lower()

    assert "key light" in tail and "rim light" in tail and "shadows" in tail
    assert "props" in tail and "kept clear around the children" in tail


def test_no_shipped_tail_names_a_building_material():
    """"concrete props" once sat in this sentence meaning "specific, tangible".
    The Gemma-3 encoder read it as props MADE OF CONCRETE and grey stone blocks
    appeared as set dressing in every shot of every episode. Any future tail
    must stay free of words with a material reading."""
    from pipeline.config import load_config

    styles = load_config()["kidsong"]["render_styles"]
    banned = ("concrete", "cement", "steel", "marble", "plastic", "granite")
    for name, entry in styles.items():
        tail = str(entry.get("prompt_tail") or "").lower()
        for word in banned:
            assert word not in tail, f"{name}'s prompt_tail says {word!r}"


# --------------------------------- the leanest arm that cuts no cast info ----
def test_the_lean_arm_removes_only_the_duplicated_assertion():
    """`pixar_toon_lean` is QM-030 lever (a): every prompt already carries a
    count clamp ("No other children appear anywhere in the frame, foreground or
    background."), so the identity clause's trailing "— no other children
    appear" states the same constraint a second time. The clause's UNIQUE
    content — the identity constraint — must survive verbatim, or this stops
    being a redundancy cut and becomes a cast-conditioning change.
    """
    from pipeline.config import load_config

    cfg = load_config()
    styles = cfg["kidsong"]["render_styles"]
    assert "pixar_toon_lean" in styles

    cfg["kidsong"]["render_style"] = "pixar_toon_lean"
    lean = resolve_style(cfg)
    base_clause = _PIXAR_TOON_DEFAULTS["identity_clause"]

    # the identity half is kept word for word
    assert lean["identity_clause"].startswith(
        "Every child on screen is one of these Black toddlers"
    )
    # only the duplicated half is gone
    assert "no other children appear" not in lean["identity_clause"].lower()
    assert len(base_clause.split()) - len(lean["identity_clause"].split()) == 5

    # and it is still a config-only variant: nothing outside the two text keys
    cfg["kidsong"]["render_style"] = "pixar_toon"
    base = resolve_style(cfg)
    differing = sorted(k for k in _PIXAR_TOON_DEFAULTS if lean[k] != base[k])
    assert differing == ["identity_clause", "prompt_tail"]


def test_the_lean_arm_still_states_both_constraints():
    """Dropping the duplicate must not drop the CONSTRAINT: the composed prompt
    still has to say both "these children and no others" and "no additional
    children in frame" — the second now carried solely by the count clamp."""
    from pipeline.config import load_config

    cfg = load_config()
    cfg = {"kidsong": {"render_style": "pixar_toon_lean",
                       "render_styles": cfg["kidsong"]["render_styles"]}}
    prompt = _shot_prompt(_all_wide(), _song(), cfg)

    assert "Every child on screen is one of these Black toddlers." in prompt
    assert "No other children appear anywhere in the frame" in prompt
    # no descriptor was lost along the way
    for descriptor in ("deep warm brown skin", "short dark natural curly hair",
                       "neat cornrow braids"):
        assert descriptor in prompt


def test_no_lossless_trim_can_bring_a_three_child_prompt_under_the_cap():
    """The measured upper bound, and the reason QM-030 stops here rather than
    proposing a third text cut. `pixar_toon_lean` is the most that can come out
    without touching a cast descriptor: 19 words of tail plus the 5-word
    duplicate. A three-child prompt is still ~40 words over. Everything left is
    wardrobe/skin/hair — the measured mitigation for QM-001/QM-015."""
    from pipeline.config import load_config

    registry = load_config()["kidsong"]["render_styles"]

    def words(style_name):
        cfg = {"kidsong": {"render_style": style_name, "render_styles": registry}}
        return len(_shot_prompt(_all_wide(), _song(), cfg).split())

    saved = words("pixar_toon") - words("pixar_toon_lean")
    assert saved == 24, f"the lossless budget moved: {saved} words"
    assert words("pixar_toon_lean") > 200, (
        "a three-child prompt now fits under the cap without cutting cast "
        "description — re-measure and rewrite QM-030, this test is the premise"
    )
