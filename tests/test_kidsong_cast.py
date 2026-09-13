"""Tests for kidsong.cast — the fixed ZubiBop cast bible.

CPU-only, no network, no GPU. Covers: bible loading/validation, case-insensitive
non-substring lookup, canonical descriptions, the cast_sentence closeup-crowding
regression (a single-character shot must not mention the other kids), negative
term unions, deterministic per-character seeding, and expected child counts.
"""
import json
import re

import pytest

from pipeline.kidsong import cast


# ------------------------------------------------------------------ loading ---
def test_bible_loads_and_validates():
    bible = cast.load_bible()
    assert isinstance(bible, dict)
    assert len(bible["characters"]) >= 2


def test_version_nonempty():
    assert cast.version()
    assert isinstance(cast.version(), str)


def test_every_character_has_a_top_garment():
    bible = cast.load_bible()
    for c in bible["characters"]:
        assert str(c.get("top") or "").strip(), f"{c.get('name')} has no top garment"


def test_no_jewellery_terms_anywhere_in_bible():
    bible = cast.load_bible()
    blob = json.dumps(bible).lower()
    assert not re.search(r"jewel|earring|piercing|pierced", blob), blob


# --------------------------------------------------------------- lookup ---
def test_character_lookup_is_case_insensitive():
    kofi = cast.character("kofi")
    assert kofi is not None
    assert kofi["name"] == "Kofi"
    assert cast.character("KOFI") is not None
    assert cast.character("KoFi") is not None


def test_character_lookup_does_not_substring_match():
    assert cast.character("Ama") is None  # must not resolve 'Amara'/'Nala' etc.
    assert cast.character("Amara") is None  # not a channel character at all
    assert cast.character("") is None
    assert cast.character(None) is None


# ------------------------------------------------------------------ describe ---
def test_describe_always_includes_skin_hair_and_top():
    for n in cast.names():
        c = cast.character(n)
        desc = cast.describe(n)
        assert c["skin"] in desc
        assert c["hair"] in desc
        assert c["top"] in desc
        assert not desc.endswith(".")


def test_describe_unknown_returns_empty_string():
    assert cast.describe("NotACharacter") == ""


def test_describe_includes_the_build_when_present():
    """A character's optional `build` (anatomy) fragment is woven into the
    description between age and skin, so distinct body sizes reach every prompt."""
    c = {"name": "Test", "age": "4-year-old boy",
         "build": "a taller, sturdier build", "skin": "deep brown skin",
         "hair": "short curls", "top": "a blue t-shirt"}
    desc = cast._describe_dict(c)
    assert "a taller, sturdier build" in desc
    assert desc.startswith("Test, a 4-year-old boy with a taller, sturdier build, deep brown skin, short curls")


def test_every_bible_build_is_a_noun_phrase():
    """`_describe_dict` splices `build` in as "<Name>, a <age> WITH <build>, ...",
    and `with` takes a noun phrase. The test above already encodes that contract
    with its "a taller, sturdier build" fixture — but nothing enforced it on the
    SHIPPED bible, and two of the three characters violated it:

        Nala, a 3-year-old girl with petite and delicate, a little shorter and
        slighter, deep brown skin, ...
        Kofi, a 4-year-old boy with noticeably tall and sturdy for a preschooler,
        with slightly broader shoulders, medium-deep brown skin, ...

    "with petite and delicate" is not English, and this is the identity sentence
    — the single most load-bearing sentence in the prompt, the one QM-001 and
    QM-015 are about, sent to a text encoder that SOURCES.md records as following
    natural film-style prose far better than fragments. Kofi's also nested a
    second `with` clause inside the first, so the attribute list parsed
    ambiguously.

    Two cheap, checkable properties separate the noun-phrase form from the
    adjective-phrase form: it opens with a determiner, and it does not smuggle in
    its own `with`.
    """
    for c in cast.load_bible()["characters"]:
        build = str(c.get("build") or "").strip()
        if not build:
            continue
        first = build.split()[0].lower().strip(",")
        assert first in ("a", "an", "the"), (
            f"{c['name']}'s build {build!r} starts with {first!r} — an adjective "
            f"phrase. It renders as \"a {c.get('age')} with {build}\"."
        )
        assert " with " not in build, (
            f"{c['name']}'s build {build!r} nests a second `with` inside the "
            "attribute list."
        )


def test_the_shipped_cast_descriptions_read_as_english():
    """The composed fragment for every real character, end to end: exactly one
    `with` introducing the attribute list, and no comma stranded before it."""
    for c in cast.load_bible()["characters"]:
        desc = cast.describe(c["name"])
        head = desc.split(", wearing ")[0]
        assert head.count(" with ") >= 1
        # the attribute list opens with a determiner or an adjective+noun, never
        # a bare predicate adjective
        attrs = head.split(" with ", 1)[1]
        assert not attrs.startswith(("petite ", "tall ", "noticeably ", "slim ")), desc


def test_describe_without_build_is_unchanged():
    """A character with no `build` field describes exactly as before — the
    byte-identical guarantee for an un-migrated bible."""
    c = {"name": "Test", "age": "3-year-old girl", "skin": "deep warm brown skin",
         "hair": "afro puffs", "top": "a yellow t-shirt"}
    desc = cast._describe_dict(c)
    assert desc == "Test, a 3-year-old girl with deep warm brown skin, afro puffs, wearing a yellow t-shirt"


# -------------------------------------------------------------- cast_sentence ---
def test_cast_sentence_single_character_excludes_the_others():
    sentence = cast.cast_sentence(["Kofi"])
    assert "Kofi" in sentence
    assert "Zuri" not in sentence
    assert "Nala" not in sentence


def test_cast_sentence_none_mentions_all_three():
    sentence = cast.cast_sentence(None)
    for n in cast.names():
        assert n in sentence


def test_cast_sentence_all_mentions_all_three():
    sentence = cast.cast_sentence(["all"])
    for n in cast.names():
        assert n in sentence


def test_cast_sentence_unknown_name_resolves_to_one_child_not_the_ensemble():
    # REGRESSION: an unresolvable name used to expand to the FULL ENSEMBLE, so a
    # shot naming one invented child described all three and rendered all three.
    sentence = cast.cast_sentence(["NotACharacter"])
    present = [n for n in cast.names() if n in sentence]
    assert len(present) == 1, sentence
    assert sentence.startswith("One adorable Black toddler:"), sentence


def test_cast_sentence_two_characters():
    sentence = cast.cast_sentence(["Kofi", "Nala"])
    assert "Kofi" in sentence
    assert "Nala" in sentence
    assert "Zuri" not in sentence
    assert sentence.startswith("Two adorable Black toddlers:")


# ------------------------------------------------------------- negative_terms ---
def test_negative_terms_kofi_includes_signature_and_floor_terms():
    terms = cast.negative_terms(["Kofi"])
    assert "yellow shirt" in terms
    for floor_term in ("shirtless child", "jewellery", "earrings", "extra children"):
        assert floor_term in terms


def test_negative_terms_are_deduped_and_order_stable():
    terms1 = cast.negative_terms(["Kofi"])
    terms2 = cast.negative_terms(["Kofi"])
    assert terms1 == terms2
    assert len(terms1) == len(set(t.lower() for t in terms1))


# ------------------------------------------------- negative_terms subtraction ---
# BUG 1: negative_terms() used to union every resolved character's must_not
# UNFILTERED, so a multi-character shot negated the very wardrobe its own
# positive prompt (cast_sentence) just requested — Zuri's must_not negates
# Kofi's blue shirt, Kofi's must_not negates Zuri's yellow shirt, so an
# ['all'] shot's negative contradicted the positive on both colors. The fix
# subtracts any must_not term already true of the resolved cast's own
# describe() text (case-insensitive, word-aware — not a naive substring
# check of the full term against the raw text), while leaving the
# channel-wide floor untouched.
def test_negative_terms_all_drops_worn_colors_but_keeps_unworn_ones():
    terms = cast.negative_terms(["all"])
    # Kofi wears blue, Zuri wears yellow -- both actually on screen, so
    # neither must appear in the ensemble's own negative prompt.
    assert "blue shirt" not in terms
    assert "yellow shirt" not in terms
    # Nobody wears purple or pink -- those stay negated.
    assert "purple dress" in terms
    assert "pink dress" in terms
    # No character in the bible has pale skin / straight / blonde / ginger
    # hair -- those must still be negated for the ensemble shot too.
    for still_negated in ("pale skin", "straight hair", "blonde hair", "ginger hair"):
        assert still_negated in terms, terms


def test_negative_terms_single_character_still_negates_the_others_colors():
    # A Kofi-only shot must still negate "yellow shirt" (that's Zuri's,
    # nobody on screen in this shot wears it) -- subtraction is scoped to
    # what's ACTUALLY resolved/on screen, not the full cast.
    kofi_terms = cast.negative_terms(["Kofi"])
    assert "yellow shirt" in kofi_terms

    zuri_terms = cast.negative_terms(["Zuri"])
    assert "blue shirt" in zuri_terms  # Kofi's color, not in this shot


def test_negative_terms_channel_floor_always_survives_subtraction():
    floor_terms = (
        "shirtless child", "bare chest", "undressed", "underwear",
        "extra children", "crowd of children", "duplicate characters",
        "jewellery", "earrings",
    )
    for names in (["all"], ["Kofi"], ["Zuri"], ["Nala"], ["Kofi", "Nala"]):
        terms = cast.negative_terms(names)
        for floor_term in floor_terms:
            assert floor_term in terms, f"{floor_term!r} missing for {names!r}: {terms}"


# --------------------------------- cross-character identity guard (subset) ---
# Measured live (2026-07-23 storyfix render): a solo Kofi closeup rendered with
# Zuri's afro puffs + denim dungarees. His must_not negated her yellow SHIRT but
# nothing negated her HAIR or BOTTOM, so a single-child slot drifted into an
# absent castmate's look. negative_terms now also negates the ABSENT cast
# members' distinctive hair/top/bottom on any subset (solo/pair) shot.
def test_negative_terms_solo_shot_negates_absent_members_hair_and_bottom():
    kofi = cast.negative_terms(["Kofi"], "closeup")
    blob = " | ".join(kofi).lower()
    # Zuri's signature hair + bottom (the exact drift observed) are negated…
    assert "afro puff" in blob, kofi
    assert "denim dungaree shorts" in blob, kofi
    # …and Nala's, too.
    assert "cornrow braids" in blob, kofi
    assert "white tights" in blob, kofi
    # But NONE of Kofi's own look is negated (that would fight the positive).
    for own in ("a blue t-shirt", "grey jogger shorts", "short dark natural curly hair"):
        assert own not in kofi, f"negated Kofi's own {own!r}: {kofi}"


def test_negative_terms_ensemble_does_not_add_descriptor_phrases():
    """['all'] has no absent members, so the cross-character guard adds nothing —
    the descriptor phrases (which are NOT in anyone's must_not) must not leak in,
    or the ensemble negative would contradict its own positive wardrobe."""
    terms = [t.lower() for t in cast.negative_terms(["all"], "wide")]
    for worn in ("denim dungaree shorts", "a blue t-shirt",
                 "a yellow pinafore dress over a white t-shirt", "white tights"):
        assert worn not in terms, f"{worn!r} leaked into the ensemble negative: {terms}"


def test_negative_terms_pair_shot_negates_only_the_absent_third():
    """A [Zuri, Kofi] shot negates Nala's look but not each other's (both are
    on screen, so their wardrobe must stay available to the positive)."""
    terms = " | ".join(cast.negative_terms(["Zuri", "Kofi"], "medium")).lower()
    assert "cornrow braids" in terms          # Nala absent -> negated
    assert "yellow pinafore" in terms         # Nala absent -> negated
    assert "denim dungaree shorts" not in terms   # Zuri present -> NOT negated
    assert "short dark natural curly hair" not in terms  # Kofi present -> NOT negated


def test_negative_terms_mutation_all_would_contradict_positive_without_the_fix():
    """Guards the actual regression: build the ensemble negative the OLD
    (unfiltered-union) way and confirm it really did contradict the positive
    prompt -- i.e. this test would have failed before the BUG 1 fix, proving
    it actually exercises the subtraction logic rather than trivially passing."""
    resolved = cast._resolve_with_fallback(["all"])
    unfiltered = []
    seen = set()
    for c in resolved:
        for t in c.get("must_not") or []:
            key = t.lower()
            if key not in seen:
                seen.add(key)
                unfiltered.append(t)
    assert "blue shirt" in unfiltered and "yellow shirt" in unfiltered  # the old bug

    fixed = cast.negative_terms(["all"])
    assert "blue shirt" not in fixed
    assert "yellow shirt" not in fixed


# ------------------------------------------------------------------- seed_for ---
def test_seed_for_single_character_is_base_plus_offset():
    kofi_offset = cast.character("Kofi")["seed_offset"]
    assert cast.seed_for(["Kofi"], 100) == 100 + kofi_offset


def test_seed_for_is_order_insensitive_for_multi_name():
    a = cast.seed_for(["Kofi", "Nala"], 20260717)
    b = cast.seed_for(["Nala", "Kofi"], 20260717)
    assert a == b


def test_seed_for_is_stable_across_calls():
    first = cast.seed_for(["all"], 20260717)
    second = cast.seed_for(["all"], 20260717)
    assert first == second


def test_seed_for_differs_per_character():
    zuri_seed = cast.seed_for(["Zuri"], 100)
    kofi_seed = cast.seed_for(["Kofi"], 100)
    nala_seed = cast.seed_for(["Nala"], 100)
    assert len({zuri_seed, kofi_seed, nala_seed}) == 3


# ------------------------------------------------------------ expected count ---
def test_expected_child_count_all():
    assert cast.expected_child_count(["all"]) == 3
    assert cast.expected_child_count(None) == 3


def test_expected_child_count_single():
    assert cast.expected_child_count(["Kofi"]) == 1


def test_expected_child_count_unknown_name_is_one_not_the_ensemble():
    # One invented name is still ONE child on screen. Both earlier regimes were
    # wrong: `_resolve_exact` said 1 while the sentence described 3 (a
    # self-contradictory prompt), then the ensemble fallback said 3 for a shot
    # that specified 1 (silently more people than the shot asked for).
    assert cast.expected_child_count(["NotACharacter"]) == 1


def test_expected_child_count_two_resolved():
    assert cast.expected_child_count(["Kofi", "Nala"]) == 2


# --------------------------------------------------- BUG 2: resolver agreement ---
# `cast_sentence` and `expected_child_count` used to use different resolvers:
# `_resolve_with_fallback` (cast_sentence) falls back to the full ensemble
# when nothing resolves, but `expected_child_count` used `_resolve_exact`
# (no fallback, floored at 1). `characters=["Amara"]` (unresolvable) then
# produced a sentence describing all three kids but a headcount of 1 --
# contradicting itself and hard-rejecting a correct render downstream.
def test_expected_child_count_agrees_with_cast_sentence_for_unresolvable_name():
    sentence = cast.cast_sentence(["Amara"])
    count = cast.expected_child_count(["Amara"])
    described = sum(1 for n in cast.names() if n in sentence)
    assert count == described == 1, (sentence, count)


def test_expected_child_count_agrees_with_cast_sentence_for_known_names():
    for names in (["Kofi"], ["Kofi", "Nala"], ["all"], None):
        sentence = cast.cast_sentence(names)
        count = cast.expected_child_count(names)
        resolved_names_in_sentence = sum(1 for n in cast.names() if n in sentence)
        assert count == resolved_names_in_sentence, (names, sentence, count)


def test_resolve_with_fallback_logs_a_warning_on_unresolvable_name(caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="kidsong.cast"):
        cast._resolve_with_fallback(["Amara"])
    assert any("Amara" in r.message for r in caplog.records)
    assert any("unresolvable" in r.message.lower() or "fall" in r.message.lower()
               for r in caplog.records)


# ------------------------------------------------------------------- names ---
def test_names_returns_canonical_display_names_in_bible_order():
    assert cast.names() == ["Zuri", "Kofi", "Nala"]


# -------------------------------------------------------------- malformed ---
_VALID_CHAR = {
    "id": "zuri",
    "name": "Zuri",
    "age": "3-year-old girl",
    "skin": "deep warm brown skin",
    "hair": "afro puffs",
    "top": "a yellow t-shirt",
    "bottom": "shorts",
    "shoes": "sneakers",
    "prop": "a toy",
    "seed_offset": 1013,
    "must_not": [],
}


def test_malformed_bible_missing_top_raises():
    other = dict(_VALID_CHAR, id="kofi", name="Kofi", seed_offset=2027, top="")
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        path = _write_bible_dir(td, [_VALID_CHAR, other])
        with pytest.raises(ValueError):
            cast.load_bible(path=path)


def test_malformed_bible_duplicate_names_raises():
    dupe = dict(_VALID_CHAR, id="zuri2", seed_offset=2027)  # same name "Zuri"
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        path = _write_bible_dir(td, [_VALID_CHAR, dupe])
        with pytest.raises(ValueError):
            cast.load_bible(path=path)


def test_malformed_bible_single_character_raises():
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        path = _write_bible_dir(td, [_VALID_CHAR])
        with pytest.raises(ValueError):
            cast.load_bible(path=path)


def test_malformed_bible_duplicate_seed_offset_raises():
    other = dict(_VALID_CHAR, id="kofi", name="Kofi")  # same seed_offset as Zuri
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        path = _write_bible_dir(td, [_VALID_CHAR, other])
        with pytest.raises(ValueError):
            cast.load_bible(path=path)


def test_malformed_bible_missing_version_raises():
    import tempfile

    other = dict(_VALID_CHAR, id="kofi", name="Kofi", seed_offset=2027)
    with tempfile.TemporaryDirectory() as td:
        path = _write_bible_dir(td, [_VALID_CHAR, other], version="")
        with pytest.raises(ValueError):
            cast.load_bible(path=path)


def _write_bible_dir(dirpath, characters, version="1.0.0"):
    import os

    path = os.path.join(dirpath, "bible.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "version": version,
                "channel": "ZubiBop",
                "ensemble_label": "Adorable Black toddlers",
                "characters": characters,
            },
            f,
        )
    return path


# ============================================================================
# Unresolvable-name fallback policy.
#
# The LLM routinely invents cast names ("Amira", "Luna"). Those names used to
# expand to the full ensemble, so a CLOSEUP OF ONE CHILD was prompted as
# "Exactly three children are on screen: ..." -- and the renderer obeyed:
# rainbow/s02 (closeup, characters=["Luna"]) came back with three children, the
# third off-model. The fallback must respect the SHOT's framing instead.
# ============================================================================


def test_unresolvable_name_in_a_closeup_yields_exactly_one_character():
    for names in (["Luna"], ["Amira"], ["NotACharacter"]):
        resolved = cast._resolve_with_fallback(names, shot_type="closeup")
        assert len(resolved) == 1, (names, resolved)


def test_unresolvable_name_in_a_closeup_head_count_matches_the_description():
    sentence = cast.cast_sentence(["Luna"], shot_type="closeup")
    count = cast.expected_child_count(["Luna"], shot_type="closeup")
    described = sum(1 for n in cast.names() if n in sentence)
    assert count == 1
    assert described == 1, sentence


def test_unresolvable_name_in_a_wide_shot_does_not_add_people():
    # s08: a wide shot specifying ONE character (invented "Amira") became
    # "Exactly three children are on screen". A wide shot may be wide without
    # being crowded -- the count must still be what the shot asked for.
    resolved = cast._resolve_with_fallback(["Amira"], shot_type="wide")
    assert len(resolved) == 1, resolved
    assert cast.expected_child_count(["Amira"], shot_type="wide") == 1


def test_unresolvable_names_never_exceed_the_number_requested():
    for shot_type in ("wide", "group", None, "medium", "closeup"):
        for names in (["A"], ["A", "B"], ["Kofi", "A"]):
            resolved = cast._resolve_with_fallback(names, shot_type=shot_type)
            assert 1 <= len(resolved) <= len(names), (shot_type, names, resolved)
            assert len(resolved) <= len(cast.names())


def test_group_shots_with_resolvable_names_are_unchanged():
    assert [c["name"] for c in cast._resolve_with_fallback(["all"], shot_type="wide")] == cast.names()
    assert [c["name"] for c in cast._resolve_with_fallback(None, shot_type="closeup")] == cast.names()
    assert cast.expected_child_count(["all"], shot_type="closeup") == len(cast.names())


def test_resolvable_names_are_unaffected_by_shot_type():
    for shot_type in (None, "closeup", "medium", "wide", "group"):
        assert [c["name"] for c in cast._resolve_with_fallback(["Kofi"], shot_type=shot_type)] == ["Kofi"]
        assert [c["name"] for c in cast._resolve_with_fallback(["Kofi", "Nala"], shot_type=shot_type)] == ["Kofi", "Nala"]
        assert cast.cast_sentence(["Kofi"], shot_type=shot_type) == cast.cast_sentence(["Kofi"])
        assert cast.negative_terms(["Kofi"], shot_type=shot_type) == cast.negative_terms(["Kofi"])
        assert cast.seed_for(["Kofi"], 1000, shot_type=shot_type) == cast.seed_for(["Kofi"], 1000)


def test_substitution_is_deterministic_within_a_process():
    first = cast.cast_sentence(["Zephyr"], shot_type="closeup")
    for _ in range(20):
        assert cast.cast_sentence(["Zephyr"], shot_type="closeup") == first


def test_substitution_is_deterministic_across_processes():
    # hash() is salted per process by PYTHONHASHSEED, so a hash()-based pick
    # would hand the same invented name a different child on every run and
    # break episode-to-episode consistency. Run real subprocesses with
    # DIFFERENT seeds and require identical answers.
    import json as _json
    import os as _os
    import subprocess
    import sys

    code = (
        "import json;from pipeline.kidsong import cast;"
        "print(json.dumps([[c['id'] for c in cast._resolve_with_fallback([n], shot_type=t)]"
        " for n in ['Amira','Luna','Zephyr','NotACharacter'] for t in ['closeup','wide']]))"
    )
    root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    outs = []
    for seed in ("0", "1", "12345"):
        env = dict(_os.environ, PYTHONHASHSEED=seed, PYTHONPATH=root)
        proc = subprocess.run(
            [sys.executable, "-c", code], cwd=root, env=env,
            capture_output=True, text=True,
        )
        assert proc.returncode == 0, proc.stderr
        outs.append(_json.loads(proc.stdout.strip().splitlines()[-1]))
    assert outs[0] == outs[1] == outs[2], outs


def test_recurring_invented_names_map_through_the_alias_table():
    # Observed in the two 20260719 runs: "Amira" appeared alongside Kofi+Zuri
    # (so she stands in for Nala) and "Luna" alongside Kofi+Nala (so she stands
    # in for Zuri). Pinning these keeps a whole episode internally consistent.
    assert cast.substitute_for("Amira")["id"] == "nala"
    assert cast.substitute_for("Luna")["id"] == "zuri"


def test_head_count_description_and_negatives_are_mutually_consistent():
    from pipeline.kidsong.generate import _COUNT_WORDS, _shot_negative, _shot_prompt

    cfg = {"kidsong": {"shot": {"style_trigger": "P1x4r"}}}
    song = {"characters": "Three adorable Black toddlers: Zuri has X. Kofi has Y. Nala has Z."}
    for shot_type in ("closeup", "medium", "wide", "group"):
        for names in (["Luna"], ["Amira"], ["Kofi"], ["Kofi", "Nala"], ["all"]):
            shot = {"id": "s01", "shot_type": shot_type, "characters": names,
                    "action": "sing", "setting": "a playroom", "camera": "static"}
            prompt = _shot_prompt(shot, song, cfg)
            negative = _shot_negative(shot, cfg, "")
            described = [n for n in cast.names() if n in prompt]

            # 1. the stated head count == the number of children described
            if "exactly one child" in prompt:
                stated = 1
            else:
                stated = next(
                    (k for k, w in _COUNT_WORDS.items()
                     if "Exactly %s child" % w in prompt),
                    None,
                )
            assert stated is not None, prompt
            assert stated == len(described), (shot_type, names, stated, described, prompt)

            # 2. the negatives must not negate what the prompt positively asks
            #    for, judged by the module's own phrase matcher (so Nala's
            #    "yellow pinafore dress over a white t-shirt" is correctly NOT
            #    treated as a "yellow shirt").
            positive = cast._normalize_for_match(
                " ".join(cast.describe(m) for m in described)
            )
            for term in negative.split(", "):
                t = term.strip()
                if t and t not in cast._CHANNEL_FLOOR:
                    assert not cast._term_satisfied(t, positive), (
                        "negative %r contradicts the prompt's own cast: %s"
                        % (t, described)
                    )


def test_prompt_never_says_one_children_or_three_child():
    # The legacy arm emitted "Exactly one children are on screen:" followed by
    # three described people -- a count contradicting both its own grammar and
    # its own cast list. That class of string must not come back.
    from pipeline.kidsong.generate import _shot_prompt

    cfg = {"kidsong": {"shot": {"style_trigger": "P1x4r"}}}
    songs = [
        {"characters": "Three adorable Black toddlers: Zuri has X. Maya has Y. Kofi has Z."},
        {"characters": ""},
        {},
    ]
    bad = ("one children", "two child ", "three child ")
    for song in songs:
        for shot_type in ("closeup", "medium", "wide", "group", "insert"):
            for names in (["Amira"], ["Luna"], ["Zuri"], ["all"], None, []):
                shot = {"id": "s01", "shot_type": shot_type, "characters": names,
                        "action": "sing", "setting": "a playroom", "camera": "static"}
                low = _shot_prompt(shot, song, cfg).lower()
                for phrase in bad:
                    assert phrase not in low, (phrase, low)


def test_legacy_arm_without_the_bible_never_states_an_unbacked_head_count(monkeypatch):
    # With the cast bible unavailable the legacy free-text parser may end up
    # describing the WHOLE blob. It must then state no head count at all rather
    # than claim len(names) -- that mismatch is what produced
    # "Exactly one children are on screen: <three people>".
    import sys as _sys

    import pipeline.kidsong as kidsong_pkg
    from pipeline.kidsong.generate import _shot_prompt

    # Same two-part disable as tests/test_kidsong_shot_prompt.py's
    # `no_cast_module` fixture: the sys.modules entry AND the already-cached
    # package attribute, or `from pipeline.kidsong import cast` short-circuits.
    monkeypatch.delattr(kidsong_pkg, "cast", raising=False)
    monkeypatch.setitem(_sys.modules, "pipeline.kidsong.cast", None)

    cfg = {"kidsong": {"shot": {"style_trigger": "P1x4r"}}}
    song = {"characters": "Three adorable Black toddlers: Zuri has X. Maya has Y. Kofi has Z."}
    shot = {"id": "s08", "shot_type": "wide", "characters": ["Amira"],
            "action": "brush teeth", "setting": "a bathroom", "camera": "static"}
    prompt = _shot_prompt(shot, song, cfg)
    assert "Exactly" not in prompt, prompt
    assert "exactly one child" not in prompt.lower(), prompt
    assert "Zuri" in prompt


def test_unresolved_names_is_exposed_for_the_qc_gate():
    assert cast.unresolved_names(["Kofi", "Amira"]) == ["Amira"]
    assert cast.unresolved_names(["Kofi", "Nala"]) == []
    assert cast.unresolved_names(["all"]) == []
    assert cast.unresolved_names(None) == []


# ------------------------------------------------- story casting helpers ---
def test_protagonist_is_deterministic_for_a_title():
    a = cast.protagonist_for("Rain, Rain, Go Away - A Sunny Sing-Along")
    b = cast.protagonist_for("Rain, Rain, Go Away - A Sunny Sing-Along")
    assert a and a == b
    ids = {c["id"] for c in cast.load_bible()["characters"]}
    assert a in ids


def test_protagonist_uses_md5_not_process_hash():
    """Pin the exact md5 formula so a refactor to Python's salted hash()
    (different every process) is caught immediately."""
    import hashlib

    title = "A Song About Something Wonderful"
    chars = cast.load_bible()["characters"]
    digest = int(hashlib.md5(title.strip().lower().encode("utf-8")).hexdigest(), 16)
    expected = chars[digest % len(chars)]["id"]
    assert cast.protagonist_for(title) == expected


def test_protagonist_override_honored_and_unknown_falls_back():
    assert cast.protagonist_for("t", {"kidsong": {"protagonist": "nala"}}) == "nala"
    # unknown override: warn + fall back to the title-derived pick, never raise
    fallback = cast.protagonist_for("t", {"kidsong": {"protagonist": "not-a-kid"}})
    assert fallback == cast.protagonist_for("t")


def test_partner_is_next_in_bible_order_with_wraparound():
    chars = [c["id"] for c in cast.load_bible()["characters"]]
    for i, cid in enumerate(chars):
        assert cast.partner_for(cid) == chars[(i + 1) % len(chars)]


def test_partner_override_honored_but_never_the_protagonist():
    assert cast.partner_for("zuri", {"kidsong": {"partner": "nala"}}) == "nala"
    # partner == protagonist is invalid -> falls back to next-in-order
    assert cast.partner_for("zuri", {"kidsong": {"partner": "zuri"}}) == cast.partner_for("zuri")


def test_story_casting_helpers_never_raise_on_garbage():
    assert isinstance(cast.protagonist_for(None), str)
    assert isinstance(cast.protagonist_for("", {}), str)
    assert isinstance(cast.partner_for("unknown-id"), str)
    assert isinstance(cast.partner_for(None, {"kidsong": "not-a-dict"}), str)
