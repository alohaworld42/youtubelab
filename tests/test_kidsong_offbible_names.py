"""An invented child name must never survive into a render prompt.

The defect, measured against every shipped shotlist in `output/`: the director
already scrubs off-bible names out of a shot's `characters` list, but the LLM
writes the same name into the free-text `action`, and `generate._shot_prompt`
renders `action` verbatim. Shipped prompts therefore read:

    "Exactly three children are on screen: Zuri, Kofi and Nala. …
     A wide shot: Amira and Kofi standing in front of the sink …"

Four names against a head count of three, and "Amira" carries no description —
so the text encoder invents a fourth child from the name alone, which is
exactly the invented-extra-child failure the head-count clamp, the negative
terms and the keyframe pass all exist to prevent. Every LTX prompting guide
this repo tracks (docs/quality/SOURCES.md) says the same thing: a prompt must
not contradict itself.

Two independent repairs are pinned here:
  * director.repair_offbible_names — runs on the LLM's action overlay, before
    any of the name-aware passes that only match bible names.
  * generate._rename_offbible_children — runs on a shot whose `characters`
    still carries an off-bible name (a hand-written or legacy shot list that
    never went through the director).

CPU-only, no network.
"""
import re

import pytest

from pipeline.kidsong import cast, director
from pipeline.kidsong import generate as G

ALLOWED = set(cast.names()) | {"all"}
BIBLE = set(cast.names())


# ------------------------------------------------------- director repair ---
def test_invented_name_in_an_action_becomes_a_bible_child():
    out = director.repair_offbible_names(
        "Amira and Kofi standing in front of the bathroom sink", ALLOWED
    )
    assert "Amira" not in out
    assert "Kofi" in out
    # Whatever it became must be a real, described character.
    replaced = out.split(" and ")[0]
    assert replaced in BIBLE


def test_the_substitution_is_deterministic():
    """Same text twice — the same bible child. `substitute_for` is alias-table
    first, else a stable hash, so a rerun never reshuffles who "Amira" is."""
    text = "Amira claps her hands"
    assert (director.repair_offbible_names(text, ALLOWED)
            == director.repair_offbible_names(text, ALLOWED))


def test_two_invented_names_never_collapse_onto_one_child():
    """"Maya clapping for Jaden" must not become "Nala clapping for Nala" —
    that is worse prose than the contradiction it replaced."""
    out = director.repair_offbible_names("Maya clapping for Jaden", ALLOWED)
    left, right = out.split(" clapping for ")
    assert left in BIBLE and right in BIBLE
    assert left != right


def test_an_invented_name_never_becomes_a_child_already_in_the_sentence():
    out = director.repair_offbible_names("Zuri waves at Amira", ALLOWED)
    assert out.count("Zuri") == 1
    assert out.split("at ")[1] in BIBLE - {"Zuri"}


def test_the_same_invented_name_twice_in_one_action_stays_one_child():
    out = director.repair_offbible_names("Amira waves while Amira jumps", ALLOWED)
    names = re.findall(r"\b(%s)\b" % "|".join(BIBLE), out)
    assert len(set(names)) == 1 and len(names) == 2


def test_a_bible_name_is_left_alone():
    text = "Zuri and Kofi and Nala clap together"
    assert director.repair_offbible_names(text, ALLOWED) == text


@pytest.mark.parametrize("text", [
    "the three Black toddlers clap together",
    "Kofi waves at his Friends in the playroom",
    "a sunny playroom with colorful toys",
    "The children sing along with happy faces",
])
def test_ordinary_prose_is_never_rewritten(text):
    """The only capitalised non-bible tokens in the whole shipped corpus were
    five invented names plus "Black" and "Friends" — a false rewrite here would
    silently change what the shot is about."""
    assert director.repair_offbible_names(text, ALLOWED) == text


def test_the_story_subject_is_not_mistaken_for_a_child():
    text = "Benny the mouse scampers down the clock case"
    assert director.repair_offbible_names(
        text, ALLOWED, story_subject="Benny the tiny round cartoon mouse"
    ) == text


def test_an_unreadable_cast_bible_leaves_the_text_untouched(monkeypatch):
    monkeypatch.setattr(cast, "substitute_for",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no bible")))
    text = "Amira and Kofi wave"
    assert director.repair_offbible_names(text, ALLOWED) == text


# ------------------------------------------------- prompt-level guarantee ---
def _prompt_for(shot):
    song = {"title": "Test", "characters": "Zuri, Kofi and Nala", "verses": []}
    return G._shot_prompt(shot, song, {})


def test_a_shot_list_that_never_saw_the_director_is_still_repaired():
    """`characters` carrying an off-bible name resolves to a bible child for
    the identity sentence; the action has to follow it."""
    shot = {
        "id": "s00", "shot_type": "wide", "camera": "static",
        "characters": ["Amira", "Kofi"],
        "action": "Amira and Kofi standing in front of the sink",
        "setting": "a bright cheerful bathroom",
    }
    prompt = _prompt_for(shot)
    assert not re.search(r"\bAmira\b", prompt, re.I)


def test_the_head_count_and_the_named_children_agree():
    """The whole point: every child the prompt NAMES must be one the identity
    sentence described."""
    shot = {
        "id": "s00", "shot_type": "wide", "camera": "static",
        "characters": ["Amira", "Kiara", "Kofi"],
        "action": "Amira and Kiara clapping while Kofi jumps",
        "setting": "a bright cheerful playroom",
    }
    prompt = _prompt_for(shot)
    named = {w for w in re.findall(r"\b[A-Z][a-z]{2,11}\b", prompt) if w in BIBLE}
    described = set(re.findall(r"\b(%s)\b" % "|".join(BIBLE),
                               prompt.split("A wide shot")[0]))
    assert named <= described, f"prompt names {named - described} without describing them"


def test_the_director_repair_survives_a_full_normalize_pass():
    """End to end through `_normalize_shots`, which is where an LLM overlay
    action actually enters."""
    from pipeline.kidsong.lyrics import FALLBACK_SONG

    song = dict(FALLBACK_SONG)
    verse_times = [(0, 15), (15, 30), (30, 45), (45, 60)]
    beats = {"bpm": 96, "beat_times": [i * 0.625 for i in range(97)]}
    raw = {"shots": [
        {"verse": 0, "shot_type": "wide", "camera": "static",
         "characters": ["Amira", "Kofi"],
         "action": "Amira and Kofi standing in front of the sink"},
    ]}
    shots = director._normalize_shots(raw, song, verse_times, beats, {})
    for s in shots:
        assert not re.search(r"\bAmira\b", str(s.get("action") or ""), re.I)
        for name in (s.get("characters") or []):
            assert name == "all" or name in BIBLE


@pytest.mark.parametrize("name", ["Ahmed", "Jared", "Fred", "Mohamed"])
def test_a_real_name_ending_in_ed_is_still_repaired(name):
    """The detector used to refuse any capitalised -ed token as "a past
    participle, not a name" — which also refused these. The shipped corpus
    contains no -ed opener at all, so the rule cost real coverage to catch
    nothing; the handful of participles that could open an action phrase are
    enumerated in `_NON_NAME_CAPS` instead."""
    out = director.repair_offbible_names(f"{name} waves at Kofi", ALLOWED)
    assert name not in out
    assert out.split(" waves")[0] in BIBLE


@pytest.mark.parametrize("participle", ["Dressed", "Seated", "Gathered", "Surrounded"])
def test_a_participle_opening_an_action_is_not_a_name(participle):
    text = f"{participle} around the low table with Kofi"
    assert director.repair_offbible_names(text, ALLOWED) == text


@pytest.mark.parametrize("gerund", ["Watering", "Smiling", "Brushing", "Skipping"])
def test_a_gerund_opening_an_action_is_not_a_name(gerund):
    """Every sentence-opening non-name in the shipped corpus was one of these."""
    text = f"{gerund} the sunflowers with Zuri"
    assert director.repair_offbible_names(text, ALLOWED) == text
