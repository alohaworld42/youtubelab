"""hyperframes + brainrot as full VIDEO_TYPES citizens.

Covers: registry shape, no-foreign-brand hygiene, kidsong-safety-gate
exclusion, idea-prompt dispatch, and the video_type("brainrot") vs.
style("brainrot") non-confusion the two coexisting "brainrot" names invite.
"""
import json
import os

import pytest

from studio import ideas as ideas_mod
from studio import models
from studio.ideas import (
    BRAINROT_IDEA_PROMPT,
    BRAINROT_IDEA_SYSTEM,
    HYPERFRAMES_IDEA_PROMPT,
    HYPERFRAMES_IDEA_SYSTEM,
    KIDSONG_IDEA_PROMPT,
    KIDSONG_IDEA_SYSTEM,
    SOLO_IDEA_PROMPTS,
    is_kids_style,
    is_kidsong_type,
)
from studio.scheduler import _made_for_kids
from studio.views import STYLES, VIDEO_TYPE_IDS, VIDEO_TYPES

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CFG = {"llm": {"backend": "groq", "groq_model": "x"}}

# A small, deliberately non-exhaustive sample of protected third-party names;
# blurbs/names/prompts must describe OUR look, never point at one of these.
FOREIGN_BRANDS = ("cocomelon", "disney", "pixar", "peppa", "bluey", "mrbeast")


def _mentions_a_brand(text):
    lowered = (text or "").lower()
    return [b for b in FOREIGN_BRANDS if b in lowered]


# --------------------------------------------------------------- registry ----
def test_new_ids_are_registered():
    assert "hyperframes" in VIDEO_TYPE_IDS
    assert "brainrot" in VIDEO_TYPE_IDS


def test_every_video_type_has_complete_non_empty_fields():
    required = ("id", "name", "emoji", "blurb")
    for entry in VIDEO_TYPES:
        for field in required:
            assert field in entry, f"{entry} missing {field!r}"
            value = entry[field]
            assert isinstance(value, str) and value.strip(), (
                f"{entry.get('id')!r}.{field} is empty"
            )
    assert VIDEO_TYPE_IDS == {t["id"] for t in VIDEO_TYPES}


def test_hyperframes_and_brainrot_entries_look_right():
    by_id = {t["id"]: t for t in VIDEO_TYPES}

    hf = by_id["hyperframes"]
    assert hf["name"] == "Hyperframes Erklärvideo"
    assert hf["emoji"] == "🧩"

    br = by_id["brainrot"]
    assert br["name"] == "Brainrot Short"
    assert br["emoji"] == "🌀"


# ------------------------------------------------------- no foreign brands ----
def test_no_video_type_name_or_blurb_names_a_foreign_brand():
    for entry in VIDEO_TYPES:
        for field in ("name", "blurb"):
            hits = _mentions_a_brand(entry[field])
            assert not hits, f"{entry['id']}.{field} mentions {hits}: {entry[field]!r}"


def test_hyperframes_prompt_file_names_no_foreign_brand():
    path = os.path.join(_REPO_ROOT, "prompts", "hyperframes.txt")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    assert content.strip()
    hits = _mentions_a_brand(content)
    assert not hits, f"prompts/hyperframes.txt mentions {hits}"


@pytest.mark.parametrize("text", [
    HYPERFRAMES_IDEA_SYSTEM, HYPERFRAMES_IDEA_PROMPT,
    BRAINROT_IDEA_SYSTEM, BRAINROT_IDEA_PROMPT,
])
def test_new_idea_prompts_name_no_foreign_brand(text):
    hits = _mentions_a_brand(text)
    assert not hits, f"idea prompt mentions {hits}: {text!r}"


# -------------------------------------------------- kidsong safety exclusion --
def test_new_types_are_not_kidsong_types():
    assert is_kidsong_type("hyperframes") is False
    assert is_kidsong_type("brainrot") is False


# ---------------------------------------------------- idea-prompt dispatch ----
def test_solo_idea_prompts_dispatch_has_entries_for_both_new_types():
    assert "hyperframes" in SOLO_IDEA_PROMPTS
    assert "brainrot" in SOLO_IDEA_PROMPTS
    for video_type in ("hyperframes", "brainrot"):
        system, template = SOLO_IDEA_PROMPTS[video_type]
        assert system.strip()
        assert template.strip()
        # distinct from the kidsong prompt pair — a preschool sing-along
        # system/prompt must never leak into these two dispatch branches
        assert system != KIDSONG_IDEA_SYSTEM
        assert template != KIDSONG_IDEA_PROMPT


@pytest.mark.parametrize("video_type,marker", [
    ("hyperframes", "explainer"),
    ("brainrot", "meme"),
])
def test_generate_ideas_uses_the_dedicated_prompt_for_new_types(
    tmp_db, monkeypatch, video_type, marker
):
    """End-to-end through the same dispatch generate_ideas() uses for kidsong,
    proving the wiring (not just the constants) is hooked up."""
    import pipeline.script_gen as sg

    captured = {}

    def fake_complete(user, cfg, system=None):
        captured["user"] = user
        captured["system"] = system
        return json.dumps({"ideas": [{"topic": f"a {video_type} idea"}]})

    monkeypatch.setattr(sg, "complete", fake_complete)

    cid = models.create_channel("Test")
    models.update_channel(cid, default_video_type=video_type)
    ch = models.get_channel(cid)

    inserted = ideas_mod.generate_ideas(ch, CFG, k=3, video_type=video_type)

    assert inserted == 1
    assert captured["user"].strip()
    assert captured["system"].strip()
    assert marker in captured["user"].lower() or marker in captured["system"].lower()
    # must not be the kidsong (preschool) voice
    assert "preschool" not in captured["system"].lower()
    assert "sing-along" not in captured["user"].lower()
    # the idea is actually filed under the requested type
    idea = models.next_pending_idea(cid)
    assert idea["video_type"] == video_type


# ------------------- video_type("brainrot") vs. style("brainrot") ------------
def test_brainrot_style_and_brainrot_video_type_coexist_as_distinct_things():
    """Documented, intentional: STYLES has "brainrot" (a look-and-sound preset)
    and VIDEO_TYPES now also has "brainrot" (a content format). They live in
    different columns and must never be read as the same signal."""
    assert "brainrot" in STYLES
    assert "brainrot" in VIDEO_TYPE_IDS


def test_brainrot_style_is_not_a_kids_style():
    assert is_kids_style("brainrot") is False


def test_brainrot_video_type_does_not_make_a_job_kids_content():
    """ACHTUNG case from the contract: a job whose video_type is the new
    "brainrot" type, on a channel with no kids signal set, must not be
    declared made_for_kids just because a same-named STYLE happens to exist."""
    channel = {"made_for_kids": 0, "style": ""}
    job = {"video_type": "brainrot", "style": None}
    assert _made_for_kids(channel, job) is False


def test_hyperframes_video_type_does_not_make_a_job_kids_content():
    channel = {"made_for_kids": 0, "style": ""}
    job = {"video_type": "hyperframes", "style": None}
    assert _made_for_kids(channel, job) is False


def test_brainrot_video_type_does_not_implicitly_select_the_brainrot_style():
    """`is_kids_style` reads job.style/channel.style only; passing the new
    video_type through it (the exact confusion this contract warns about)
    must not accidentally read as a kids style either."""
    assert is_kids_style("brainrot") is False
    # sanity: is_kidsong_type must equally never be fed a style by mistake
    # and mistake it for the "kids" style value
    assert is_kidsong_type("brainrot") is False
