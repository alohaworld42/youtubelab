"""Covers the two brainrot-job-killer fixes:

1. prompts/brainrot.txt exists, loads, stays on the monetization-safe rules
   every other video type follows, and names no foreign brand.
2. generate_script() can target a non-English script language (read from
   cfg["kidsong"]["language"], overridable via the `language` kwarg) without
   changing a single byte of the prompt any existing (English) channel gets.

No network: the LLM call (pipeline.script_gen.complete) is monkeypatched:
these tests only ever inspect the PROMPT STRING built for it, never a real
LLM response.
"""
import json
import os

import pytest

import pipeline.script_gen as sg
from pipeline.script_gen import (
    _language_instruction,
    _load_prompt,
    _normalize_target_language,
    _resolve_language,
    generate_script,
)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Small, deliberately non-exhaustive sample of protected third-party names —
# same spirit as tests/test_video_types.py's FOREIGN_BRANDS list.
FOREIGN_BRANDS = ("cocomelon", "disney", "pixar", "peppa", "bluey", "mrbeast")

MONETIZATION_MARKERS = (
    "insult",
    "hate",
    "sexual",
    "identifiable private individual",
    "trademarked",
)


def _fake_complete(reply):
    """A monkeypatch replacement for sg.complete that records the prompt it
    was called with and returns a minimal, always-valid script JSON."""
    captured = {}

    def fake(user, cfg, system=None):
        captured["user"] = user
        captured["system"] = system
        return reply

    return fake, captured


VALID_REPLY = json.dumps({"title": "t", "lines": [{"speaker": "narrator", "text": "hi"}]})


# --------------------------------------------------------- prompts/brainrot ----
def test_load_prompt_brainrot_does_not_raise():
    text = _load_prompt("brainrot", _REPO_ROOT)
    assert text.strip()
    assert "{{TOPIC}}" in text


def test_brainrot_prompt_names_no_foreign_brand():
    text = _load_prompt("brainrot", _REPO_ROOT).lower()
    hits = [b for b in FOREIGN_BRANDS if b in text]
    assert not hits, f"prompts/brainrot.txt mentions {hits}"


def test_brainrot_prompt_states_monetization_boundaries():
    # Collapse whitespace (word-wrapped .txt lines) so a marker phrase that
    # happens to wrap across two lines in the file still matches.
    text = " ".join(_load_prompt("brainrot", _REPO_ROOT).lower().split())
    missing = [m for m in MONETIZATION_MARKERS if m not in text]
    assert not missing, f"prompts/brainrot.txt is missing boundary markers: {missing}"


def test_brainrot_prompt_matches_facts_and_hyperframes_output_contract():
    """Same JSON shape generate_script's parser expects: title/description/
    tags/lines, with speaker+text on each line."""
    text = _load_prompt("brainrot", _REPO_ROOT)
    for key in ('"title"', '"description"', '"tags"', '"lines"', '"speaker"', '"text"'):
        assert key in text


# ------------------------------------------------ _normalize_target_language --
@pytest.mark.parametrize("bad", [42, ["de"], {}, "", "   ", None])
def test_normalize_target_language_falls_back_to_en_on_malformed_input(bad):
    assert _normalize_target_language(bad) == "en"


@pytest.mark.parametrize("raw,expected", [
    ("de", "de"), ("DE", "de"), ("de-DE", "de"), ("  de  ", "de"), ("en", "en"),
])
def test_normalize_target_language_normalizes_valid_tags(raw, expected):
    assert _normalize_target_language(raw) == expected


# -------------------------------------------------------------- _resolve_language --
def test_resolve_language_prefers_explicit_kwarg_over_config():
    cfg = {"kidsong": {"language": "de"}}
    assert _resolve_language(cfg, language="en") == "en"


def test_resolve_language_reads_kidsong_language_key():
    cfg = {"kidsong": {"language": "de"}}
    assert _resolve_language(cfg) == "de"


def test_resolve_language_defaults_to_en_when_unset():
    assert _resolve_language({}) == "en"
    assert _resolve_language({"kidsong": {}}) == "en"


@pytest.mark.parametrize("bad", [42, ["de"], ""])
def test_resolve_language_falls_back_to_en_on_broken_config_value(bad):
    assert _resolve_language({"kidsong": {"language": bad}}) == "en"


# ---------------------------------------------------------- _language_instruction --
def test_language_instruction_is_empty_for_english():
    assert _language_instruction("en") == ""


def test_language_instruction_names_german_for_de():
    instr = _language_instruction("de")
    assert instr
    assert "german" in instr.lower()


# ------------------------------------------- generate_script byte-identical(en) --
def _base_cfg(**kidsong_overrides):
    cfg = {"_root": _REPO_ROOT, "llm": {"backend": "groq", "groq_model": "x"}}
    if kidsong_overrides:
        cfg["kidsong"] = dict(kidsong_overrides)
    return cfg


def _expected_pre_feature_prompt(video_type, topic, extra_context, root):
    """Reimplements generate_script's PRE-feature prompt assembly verbatim,
    as the regression oracle: whatever this produces is what every existing
    (English) channel must still get, byte for byte."""
    template = _load_prompt(video_type, root)
    user = template.replace("{{TOPIC}}", topic or "pick a wildly engaging topic yourself")
    if extra_context:
        user += f"\n\nChannel context (match this niche and audience): {extra_context}"
    return user


@pytest.mark.parametrize("cfg_kwargs,language_kwarg", [
    ({}, None),                       # no kidsong block at all
    ({"language": "en"}, None),       # explicit en in config
    ({}, "en"),                       # explicit en via kwarg
    ({"language": None}, None),       # language key present but None
    ({"language": 42}, None),         # malformed config value -> still en
    ({"language": []}, None),         # malformed config value -> still en
    ({"language": ""}, None),         # malformed config value -> still en
])
def test_generate_script_prompt_byte_identical_to_pre_feature_state(
    monkeypatch, cfg_kwargs, language_kwarg
):
    fake, captured = _fake_complete(VALID_REPLY)
    monkeypatch.setattr(sg, "complete", fake)

    cfg = _base_cfg(**cfg_kwargs)
    kwargs = {} if language_kwarg is None else {"language": language_kwarg}
    generate_script("facts", topic="octopuses", cfg=cfg, extra_context="a science channel", **kwargs)

    expected = _expected_pre_feature_prompt("facts", "octopuses", "a science channel", _REPO_ROOT)
    assert captured["user"] == expected


def test_generate_script_prompt_byte_identical_with_no_topic_or_context(monkeypatch):
    fake, captured = _fake_complete(VALID_REPLY)
    monkeypatch.setattr(sg, "complete", fake)

    cfg = _base_cfg()
    generate_script("facts", cfg=cfg)

    expected = _expected_pre_feature_prompt("facts", None, None, _REPO_ROOT)
    assert captured["user"] == expected


# --------------------------------------------------- generate_script German ----
def test_generate_script_prompt_carries_german_instruction_via_config(monkeypatch):
    fake, captured = _fake_complete(VALID_REPLY)
    monkeypatch.setattr(sg, "complete", fake)

    cfg = _base_cfg(language="de")
    generate_script("facts", topic="octopuses", cfg=cfg, extra_context="a science channel")

    baseline = _expected_pre_feature_prompt("facts", "octopuses", "a science channel", _REPO_ROOT)
    assert captured["user"] != baseline
    assert captured["user"].startswith(baseline)  # strictly additive suffix
    assert "german" in captured["user"].lower()


def test_generate_script_prompt_carries_german_instruction_via_kwarg_override(monkeypatch):
    fake, captured = _fake_complete(VALID_REPLY)
    monkeypatch.setattr(sg, "complete", fake)

    # config says English, explicit kwarg overrides to German
    cfg = _base_cfg(language="en")
    generate_script("facts", topic="octopuses", cfg=cfg, language="de")

    assert "german" in captured["user"].lower()


def test_generate_script_kwarg_overrides_german_config_back_to_english(monkeypatch):
    fake, captured = _fake_complete(VALID_REPLY)
    monkeypatch.setattr(sg, "complete", fake)

    cfg = _base_cfg(language="de")
    generate_script("facts", topic="octopuses", cfg=cfg, language="en")

    baseline = _expected_pre_feature_prompt("facts", "octopuses", None, _REPO_ROOT)
    assert captured["user"] == baseline


# --------------------------------------------------- brainrot job end-to-end ----
def test_generate_script_brainrot_no_longer_raises(monkeypatch):
    """The bug this fixes: a scheduled brainrot job used to die immediately
    with FileNotFoundError before ever reaching the LLM."""
    fake, captured = _fake_complete(VALID_REPLY)
    monkeypatch.setattr(sg, "complete", fake)

    cfg = _base_cfg()
    out = generate_script("brainrot", topic="a cursed kitchen gadget", cfg=cfg)

    assert out["title"] == "t"
    assert captured["user"]
