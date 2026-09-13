"""Tests for the pure LLM-output handling in pipeline.script_gen.

These cover the parts that don't touch a backend: JSON extraction from messy
LLM replies and the normalization that protects the rest of the pipeline from
whatever shape the model actually returns.
"""
import pytest

from pipeline.script_gen import _extract_json, _normalize_script, _normalize_tags


# ------------------------------------------------------------- _extract_json --
def test_extract_json_plain_object():
    assert _extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_strips_code_fence():
    raw = '```json\n{"title": "hi", "lines": []}\n```'
    assert _extract_json(raw) == {"title": "hi", "lines": []}


def test_extract_json_ignores_surrounding_prose():
    raw = 'Sure! Here is your script:\n{"title": "x"}\nHope that helps.'
    assert _extract_json(raw) == {"title": "x"}


def test_extract_json_raises_when_no_object():
    with pytest.raises(ValueError):
        _extract_json("I could not do that.")


# ----------------------------------------------------------- _normalize_tags --
def test_normalize_tags_from_list():
    assert _normalize_tags(["Cats", "Dogs"]) == ["Cats", "Dogs"]


def test_normalize_tags_from_comma_string():
    # LLMs frequently return tags as one string instead of a list.
    assert _normalize_tags("shorts, brainrot, fyp") == ["shorts", "brainrot", "fyp"]


def test_normalize_tags_strips_hash_and_dupes():
    assert _normalize_tags(["#fyp", "FYP", "  fyp  "]) == ["fyp"]


def test_normalize_tags_coerces_non_strings():
    assert _normalize_tags([1, 2, "three"]) == ["1", "2", "three"]


@pytest.mark.parametrize("bad", [None, "", [], 42, {}])
def test_normalize_tags_falls_back_when_empty_or_wrong_type(bad):
    assert _normalize_tags(bad) == ["shorts", "brainrot", "fyp"]


# --------------------------------------------------------- _normalize_script --
def _min_script(**over):
    base = {"title": "t", "lines": [{"speaker": "narrator", "text": "hello"}]}
    base.update(over)
    return base


def test_normalize_script_happy_path():
    out = _normalize_script(_min_script(description="d", tags=["a"]))
    assert out["title"] == "t"
    assert out["description"] == "d"
    assert out["tags"] == ["a"]
    assert out["lines"] == [{"speaker": "narrator", "text": "hello"}]


def test_normalize_script_defaults_missing_fields():
    out = _normalize_script(_min_script())
    assert out["description"] == ""
    assert out["tags"] == ["shorts", "brainrot", "fyp"]


def test_normalize_script_coerces_non_string_title():
    # A numeric title would crash youtube_upload's title[:100]; it must become a str.
    out = _normalize_script(_min_script(title=12345))
    assert out["title"] == "12345"


def test_normalize_script_blank_title_uses_default():
    out = _normalize_script(_min_script(title="   "))
    assert out["title"] == "You won't believe this 🤯"


def test_normalize_script_unknown_speaker_becomes_narrator():
    out = _normalize_script(_min_script(lines=[{"speaker": "villain", "text": "hi"}]))
    assert out["lines"][0]["speaker"] == "narrator"


def test_normalize_script_drops_empty_and_non_dict_lines():
    lines = [
        {"speaker": "speaker_a", "text": "  keep me  "},
        {"speaker": "speaker_a", "text": "   "},  # whitespace only -> dropped
        "not a dict",  # -> dropped
    ]
    out = _normalize_script(_min_script(lines=lines))
    assert out["lines"] == [{"speaker": "speaker_a", "text": "keep me"}]


def test_normalize_script_raises_with_no_usable_lines():
    with pytest.raises(ValueError):
        _normalize_script({"title": "t", "lines": [{"speaker": "narrator", "text": ""}]})


def test_normalize_script_raises_on_non_dict_input():
    with pytest.raises(ValueError):
        _normalize_script(["not", "a", "dict"])
