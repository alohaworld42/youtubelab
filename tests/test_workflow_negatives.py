"""Tests for the NEGATIVE prompt baked into the ComfyUI LTX-2.3 workflow graphs.

``workflows/*.json`` are the authoritative, API-format graphs the kidsong
pipeline POSTs verbatim to ComfyUI (see AGENTS.md). They are patched by
``_meta.title``, never by node id, so the load-bearing titles (PROMPT,
NEGATIVE, SEED, WIDTH, HEIGHT, FRAMES, FILENAME_PREFIX, INPUT_IMAGE,
LORA_STYLE) must survive any edit. This module only checks the static JSON
on disk — no ComfyUI, no GPU, no network.
"""
import json
import os

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WORKFLOWS_DIR = os.path.join(_ROOT, "workflows")

# (filename, expected node count, extra patch-target titles beyond the
# common set shared by every graph)
_FILES = {
    "t2v": ("ltx23_t2v_toon.json", 23, []),
    "hires": ("ltx23_t2v_toon_hires.json", 33, []),
    "i2v": ("ltx23_i2v_toon.json", 26, ["INPUT_IMAGE"]),
    "i2v_hires": ("ltx23_i2v_toon_hires.json", 37, ["INPUT_IMAGE"]),
}

_COMMON_PATCH_TITLES = [
    "PROMPT", "NEGATIVE", "SEED", "WIDTH", "HEIGHT", "FRAMES",
    "FILENAME_PREFIX", "LORA_STYLE",
]

# Terms measured (54 reviewed takes, 39 failures) to have helped before this
# change; regressing any of these is a hard failure.
_LEGACY_EFFECTIVE_TERMS = [
    "pale-skinned child", "red-haired child", "blond child",
    "shirtless child", "bare chest", "undressed", "underwear",
    "deformed hands", "fused fingers", "watermark", "morphing",
]

# Terms added in this change to attack the dominant measured failure mode:
# the model inventing an extra, off-cast child (67% of identity failures in
# WIDE/group shots) plus cross-shot identity drift and wardrobe swaps.
_NEW_CROWD_TERMS = [
    "crowd", "many children", "extra child", "background children",
    "unnamed child", "additional kids", "group of strangers",
    "duplicate characters", "cloned face", "twins",
]
_NEW_IDENTITY_DRIFT_TERMS = [
    "inconsistent character design", "off-model character",
    "changing hairstyle", "changing outfit", "different child each frame",
]
_NEW_WARDROBE_TERMS = [
    "mismatched outfit", "wrong shirt color", "swapped clothing",
]
_NEW_SAFETY_TERMS = [
    "jewellery", "earrings", "necklace", "piercing",
]
_NEW_TERMS = (
    _NEW_CROWD_TERMS + _NEW_IDENTITY_DRIFT_TERMS + _NEW_WARDROBE_TERMS
    + _NEW_SAFETY_TERMS
)

# Terms added in the de-creep overhaul (W3): rendered toddlers walking/
# looming toward the camera, an approach-and-stare motion pattern no
# existing negative term covered.
_NEW_DECREEP_TERMS = [
    "walking toward the camera", "approaching the camera",
    "looming toward the lens", "staring into the lens",
]

# The over-fitted, song-specific term this change generalises away.
_DROPPED_OVERFITTED_TERM = "child holding two toothbrushes"

# An over-long negative dilutes every term's weight; keep some headroom
# under this cap.
_TERM_CAP = 120


# ------------------------------------------------------------------ helpers ---
def _load(key):
    filename, _count, _extra_titles = _FILES[key]
    with open(os.path.join(_WORKFLOWS_DIR, filename), encoding="utf-8") as fh:
        return json.load(fh)


def _negative_node(workflow):
    matches = [
        node for node in workflow.values()
        if node.get("_meta", {}).get("title") == "NEGATIVE"
    ]
    assert len(matches) == 1, "expected exactly one NEGATIVE node"
    return matches[0]


def _negative_terms(workflow):
    text = _negative_node(workflow)["inputs"]["text"]
    return [t.strip() for t in text.split(",")]


# --------------------------------------------------------------- API format ---
@pytest.mark.parametrize("key", list(_FILES))
def test_workflow_file_parses_as_json_and_is_api_format(key):
    workflow = _load(key)
    assert isinstance(workflow, dict) and workflow
    # API format is {node_id: {class_type, inputs, _meta}}; UI format has
    # top-level "nodes"/"links"/"version" keys instead.
    assert "nodes" not in workflow and "links" not in workflow
    for node_id, node in workflow.items():
        assert node_id.isdigit(), f"non-numeric node id {node_id!r}"
        assert "class_type" in node and "inputs" in node and "_meta" in node


@pytest.mark.parametrize("key", list(_FILES))
def test_node_ids_and_count_unchanged(key):
    filename, expected_count, _extra_titles = _FILES[key]
    workflow = _load(key)
    assert len(workflow) == expected_count, (
        f"{filename}: expected {expected_count} nodes, found {len(workflow)}"
    )
    assert {int(nid) for nid in workflow} == set(range(1, expected_count + 1))


@pytest.mark.parametrize("key", list(_FILES))
def test_all_patch_target_titles_present(key):
    filename, _count, extra_titles = _FILES[key]
    workflow = _load(key)
    titles = {node.get("_meta", {}).get("title") for node in workflow.values()}
    for title in _COMMON_PATCH_TITLES + extra_titles:
        assert title in titles, f"{filename}: missing patch-target title {title!r}"


# ------------------------------------------------------------ NEGATIVE node ---
@pytest.mark.parametrize("key", list(_FILES))
def test_negative_node_is_a_single_cliptextencode(key):
    workflow = _load(key)
    node = _negative_node(workflow)
    assert node["class_type"] == "CLIPTextEncode"
    assert isinstance(node["inputs"].get("text"), str) and node["inputs"]["text"]


def test_negative_text_identical_across_all_files():
    texts = {key: _negative_node(_load(key))["inputs"]["text"] for key in _FILES}
    values = list(texts.values())
    assert all(v == values[0] for v in values), (
        f"NEGATIVE text diverged between files: {texts}"
    )


def test_prompt_node_untouched_placeholder():
    # Guard rail: this change must not have touched the PROMPT node.
    for key in _FILES:
        workflow = _load(key)
        prompt_node = next(
            n for n in workflow.values() if n["_meta"]["title"] == "PROMPT"
        )
        assert prompt_node["inputs"]["text"] == "P1x4r, 3D CGI toon style, placeholder"


# --------------------------------------------------------------- term content ---
@pytest.mark.parametrize("term", _LEGACY_EFFECTIVE_TERMS)
@pytest.mark.parametrize("key", list(_FILES))
def test_measured_effective_legacy_terms_survive(key, term):
    terms = _negative_terms(_load(key))
    assert term in terms, f"{key}: regressed measured-effective term {term!r}"


@pytest.mark.parametrize("term", _NEW_TERMS)
@pytest.mark.parametrize("key", list(_FILES))
def test_new_cast_integrity_terms_present(key, term):
    terms = _negative_terms(_load(key))
    assert term in terms, f"{key}: missing new cast-integrity term {term!r}"


@pytest.mark.parametrize("term", _NEW_DECREEP_TERMS)
@pytest.mark.parametrize("key", list(_FILES))
def test_new_decreep_terms_present(key, term):
    terms = _negative_terms(_load(key))
    assert term in terms, f"{key}: missing new de-creep term {term!r}"


@pytest.mark.parametrize("key", list(_FILES))
def test_duplicate_objects_term_retained(key):
    terms = _negative_terms(_load(key))
    assert "duplicate objects in one hand" in terms


@pytest.mark.parametrize("key", list(_FILES))
def test_overfitted_toothbrush_term_generalised_away(key):
    terms = _negative_terms(_load(key))
    assert _DROPPED_OVERFITTED_TERM not in terms


@pytest.mark.parametrize("key", list(_FILES))
def test_no_term_appears_twice(key):
    terms = _negative_terms(_load(key))
    assert len(terms) == len(set(terms)), "duplicate term(s) in NEGATIVE string"


@pytest.mark.parametrize("key", list(_FILES))
def test_term_count_under_cap(key):
    terms = _negative_terms(_load(key))
    assert 0 < len(terms) <= _TERM_CAP


@pytest.mark.parametrize("key", list(_FILES))
def test_negative_string_is_a_single_flat_comma_separated_line(key):
    text = _negative_node(_load(key))["inputs"]["text"]
    assert "\n" not in text
    # every term round-trips non-empty after stripping (no stray double commas)
    terms = [t.strip() for t in text.split(",")]
    assert all(terms)
