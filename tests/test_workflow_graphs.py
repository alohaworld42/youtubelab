"""Validator for the ComfyUI API-format graphs under ``workflows/``.

Every graph here is hand-authored JSON (ComfyUI's "Save (API Format)" shape:
a dict of node-id -> {"class_type", "inputs", "_meta": {"title": ...}}) that
never runs through ComfyUI's own graph loader during CI — the only guard
against a graph that is malformed, internally inconsistent, or silently
incompatible with the patch-by-title contract in
``pipeline.kidsong.comfy.ComfyClient._apply_patches`` is this file.

Three kinds of checks:

1. Structural, for every ``workflows/*.json`` file: valid JSON, a dict, every
   node carries ``class_type`` + ``inputs``, no two nodes share a
   ``_meta.title`` (title patching would be ambiguous), and every
   ``[node_id, slot]`` link input actually resolves to a node in the same
   graph (a dangling one means the graph will not load in ComfyUI at all).

2. Contract, for every ``ltx23_t2v_*`` graph: it must carry every node title
   ``pipeline/kidsong/comfy.py`` patches on a text-to-video render (see
   ``generate.py``'s ``base_patches`` / comfy.py's ``_PRIMARY_INPUT``) —
   PROMPT, NEGATIVE, LORA_STYLE, SEED, FRAMES, WIDTH, HEIGHT,
   FILENAME_PREFIX. Missing one of these means a render under that graph
   raises ``KeyError`` from ``_apply_patches`` instead of rendering.

3. Look-specific, for the four non-toon looks derived from the toon graphs
   (``ltx23_t2v_explainer[_hires]``, ``ltx23_t2v_meme[_hires]``): the exact
   same node-title set as their toon pendant (the guarantee that the
   existing patch code needs no changes to drive them), and a neutralized
   style LoRA (``LORA_STYLE`` strength 0.0) since both looks are
   prompt-driven, not LoRA-driven.
"""
import json
from pathlib import Path

import pytest

from pipeline.kidsong.comfy import _PRIMARY_INPUT

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS_DIR = ROOT / "workflows"

WORKFLOW_PATHS = sorted(WORKFLOWS_DIR.glob("*.json"))
WORKFLOW_IDS = [p.name for p in WORKFLOW_PATHS]

# The node titles a t2v render unconditionally patches (generate.py's
# base_patches, applied to both the plain and "_hires" graph of a style):
# positive/negative prompt, the style LoRA, seed, frame count, resolution,
# and the save node. Cross-checked below against comfy.py's own
# _PRIMARY_INPUT map so this list cannot silently drift out of sync with the
# real patch contract. Deliberately EXCLUDES the image-conditioning titles
# (INPUT_IMAGE, FIRST_IMAGE, LAST_IMAGE, GUIDE_FIRST, GUIDE_LAST) — those
# belong to the i2v/flf2v graphs, not t2v.
T2V_REQUIRED_TITLES = frozenset({
    "PROMPT", "NEGATIVE", "LORA_STYLE", "SEED", "FRAMES",
    "WIDTH", "HEIGHT", "FILENAME_PREFIX",
})

# The four new non-toon-look graphs and the toon pendant each is derived
# from (same resolution/frame-count/sampler-step family, LoRA neutralized).
DERIVED_FROM = {
    "ltx23_t2v_explainer.json": "ltx23_t2v_toon.json",
    "ltx23_t2v_explainer_hires.json": "ltx23_t2v_toon_hires.json",
    "ltx23_t2v_meme.json": "ltx23_t2v_toon.json",
    "ltx23_t2v_meme_hires.json": "ltx23_t2v_toon_hires.json",
}


def _load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _titles(workflow):
    return {node.get("_meta", {}).get("title") for node in workflow.values()}


def _by_title(workflow):
    return {node.get("_meta", {}).get("title"): node for node in workflow.values()}


def test_workflows_dir_has_graphs_to_check():
    # A guard against the whole file silently passing on an empty glob (e.g.
    # a rename that broke WORKFLOWS_DIR would collapse every parametrized
    # test below to "no cases" rather than a visible failure).
    assert WORKFLOW_PATHS, f"No *.json graphs found under {WORKFLOWS_DIR}"


def test_t2v_required_titles_are_known_to_comfy_patch_contract():
    """Keep T2V_REQUIRED_TITLES honest against comfy.py's own map: every
    title we require here must be one comfy.py actually knows how to patch,
    and none of the image-conditioning titles (i2v/flf2v-only) leak in."""
    assert T2V_REQUIRED_TITLES <= set(_PRIMARY_INPUT)
    image_only = {"INPUT_IMAGE", "FIRST_IMAGE", "LAST_IMAGE",
                  "GUIDE_FIRST", "GUIDE_LAST"}
    assert not (T2V_REQUIRED_TITLES & image_only)


# --------------------------------------------------------------- structural ---
@pytest.mark.parametrize("path", WORKFLOW_PATHS, ids=WORKFLOW_IDS)
def test_graph_is_a_valid_node_dict(path):
    workflow = _load(path)
    assert isinstance(workflow, dict) and workflow, f"{path.name}: not a non-empty dict"
    for node_id, node in workflow.items():
        assert isinstance(node, dict), f"{path.name}: node {node_id!r} is not a dict"
        assert "class_type" in node, f"{path.name}: node {node_id!r} missing class_type"
        assert isinstance(node["class_type"], str) and node["class_type"], \
            f"{path.name}: node {node_id!r} has an empty class_type"
        assert "inputs" in node, f"{path.name}: node {node_id!r} missing inputs"
        assert isinstance(node["inputs"], dict), \
            f"{path.name}: node {node_id!r} inputs is not a dict"


@pytest.mark.parametrize("path", WORKFLOW_PATHS, ids=WORKFLOW_IDS)
def test_no_duplicate_node_titles(path):
    """A duplicate _meta.title makes patch-by-title (ComfyClient._apply_patches)
    ambiguous -- both nodes get the patch, or whichever the dict iteration
    order happens to hit last silently wins. This must never happen."""
    workflow = _load(path)
    titles = [
        node.get("_meta", {}).get("title")
        for node in workflow.values()
        if node.get("_meta", {}).get("title") is not None
    ]
    seen_twice = {t for t in titles if titles.count(t) > 1}
    assert not seen_twice, (
        f"{path.name}: duplicate _meta.title(s) {sorted(seen_twice)} -- "
        f"patching by title would be ambiguous."
    )


@pytest.mark.parametrize("path", WORKFLOW_PATHS, ids=WORKFLOW_IDS)
def test_no_dangling_link_references(path):
    """Every [node_id, slot] input link must resolve to a node in the same
    graph -- a dangling one means ComfyUI refuses to load the workflow."""
    workflow = _load(path)
    for node_id, node in workflow.items():
        for input_name, value in node.get("inputs", {}).items():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                target_id, _slot = value
                assert target_id in workflow, (
                    f"{path.name}: node {node_id!r} input {input_name!r} "
                    f"references non-existent node {target_id!r}"
                )


# ------------------------------------------------------------------ contract ---
T2V_PATHS = [p for p in WORKFLOW_PATHS if p.name.startswith("ltx23_t2v_")]
T2V_IDS = [p.name for p in T2V_PATHS]


def test_t2v_graphs_exist():
    assert T2V_PATHS, "No ltx23_t2v_*.json graphs found"


@pytest.mark.parametrize("path", T2V_PATHS, ids=T2V_IDS)
def test_t2v_graph_carries_every_patched_title(path):
    workflow = _load(path)
    titles = _titles(workflow)
    missing = T2V_REQUIRED_TITLES - titles
    assert not missing, (
        f"{path.name}: missing node title(s) {sorted(missing)} that "
        f"pipeline/kidsong/comfy.py patches on every t2v render -- a render "
        f"under this graph would raise KeyError from _apply_patches."
    )


# -------------------------------------------------------------- derived looks ---
@pytest.mark.parametrize("name,base_name", sorted(DERIVED_FROM.items()))
def test_derived_look_shares_its_toon_pendants_title_set(name, base_name):
    """explainer_clean/meme_pop must reuse the toon graph's exact node-title
    set so pipeline.kidsong.comfy._apply_patches (which patches by title)
    works unchanged against them."""
    new_graph = _load(WORKFLOWS_DIR / name)
    base_graph = _load(WORKFLOWS_DIR / base_name)
    assert _titles(new_graph) == _titles(base_graph), (
        f"{name}: node-title set differs from its pendant {base_name} -- "
        f"the shared patch code would break for one of them."
    )


@pytest.mark.parametrize("name", sorted(DERIVED_FROM))
def test_derived_look_has_lora_style_neutralized(name):
    """explainer_clean and meme_pop are prompt-driven looks: config.example.json
    defines both with style_strength 0.0, and the graph's own baked-in
    default must already be 0.0 so a direct load (without a patch) matches."""
    workflow = _load(WORKFLOWS_DIR / name)
    node = _by_title(workflow)["LORA_STYLE"]
    assert node["inputs"].get("strength_model") == 0.0, (
        f"{name}: LORA_STYLE.strength_model must default to 0.0"
    )


@pytest.mark.parametrize("name,base_name", sorted(DERIVED_FROM.items()))
def test_derived_look_matches_toon_pendant_hardware_footprint(name, base_name):
    """VRAM guardrail: a new look must not raise resolution, frame count, or
    sampler step count above its toon pendant (RTX 5070, 12GB VRAM)."""
    new_graph = _load(WORKFLOWS_DIR / name)
    base_graph = _load(WORKFLOWS_DIR / base_name)
    new_by_title = _by_title(new_graph)
    base_by_title = _by_title(base_graph)
    for title in ("WIDTH", "HEIGHT", "FRAMES"):
        assert new_by_title[title]["inputs"]["value"] == \
            base_by_title[title]["inputs"]["value"], \
            f"{name}: {title} differs from {base_name}"
    for title in ("SIGMAS", "SIGMAS_REFINE"):
        if title in new_by_title or title in base_by_title:
            assert new_by_title[title]["inputs"]["sigmas"] == \
                base_by_title[title]["inputs"]["sigmas"], \
                f"{name}: {title} sampler schedule differs from {base_name}"
