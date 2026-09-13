"""Structural tests for workflows/zimage_ref.json — the Z-Image Turbo
text->image graph that renders one canonical CHARACTER REFERENCE still.

This graph is deliberately NOT in tests/test_workflow_negatives.py::_FILES:
that module bakes the LTX toon PROMPT placeholder and the shared NEGATIVE
string and would reject a Z-Image graph outright. These checks instead pin the
load-bearing facts derived from ComfyUI's own /object_info and the official
`image_z_image_turbo` template: the CLIP loads with type `lumina2`, the turbo
sampler settings (ModelSamplingAuraFlow shift, 8 steps, cfg 1, res_multistep /
simple), and the patch-target titles the pipeline drives (PROMPT, SEED, WIDTH,
HEIGHT, FILENAME_PREFIX). Static JSON on disk only — no ComfyUI, no GPU, no
network.
"""
import json
import os

from pipeline.kidsong.comfy import ComfyClient

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WF_PATH = os.path.join(_ROOT, "workflows", "zimage_ref.json")


def _load():
    with open(_WF_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def _by_title(wf, title):
    matches = [n for n in wf.values() if n.get("_meta", {}).get("title") == title]
    assert len(matches) == 1, f"expected exactly one node titled {title!r}, got {len(matches)}"
    return matches[0]


def _by_class(wf, class_type):
    return [n for n in wf.values() if n.get("class_type") == class_type]


# --------------------------------------------------------------- API format ---
def test_parses_as_api_format():
    wf = _load()
    assert isinstance(wf, dict) and wf
    assert "nodes" not in wf and "links" not in wf
    for node_id, node in wf.items():
        assert node_id.isdigit(), f"non-numeric node id {node_id!r}"
        assert "class_type" in node and "inputs" in node and "_meta" in node


def test_patch_target_titles_present():
    wf = _load()
    titles = {n.get("_meta", {}).get("title") for n in wf.values()}
    for title in ("PROMPT", "SEED", "WIDTH", "HEIGHT", "FILENAME_PREFIX"):
        assert title in titles, f"missing patch-target title {title!r}"


# --------------------------------------------------------------- loaders ---
def test_unet_loader_loads_zimage_turbo():
    node = _by_title(_load(), "UNET_LOADER")
    assert node["class_type"] == "UNETLoader"
    assert node["inputs"]["unet_name"] == "z_image_turbo_bf16.safetensors"
    assert node["inputs"]["weight_dtype"] == "default"


def test_clip_loader_uses_qwen_with_lumina2_type():
    # The single most guess-prone field: Z-Image's CLIPLoader `type` is
    # `lumina2` (verified against the official image_z_image_turbo template and
    # the live /object_info enum), NOT `qwen_image` or any other.
    node = _by_title(_load(), "CLIP_LOADER")
    assert node["class_type"] == "CLIPLoader"
    assert node["inputs"]["clip_name"] == "qwen_3_4b.safetensors"
    assert node["inputs"]["type"] == "lumina2"


def test_vae_loader_loads_ae():
    node = _by_title(_load(), "VAE_LOADER")
    assert node["class_type"] == "VAELoader"
    assert node["inputs"]["vae_name"] == "ae.safetensors"


# ---------------------------------------------------------- turbo sampling ---
def test_model_sampling_auraflow_present_and_wired():
    wf = _load()
    ms = _by_title(wf, "MODEL_SAMPLING")
    assert ms["class_type"] == "ModelSamplingAuraFlow"
    # driven by the UNETLoader, feeds the KSampler's model input
    assert ms["inputs"]["model"][0] == "1"
    assert float(ms["inputs"]["shift"]) == 3.0


def test_ksampler_uses_turbo_settings():
    ks = _by_title(_load(), "SAMPLER")
    assert ks["class_type"] == "KSampler"
    assert ks["inputs"]["steps"] == 8
    assert float(ks["inputs"]["cfg"]) == 1.0
    assert ks["inputs"]["sampler_name"] == "res_multistep"
    assert ks["inputs"]["scheduler"] == "simple"
    assert float(ks["inputs"]["denoise"]) == 1.0
    # every required KSampler input is present (positive/negative/latent/seed/model)
    for name in ("model", "positive", "negative", "latent_image", "seed"):
        assert name in ks["inputs"], f"KSampler missing {name!r}"


# ---------------------------------------------------------- conditioning ---
def test_positive_encoder_is_zimage_omni_titled_prompt():
    node = _by_title(_load(), "PROMPT")
    assert node["class_type"] == "TextEncodeZImageOmni"
    # the Omni node's text input is named `prompt` (not `text`); refs.py patches
    # it via the dict form for exactly this reason.
    assert isinstance(node["inputs"].get("prompt"), str) and node["inputs"]["prompt"]
    assert node["inputs"]["auto_resize_images"] is True


def test_negative_is_zeroed_out_positive():
    wf = _load()
    neg = _by_title(wf, "NEGATIVE")
    assert neg["class_type"] == "ConditioningZeroOut"
    # zero-out the positive conditioning (the turbo template's negative, since
    # cfg 1.0 uses no real negative); the KSampler's negative reads from it.
    prompt_id = next(nid for nid, n in wf.items()
                     if n.get("_meta", {}).get("title") == "PROMPT")
    assert neg["inputs"]["conditioning"][0] == prompt_id
    ks = _by_title(wf, "SAMPLER")
    neg_id = next(nid for nid, n in wf.items()
                  if n.get("_meta", {}).get("title") == "NEGATIVE")
    assert ks["inputs"]["negative"][0] == neg_id
    assert ks["inputs"]["positive"][0] == prompt_id


def test_latent_is_empty_sd3_driven_by_width_height():
    wf = _load()
    latent = _by_title(wf, "LATENT")
    assert latent["class_type"] == "EmptySD3LatentImage"
    width_id = next(nid for nid, n in wf.items()
                    if n.get("_meta", {}).get("title") == "WIDTH")
    height_id = next(nid for nid, n in wf.items()
                     if n.get("_meta", {}).get("title") == "HEIGHT")
    assert latent["inputs"]["width"][0] == width_id
    assert latent["inputs"]["height"][0] == height_id
    ks = _by_title(wf, "SAMPLER")
    latent_id = next(nid for nid, n in wf.items()
                     if n.get("_meta", {}).get("title") == "LATENT")
    assert ks["inputs"]["latent_image"][0] == latent_id


def test_saves_a_still_image():
    node = _by_title(_load(), "FILENAME_PREFIX")
    assert node["class_type"] == "SaveImage"
    assert isinstance(node["inputs"].get("filename_prefix"), str)
    # no SaveVideo/CreateVideo in a still-image reference graph
    assert not _by_class(_load(), "SaveVideo")
    assert not _by_class(_load(), "CreateVideo")


# ------------------------------------------------------------ patchability ---
def test_reference_patch_set_applies_without_keyerror():
    """The exact patch set refs.generate_reference sends must land on the right
    inputs (PROMPT->prompt, SEED/WIDTH/HEIGHT->value, FILENAME_PREFIX->
    filename_prefix) with no missing-title KeyError."""
    wf = _load()
    patched = ComfyClient._apply_patches(wf, {
        "PROMPT": {"prompt": "REF-PROMPT"},
        "SEED": {"value": 4242},
        "WIDTH": {"value": 640},
        "HEIGHT": {"value": 896},
        "FILENAME_PREFIX": {"filename_prefix": "cast_refs/zuri"},
    })
    assert _by_title(patched, "PROMPT")["inputs"]["prompt"] == "REF-PROMPT"
    assert _by_title(patched, "SEED")["inputs"]["value"] == 4242
    assert _by_title(patched, "WIDTH")["inputs"]["value"] == 640
    assert _by_title(patched, "HEIGHT")["inputs"]["value"] == 896
    assert _by_title(patched, "FILENAME_PREFIX")["inputs"]["filename_prefix"] == "cast_refs/zuri"
