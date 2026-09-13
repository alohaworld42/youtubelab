"""Tests for tools/comfy_workflow_to_ui.py — the API -> UI workflow converter.

Everything here runs against a stubbed ``/object_info``; no network, no
ComfyUI server. The stub mirrors the real shapes we depend on:
``input_order`` (authoritative widget ordering), ``control_after_generate``
(appends a trailing widget), ``image_upload`` (ditto), inline combo option
lists, optional link inputs, and multi-output nodes.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

from comfy_workflow_to_ui import (  # noqa: E402
    ConversionError,
    convert_workflow,
    is_widget_spec,
    link_plan,
    verify_conversion,
    widget_plan,
)


# --------------------------------------------------------------- fixtures ---
OBJECT_INFO = {
    "CheckpointLoaderSimple": {
        "input": {"required": {"ckpt_name": [["model_a.safetensors"], {}]}},
        "input_order": {"required": ["ckpt_name"]},
        "output": ["MODEL", "CLIP", "VAE"],
        "output_name": ["MODEL", "CLIP", "VAE"],
    },
    "CLIPTextEncode": {
        "input": {"required": {
            "text": ["STRING", {"multiline": True}],
            "clip": ["CLIP", {}],
        }},
        "input_order": {"required": ["text", "clip"]},
        "output": ["CONDITIONING"],
        "output_name": ["CONDITIONING"],
    },
    "PrimitiveInt": {
        # control_after_generate as a string default -> trailing "fixed"
        "input": {"required": {
            "value": ["INT", {"control_after_generate": "fixed"}]}},
        "input_order": {"required": ["value"]},
        "output": ["INT"],
        "output_name": ["INT"],
    },
    "RandomNoise": {
        # control_after_generate as a bool -> we choose "fixed"
        "input": {"required": {
            "noise_seed": ["INT", {"default": 0, "control_after_generate": True}]}},
        "input_order": {"required": ["noise_seed"]},
        "output": ["NOISE"],
        "output_name": ["NOISE"],
    },
    "LoadImage": {
        "input": {"required": {"image": [["example.png"], {"image_upload": True}]}},
        "input_order": {"required": ["image"]},
        "output": ["IMAGE", "MASK"],
        "output_name": ["IMAGE", "MASK"],
    },
    "EmptyLatent": {
        # INT widgets that the pipeline drives via links
        "input": {"required": {
            "width": ["INT", {"default": 768}],
            "height": ["INT", {"default": 512}],
            "batch_size": ["INT", {"default": 1}],
        }},
        "input_order": {"required": ["width", "height", "batch_size"]},
        "output": ["LATENT"],
        "output_name": ["LATENT"],
    },
    "SaveVideo": {
        "input": {"required": {
            "video": ["VIDEO", {}],
            "filename_prefix": ["STRING", {"default": "video/ComfyUI"}],
            "format": ["COMBO", {"default": "auto", "options": ["auto", "mp4"]}],
        }},
        "input_order": {"required": ["video", "filename_prefix", "format"],
                        "hidden": ["prompt", "extra_pnginfo"]},
        "output": ["VIDEO"],
        "output_name": ["VIDEO"],
    },
    "CreateVideo": {
        # 'audio' is an OPTIONAL link input, left unconnected by the workflow
        "input": {
            "required": {"images": ["IMAGE", {}], "fps": ["FLOAT", {"default": 30.0}]},
            "optional": {"audio": ["AUDIO", {}]},
        },
        "input_order": {"required": ["images", "fps"], "optional": ["audio"]},
        "output": ["VIDEO"],
        "output_name": ["VIDEO"],
    },
}


def api_workflow():
    """A small but representative API-format graph."""
    return {
        "1": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": "model_a.safetensors"},
              "_meta": {"title": "CHECKPOINT"}},
        "2": {"class_type": "CLIPTextEncode",
              "inputs": {"text": "a happy toon puppy", "clip": ["1", 1]},
              "_meta": {"title": "PROMPT"}},
        "3": {"class_type": "PrimitiveInt", "inputs": {"value": 512},
              "_meta": {"title": "WIDTH"}},
        "4": {"class_type": "PrimitiveInt", "inputs": {"value": 896},
              "_meta": {"title": "HEIGHT"}},
        "5": {"class_type": "EmptyLatent",
              "inputs": {"width": ["3", 0], "height": ["4", 0], "batch_size": 1},
              "_meta": {"title": "LATENT"}},
        "6": {"class_type": "RandomNoise", "inputs": {"noise_seed": 42},
              "_meta": {"title": "SEED"}},
        "7": {"class_type": "CreateVideo",
              "inputs": {"images": ["8", 0], "fps": 24},
              "_meta": {"title": "CREATE_VIDEO"}},
        "8": {"class_type": "LoadImage", "inputs": {"image": "shot.png"},
              "_meta": {"title": "INPUT_IMAGE"}},
        "9": {"class_type": "SaveVideo",
              "inputs": {"video": ["7", 0], "filename_prefix": "kidsong/x",
                         "format": "auto"},
              "_meta": {"title": "FILENAME_PREFIX"}},
    }


@pytest.fixture()
def converted():
    return convert_workflow(api_workflow(), OBJECT_INFO)


# ------------------------------------------------------------ spec basics ---
def test_is_widget_spec_distinguishes_widgets_from_links():
    assert is_widget_spec(["INT", {}])
    assert is_widget_spec(["STRING", {}])
    assert is_widget_spec(["COMBO", {"options": ["a"]}])
    assert is_widget_spec([["a.safetensors", "b.safetensors"], {}])  # inline combo
    assert not is_widget_spec(["MODEL", {}])
    assert not is_widget_spec(["LATENT", {}])
    assert not is_widget_spec([])


def test_widget_plan_appends_control_after_generate():
    assert [n for n, _k, _d in widget_plan(OBJECT_INFO["RandomNoise"])] == \
        ["noise_seed", "noise_seed"]
    kinds = [k for _n, k, _d in widget_plan(OBJECT_INFO["RandomNoise"])]
    assert kinds == ["value", "control"]


def test_widget_plan_appends_upload_widget():
    plan = widget_plan(OBJECT_INFO["LoadImage"])
    assert [k for _n, k, _d in plan] == ["value", "upload"]
    assert plan[1][2] == "image"


def test_widget_plan_follows_input_order_and_skips_links():
    names = [n for n, k, _d in widget_plan(OBJECT_INFO["SaveVideo"]) if k == "value"]
    assert names == ["filename_prefix", "format"]      # 'video' is a link
    assert link_plan(OBJECT_INFO["SaveVideo"]) == [("video", "VIDEO")]


def test_widget_plan_skips_hidden_inputs():
    # SaveVideo declares hidden prompt/extra_pnginfo; they must not become widgets
    names = [n for n, _k, _d in widget_plan(OBJECT_INFO["SaveVideo"])]
    assert "prompt" not in names and "extra_pnginfo" not in names


# -------------------------------------------------------------- structure ---
def test_top_level_shape_is_ui_format(converted):
    # 'version' is what ComfyUI's validateComfyWorkflow() checks first.
    assert converted["version"] == 0.4
    for key in ("nodes", "links", "last_node_id", "last_link_id", "extra"):
        assert key in converted
    assert converted["last_node_id"] == 9
    assert converted["last_link_id"] == len(converted["links"])


def test_every_node_survives_with_class_and_properties(converted):
    src = api_workflow()
    assert len(converted["nodes"]) == len(src)
    by_id = {n["id"]: n for n in converted["nodes"]}
    for nid, node in src.items():
        assert by_id[int(nid)]["type"] == node["class_type"]
        assert by_id[int(nid)]["properties"]["Node name for S&R"] == \
            node["class_type"]
        assert by_id[int(nid)]["mode"] == 0


def test_meta_titles_are_preserved_as_node_titles(converted):
    """The pipeline patches by _meta.title, so titles are a hard contract."""
    src = api_workflow()
    got = {n["id"]: n.get("title") for n in converted["nodes"]}
    for nid, node in src.items():
        assert got[int(nid)] == node["_meta"]["title"]
    for title in ("PROMPT", "SEED", "WIDTH", "HEIGHT", "FILENAME_PREFIX",
                  "INPUT_IMAGE"):
        assert title in got.values()


# ---------------------------------------------------------- widget values ---
def test_widgets_values_ordering_matches_object_info(converted):
    by_title = {n["title"]: n for n in converted["nodes"]}
    # ordered exactly as input_order dictates, links excluded
    assert by_title["FILENAME_PREFIX"]["widgets_values"] == ["kidsong/x", "auto"]
    assert by_title["PROMPT"]["widgets_values"] == ["a happy toon puppy"]
    assert by_title["CREATE_VIDEO"]["widgets_values"] == [24]


def test_seed_and_int_nodes_get_control_after_generate_value(converted):
    by_title = {n["title"]: n for n in converted["nodes"]}
    assert by_title["SEED"]["widgets_values"] == [42, "fixed"]
    assert by_title["WIDTH"]["widgets_values"] == [512, "fixed"]
    assert by_title["HEIGHT"]["widgets_values"] == [896, "fixed"]


def test_load_image_gets_upload_widget_value(converted):
    by_title = {n["title"]: n for n in converted["nodes"]}
    assert by_title["INPUT_IMAGE"]["widgets_values"] == ["shot.png", "image"]


def test_linked_widget_keeps_its_positional_slot(converted):
    """width/height are link-driven but must still occupy widgets_values."""
    latent = next(n for n in converted["nodes"] if n["title"] == "LATENT")
    # 3 widgets: width, height, batch_size — defaults for the two linked ones
    assert latent["widgets_values"] == [768, 512, 1]
    slots = {i["name"]: i for i in latent["inputs"]}
    for name in ("width", "height"):
        assert slots[name]["shape"] == 7
        assert slots[name]["widget"] == {"name": name}
        assert slots[name]["link"] is not None
    assert "batch_size" not in slots      # unconnected widget -> no slot


def test_widget_count_always_matches_object_info(converted):
    for node in converted["nodes"]:
        expected = len(widget_plan(OBJECT_INFO[node["type"]]))
        assert len(node["widgets_values"]) == expected, node["type"]


# ------------------------------------------------------------------ links ---
def test_links_are_wired_with_unique_ids_and_valid_slots(converted):
    by_id = {n["id"]: n for n in converted["nodes"]}
    ids = [link[0] for link in converted["links"]]
    assert len(ids) == len(set(ids))
    for link_id, origin_id, origin_slot, target_id, target_slot, _type in \
            converted["links"]:
        origin, target = by_id[origin_id], by_id[target_id]
        assert 0 <= origin_slot < len(origin["outputs"])
        assert 0 <= target_slot < len(target["inputs"])
        assert link_id in origin["outputs"][origin_slot]["links"]
        assert target["inputs"][target_slot]["link"] == link_id


def test_link_targets_the_correct_named_input(converted):
    by_id = {n["id"]: n for n in converted["nodes"]}
    # CLIPTextEncode.clip comes from CheckpointLoaderSimple output slot 1 (CLIP)
    link = next(l for l in converted["links"] if l[3] == 2)
    assert link[1] == 1 and link[2] == 1
    assert by_id[2]["inputs"][link[4]]["name"] == "clip"
    assert link[5] == "CLIP"


def test_unconnected_optional_link_input_still_gets_a_slot(converted):
    create = next(n for n in converted["nodes"] if n["title"] == "CREATE_VIDEO")
    names = {i["name"] for i in create["inputs"]}
    assert names == {"images", "audio"}
    audio = next(i for i in create["inputs"] if i["name"] == "audio")
    assert audio["link"] is None


def test_link_count_matches_source(converted):
    expected = sum(
        1 for node in api_workflow().values()
        for value in node["inputs"].values()
        if isinstance(value, list) and len(value) == 2
    )
    assert len(converted["links"]) == expected


# ----------------------------------------------------------------- layout ---
def test_nodes_are_laid_out_in_a_readable_grid(converted):
    positions = [tuple(n["pos"]) for n in converted["nodes"]]
    assert len(set(positions)) == len(positions)          # nothing stacked
    assert len({p[0] for p in positions}) > 1             # multiple columns
    # a node sits strictly right of its upstream dependency
    by_id = {n["id"]: n for n in converted["nodes"]}
    for _lid, origin_id, _os, target_id, _ts, _t in converted["links"]:
        assert by_id[origin_id]["pos"][0] < by_id[target_id]["pos"][0]


def test_order_field_is_unique_and_sequential(converted):
    orders = sorted(n["order"] for n in converted["nodes"])
    assert orders == list(range(len(converted["nodes"])))


# ------------------------------------------------------------ round-trip ----
def test_verify_conversion_reports_no_problems(converted):
    assert verify_conversion(api_workflow(), converted, OBJECT_INFO) == []


def test_round_trip_preserves_every_scalar_input_value(converted):
    """Every non-link input value in the source lands on the right widget."""
    by_id = {n["id"]: n for n in converted["nodes"]}
    checked = 0
    for nid, node in api_workflow().items():
        plan = widget_plan(OBJECT_INFO[node["class_type"]])
        positions = {name: i for i, (name, kind, _d) in enumerate(plan)
                     if kind == "value"}
        for input_name, value in node["inputs"].items():
            if isinstance(value, list) and len(value) == 2:
                continue
            assert by_id[int(nid)]["widgets_values"][positions[input_name]] == value
            checked += 1
    assert checked > 0


def test_output_is_json_serialisable(converted):
    assert json.loads(json.dumps(converted)) == converted


# ------------------------------------------------------------ error paths ---
def test_unknown_class_raises_rather_than_emitting_a_broken_graph():
    workflow = {"1": {"class_type": "NoSuchNode", "inputs": {}, "_meta": {}}}
    with pytest.raises(ConversionError, match="NoSuchNode"):
        convert_workflow(workflow, OBJECT_INFO)


def test_link_to_missing_node_raises():
    workflow = {
        "1": {"class_type": "CLIPTextEncode",
              "inputs": {"text": "hi", "clip": ["99", 0]}, "_meta": {}},
    }
    with pytest.raises(ConversionError, match="unknown node"):
        convert_workflow(workflow, OBJECT_INFO)


def test_link_to_out_of_range_output_slot_raises():
    workflow = {
        "1": {"class_type": "PrimitiveInt", "inputs": {"value": 1}, "_meta": {}},
        "2": {"class_type": "CLIPTextEncode",
              "inputs": {"text": "hi", "clip": ["1", 5]}, "_meta": {}},
    }
    with pytest.raises(ConversionError, match="output slot"):
        convert_workflow(workflow, OBJECT_INFO)


def test_empty_workflow_raises():
    with pytest.raises(ConversionError):
        convert_workflow({}, OBJECT_INFO)


def test_non_numeric_node_id_raises():
    workflow = {"abc": {"class_type": "PrimitiveInt", "inputs": {"value": 1},
                        "_meta": {}}}
    with pytest.raises(ConversionError, match="numeric"):
        convert_workflow(workflow, OBJECT_INFO)


def test_verify_conversion_detects_a_corrupted_widget_array(converted):
    """The verifier must actually catch the failure mode it guards against."""
    converted["nodes"][0]["widgets_values"].append("bogus")
    problems = verify_conversion(api_workflow(), converted, OBJECT_INFO)
    assert any("widgets_values" in p for p in problems)


def test_verify_conversion_detects_a_lost_title(converted):
    node = next(n for n in converted["nodes"] if n["title"] == "PROMPT")
    node["title"] = "RENAMED"
    problems = verify_conversion(api_workflow(), converted, OBJECT_INFO)
    assert any("PROMPT" in p for p in problems)


def test_verify_conversion_detects_a_dangling_link(converted):
    converted["links"][0][1] = 999
    problems = verify_conversion(api_workflow(), converted, OBJECT_INFO)
    assert any("does not exist" in p for p in problems)
