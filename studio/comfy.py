"""
comfy.py — Local AI video generation via a running ComfyUI server.

ComfyUI (https://github.com/comfyanonymous/ComfyUI) runs the actual model
(recommended: LTX-Video / LTX-2, the only text-to-video that fits 8GB VRAM like
an RTX 4060). We keep that heavy stack OUT of this app: the user runs ComfyUI
separately, and we talk to its HTTP API. If it isn't running or isn't enabled,
every call returns None and the asset chain falls through to stock/generated —
nothing breaks.

Flow: load a user-exported API-format workflow JSON, overwrite the positive
prompt, POST /prompt, poll /history/<id>, download the produced clip via /view.
"""
import json
import logging
import os
import time

import requests

log = logging.getLogger("studio.comfy")


def _conf(cfg):
    return cfg.get("comfyui") or {}


def is_enabled(cfg):
    return bool(_conf(cfg).get("enabled"))


def is_available(cfg):
    """True if ComfyUI is enabled AND the server answers."""
    if not is_enabled(cfg):
        return False
    host = _conf(cfg).get("host", "http://127.0.0.1:8188")
    try:
        r = requests.get(f"{host}/system_stats", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def build_prompt(cfg, topic, script):
    template = _conf(cfg).get(
        "prompt_template", "3d cartoon animation, {topic}, colorful, for children"
    )
    t = (topic or "").strip()
    if not t and script:
        t = (script.get("title") or "").strip()
    return template.format(topic=t or "a fun happy scene")


def _find_prompt_node(workflow, title_or_id):
    """Locate the CLIPTextEncode node to inject text into.

    Match by explicit node id, else by _meta.title, else the first CLIPTextEncode.
    """
    if title_or_id and str(title_or_id) in workflow:
        return str(title_or_id)
    for nid, node in workflow.items():
        if node.get("_meta", {}).get("title", "").lower() == str(title_or_id).lower():
            return nid
    for nid, node in workflow.items():
        if node.get("class_type") == "CLIPTextEncode":
            return nid
    return None


def _load_workflow(cfg):
    path = _conf(cfg).get("workflow_path")
    if not path:
        return None
    if not os.path.isabs(path):
        path = os.path.join(cfg["_root"], path)
    if not os.path.exists(path):
        log.warning("ComfyUI workflow not found: %s", path)
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_clip(cfg, topic, script, out_path, seed=None,
                  prompt_text=None, length_frames=None):
    """Generate one clip via ComfyUI. Returns out_path on success, else None.

    prompt_text overrides the template-built prompt (scene engine passes exact
    per-scene prompts); length_frames overrides comfyui.frames for this call.
    Never raises — AI video is a best-effort tier.
    """
    if not is_available(cfg):
        return None
    conf = _conf(cfg)
    host = conf.get("host", "http://127.0.0.1:8188")
    workflow = _load_workflow(cfg)
    if not workflow:
        return None

    node_id = _find_prompt_node(workflow, conf.get("prompt_node_title", "positive"))
    if not node_id:
        log.warning("No CLIPTextEncode node found in workflow — cannot inject prompt")
        return None
    workflow[node_id].setdefault("inputs", {})["text"] = (
        prompt_text or build_prompt(cfg, topic, script)
    )

    # Vary the seed per call so repeated topics don't produce identical clips.
    if seed is None:
        seed = int(time.time() * 1000) % 2_147_483_647
    for node in workflow.values():
        inputs = node.get("inputs", {})
        if "seed" in inputs:
            inputs["seed"] = seed
        elif "noise_seed" in inputs:
            inputs["noise_seed"] = seed

    # Apply resolution/length/fps from config to any node that declares them
    # (EmptyLTXVLatentVideo, LTXVConditioning, VHS_VideoCombine, ...) so a
    # user-swapped workflow still respects the studio's video dimensions.
    dims = {
        "width": conf.get("width", 640),
        "height": conf.get("height", 1088),
        "length": length_frames or conf.get("frames", 97),
    }
    fps = conf.get("frame_rate", 24)
    for node in workflow.values():
        inputs = node.get("inputs", {})
        for key, val in dims.items():
            if key in inputs:
                inputs[key] = val
        if "frame_rate" in inputs:
            inputs["frame_rate"] = fps

    try:
        resp = requests.post(f"{host}/prompt", json={"prompt": workflow}, timeout=30)
        resp.raise_for_status()
        prompt_id = resp.json()["prompt_id"]
    except Exception as e:
        log.warning("ComfyUI submit failed: %s", e)
        return None

    deadline = time.time() + int(conf.get("timeout_seconds", 600))
    outputs = None
    while time.time() < deadline:
        try:
            h = requests.get(f"{host}/history/{prompt_id}", timeout=10).json()
        except Exception:
            time.sleep(2)
            continue
        if prompt_id in h:
            outputs = h[prompt_id].get("outputs", {})
            break
        time.sleep(2)
    if not outputs:
        log.warning("ComfyUI generation timed out for prompt %s", prompt_id)
        return None

    ref = _first_output_file(outputs)
    if not ref:
        log.warning("ComfyUI produced no video/image output")
        return None

    try:
        params = {"filename": ref["filename"], "subfolder": ref.get("subfolder", ""),
                  "type": ref.get("type", "output")}
        data = requests.get(f"{host}/view", params=params, timeout=120)
        data.raise_for_status()
        from pipeline.atomicio import atomic_write_bytes

        atomic_write_bytes(out_path, [data.content])
        return out_path
    except Exception as e:
        log.warning("ComfyUI download failed: %s", e)
        return None


def _first_output_file(outputs):
    """Pull the first produced file, preferring video (gifs/videos) over images."""
    for key in ("gifs", "videos"):
        for node_out in outputs.values():
            items = node_out.get(key)
            if items:
                return items[0]
    for node_out in outputs.values():
        items = node_out.get("images")
        if items:
            return items[0]
    return None
