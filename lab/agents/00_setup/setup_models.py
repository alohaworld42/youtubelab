"""Stage 0: Modelle laden (Resume, nie ueberschreiben), DINOv2 warmup, Graph-Validierung.

    python setup_models.py [--download] [--optional NAME,NAME] [--warmup] [--validate] [--all]

Downloads gehen nach <models_dir>/<dest>/<basename>. Vorhandene Dateien (>= 98 % der
erwarteten Groesse) werden uebersprungen. Groessen werden geloggt.
"""
import argparse
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "common"))
import comfy_lab as L  # noqa: E402


def download(cfg, item, log):
    from huggingface_hub import hf_hub_download
    dest_dir = os.path.join(cfg["models_dir"], item["dest"])
    os.makedirs(dest_dir, exist_ok=True)
    final = os.path.join(dest_dir, os.path.basename(item["file"]))
    expected = float(item.get("gb", 0)) * 1e9
    if os.path.isfile(final) and (expected == 0 or os.path.getsize(final) >= 0.98 * expected):
        log.info("have  %-70s %6.2f GB", os.path.basename(final), os.path.getsize(final) / 1e9)
        return final
    tmp_dir = os.path.join(cfg["temp_dir"], "hf_dl")
    os.makedirs(tmp_dir, exist_ok=True)
    log.info("fetch %s/%s (%.2f GB) ...", item["repo"], item["file"], item.get("gb", 0))
    t0 = time.time()
    got = hf_hub_download(repo_id=item["repo"], filename=item["file"], local_dir=tmp_dir, resume_download=True)
    if os.path.isfile(final):
        log.warning("target appeared meanwhile, keeping existing %s", final)
        return final
    shutil.move(got, final)
    log.info("done  %-70s %6.2f GB in %.0fs", os.path.basename(final), os.path.getsize(final) / 1e9, time.time() - t0)
    return final


def warmup_dino(cfg, log):
    os.environ["HF_HOME"] = cfg["hf_home"]
    from transformers import AutoImageProcessor, AutoModel
    t0 = time.time()
    AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    AutoModel.from_pretrained("facebook/dinov2-small")
    log.info("dinov2-small ready (%.0fs)", time.time() - t0)


def validate(cfg, log):
    """Check node classes of all engines against the running server's /object_info."""
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "01_keyframe"))
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "02_video"))
    import klein_agent, qwen_edit_agent, h3_r2v_agent  # noqa: E402
    client = L.make_client(cfg, 60)
    client.ensure_up()
    info = client._fetch_object_info()
    graphs = {
        "klein": klein_agent.build_graph(cfg["keyframe"], 1344, 768, 1, "p", ["a.png", "b.png", "c.png"], "x"),
        "qwen": qwen_edit_agent.build_graph(cfg["keyframe"], 1344, 768, 1, "p", ["a.png", "b.png", "c.png"], "x"),
        "h3": h3_r2v_agent.build_graph(cfg["h3"], 1344, 768, 90, 1, "p", ["a.png", "b.png", "c.png"], "x"),
    }
    ok = True
    for name, g in graphs.items():
        missing = sorted({n["class_type"] for n in g.values() if n["class_type"] not in info})
        bad_inputs = []
        for nid, n in g.items():
            spec = info.get(n["class_type"], {}).get("input", {})
            allowed = set(spec.get("required", {})) | set(spec.get("optional", {}))
            for k in n["inputs"]:
                base = k.split(".")[0]
                if k not in allowed and base not in allowed:
                    bad_inputs.append(f"{n['class_type']}.{k}")
        # model file presence
        files = []
        for n in g.values():
            for k, v in n["inputs"].items():
                if k in ("unet_name", "clip_name", "vae_name", "lora_name") and isinstance(v, str):
                    files.append((k, v))
        missing_files = []
        for k, v in files:
            sub = {"unet_name": ["unet", "diffusion_models"], "clip_name": ["text_encoders", "clip"],
                   "vae_name": ["vae"], "lora_name": ["loras"]}[k]
            if not any(os.path.isfile(os.path.join(cfg["models_dir"], s, v)) for s in sub):
                missing_files.append(v)
        status = "OK" if not (missing or bad_inputs or missing_files) else "FAIL"
        ok = ok and status == "OK"
        log.info("validate %-6s %s  missing_nodes=%s bad_inputs=%s missing_files=%s", name, status, missing, bad_inputs, missing_files)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--optional", default="", help="comma list of substrings of optional download files")
    ap.add_argument("--warmup", action="store_true")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    cfg = L.lab_config()
    log = L.setup_logging(os.path.join(cfg["lab_dir"], "logs"), "setup")
    rc = 0
    if a.download or a.all:
        for item in cfg["downloads"]:
            download(cfg, item, log)
    if a.optional:
        for item in cfg["downloads_optional"]:
            if any(s and s.lower() in item["file"].lower() for s in a.optional.split(",")):
                download(cfg, item, log)
    if a.warmup or a.all:
        warmup_dino(cfg, log)
    if a.validate or a.all:
        if not validate(cfg, log):
            rc = 2
    sys.exit(rc)


if __name__ == "__main__":
    main()
