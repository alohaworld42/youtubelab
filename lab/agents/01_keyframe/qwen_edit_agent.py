"""K2: Keyframes mit Qwen-Image-Edit-2511 (GGUF) — image1..3 = Cast-Refs, neue Szene bei WxH.

    python qwen_edit_agent.py --run proof01 [--shots s01,s02] [--resume] [--size 1344x768] [--dump-workflows]

Ausgabe: runs/<id>/keyframes/K2/<shot>_<W>x<H>.png
Graph = offizielles ComfyUI-Template "Image Edit (Qwen-Image 2511)" mit
UnetLoaderGGUF statt fp8 und EmptySD3LatentImage (neue Leinwand) statt VAEEncode(image1).
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "common"))
import comfy_lab as L  # noqa: E402
import shots as S  # noqa: E402

ENGINE = "K2"


def build_graph(kc, w, h, seed, prompt, staged_refs, prefix):
    q = kc["qwen"]
    g = {
        "1": L.title({"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": q["unet_gguf"]}}, "UNET_LOADER"),
        "2": L.title({"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["1", 0], "shift": float(q["shift"])}}, "MODEL_SAMPLING"),
        "3": L.title({"class_type": "CFGNorm", "inputs": {"model": ["2", 0], "strength": 1.0}}, "CFG_NORM"),
        "4": L.title({"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["3", 0], "lora_name": q["lora"], "strength_model": 1.0}}, "LORA_LIGHTNING"),
        "5": L.title({"class_type": "CLIPLoader", "inputs": {"clip_name": q["clip"], "type": "qwen_image", "device": "default"}}, "CLIP_LOADER"),
        "6": L.title({"class_type": "VAELoader", "inputs": {"vae_name": q["vae"]}}, "VAE_LOADER"),
        "14": L.title({"class_type": "EmptySD3LatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}}, "LATENT"),
        "16": L.title({"class_type": "VAEDecode", "inputs": {"samples": ["15", 0], "vae": ["6", 0]}}, "DECODE"),
        "17": L.title({"class_type": "SaveImage", "inputs": {"images": ["16", 0], "filename_prefix": prefix}}, "FILENAME_PREFIX"),
    }
    imgs = {}
    for i, name in enumerate(staged_refs):
        li = str(7 + i)
        g[li] = L.title({"class_type": "LoadImage", "inputs": {"image": name}}, f"IMAGE{i + 1}")
        imgs[f"image{i + 1}"] = [li, 0]
    g["10"] = L.title({"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"clip": ["5", 0], "prompt": prompt, "vae": ["6", 0], **imgs}}, "PROMPT")
    g["11"] = L.title({"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"clip": ["5", 0], "prompt": "", "vae": ["6", 0], **imgs}}, "NEGATIVE")
    g["12"] = L.title({"class_type": "FluxKontextMultiReferenceLatentMethod", "inputs": {"conditioning": ["10", 0], "reference_latents_method": "index_timestep_zero"}}, "REF_METHOD_POS")
    g["13"] = L.title({"class_type": "FluxKontextMultiReferenceLatentMethod", "inputs": {"conditioning": ["11", 0], "reference_latents_method": "index_timestep_zero"}}, "REF_METHOD_NEG")
    g["15"] = L.title({"class_type": "KSampler", "inputs": {
        "model": ["4", 0], "seed": int(seed), "steps": int(q["steps"]), "cfg": float(q["cfg"]),
        "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0,
        "positive": ["12", 0], "negative": ["13", 0], "latent_image": ["14", 0]}}, "SAMPLER")
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--shots", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--size", default=None)
    ap.add_argument("--dump-workflows", action="store_true")
    a = ap.parse_args()

    cfg = L.lab_config()
    kc = cfg["keyframe"]
    w, h = (map(int, a.size.lower().split("x")) if a.size else (kc["width"], kc["height"]))
    paths = L.run_paths(cfg, a.run)
    log = L.setup_logging(paths["root"], ENGINE)
    cast, by_id = S.load_cast(cfg["lab_dir"])
    S.write_default_shots(paths["shots"])
    all_shots = S.load_shots(paths["shots"])
    only = a.shots.split(",") if a.shots else None
    todo = [s for s in all_shots if ENGINE in s.get("engines", []) and (not only or s["id"] in only)]
    out_dir = os.path.join(paths["keyframes"], ENGINE)
    os.makedirs(out_dir, exist_ok=True)

    if a.dump_workflows:
        import json
        wf = os.path.join(HERE, "workflows")
        os.makedirs(wf, exist_ok=True)
        for n in (1, 2, 3):
            g = build_graph(kc, w, h, 12345, "PROMPT PLACEHOLDER", [f"ref{i + 1}.png" for i in range(n)], "stills/qwen")
            with open(os.path.join(wf, f"qwen_edit_2511_ref{n}.json"), "w", encoding="utf-8") as f:
                json.dump(g, f, indent=2)
        log.info("dumped qwen_edit_2511_ref1..3.json to %s", wf)

    client = L.make_client(cfg, kc["render_timeout"])
    client._graph_dump_dir = paths["graphs"]
    client.ensure_up()
    guard = L.gpu_guard(cfg, log)
    guard.__enter__()
    try:
        for shot in todo:
            out = os.path.join(out_dir, f"{shot['id']}_{w}x{h}.png")
            if a.resume and L.done(out):
                log.info("%s: exists, skip", shot["id"])
                continue
            seed = S.shot_seed(shot, cast, by_id, all_shots)
            prompt = S.keyframe_prompt(shot, by_id)
            staged = [L.stage_ref(client, os.path.join(cfg["lab_dir"], by_id[c]["ref"])) for c in shot["cast"]]
            g = build_graph(kc, w, h, seed, prompt, staged, f"lab/{a.run}/{ENGINE}_{shot['id']}")
            log.info("%s: rendering %dx%d seed=%d refs=%d", shot["id"], w, h, seed, len(staged))
            path, secs = client.render_graph(g, out, dump_name=f"{ENGINE}_{shot['id']}")
            log.info("%s: done in %.0fs -> %s", shot["id"], secs, path)
            L.mark_state(paths, engine=ENGINE, shot=shot["id"], seed=seed, seconds=round(secs), file=path, prompt=prompt)
            client.cleanup_staged()
    finally:
        client.free()
        guard.__exit__(None, None, None)


if __name__ == "__main__":
    main()
