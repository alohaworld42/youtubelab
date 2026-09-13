"""K1: Keyframes mit FLUX.2 Klein 4B + ReferenceLatent-Kette (1 Ref je anwesendes Kind).

    python klein_agent.py --run proof01 [--shots s01,s02] [--resume] [--size 1344x768] [--dump-workflows]

Ausgabe: runs/<id>/keyframes/K1/<shot>_<W>x<H>.png
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "common"))
import comfy_lab as L  # noqa: E402
import shots as S  # noqa: E402

ENGINE = "K1"


def build_graph(kc, w, h, seed, prompt, staged_refs, prefix):
    k = kc["klein"]
    g = {
        "1": L.title({"class_type": "UNETLoader", "inputs": {"unet_name": k["unet"], "weight_dtype": k["weight_dtype"]}}, "UNET_LOADER"),
        "2": L.title({"class_type": "CLIPLoader", "inputs": {"clip_name": k["clip"], "type": "flux2", "device": "default"}}, "CLIP_LOADER"),
        "3": L.title({"class_type": "VAELoader", "inputs": {"vae_name": k["vae"]}}, "VAE_LOADER"),
        "8": L.title({"class_type": "EmptyFlux2LatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}}, "LATENT"),
        "9": L.title({"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": prompt}}, "PROMPT"),
        "24": L.title({"class_type": "RandomNoise", "inputs": {"noise_seed": int(seed)}}, "SEED"),
        "25": L.title({"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}}, "SAMPLER_SELECT"),
        "26": L.title({"class_type": "Flux2Scheduler", "inputs": {"steps": int(k["steps"]), "width": w, "height": h}}, "SIGMAS"),
        "28": L.title({"class_type": "VAEDecode", "inputs": {"samples": ["27", 0], "vae": ["3", 0]}}, "DECODE"),
        "29": L.title({"class_type": "SaveImage", "inputs": {"images": ["28", 0], "filename_prefix": prefix}}, "FILENAME_PREFIX"),
    }
    cond = ["9", 0]
    for i, name in enumerate(staged_refs):
        li, vi, ri = str(14 + i), str(17 + i), str(20 + i)
        g[li] = L.title({"class_type": "LoadImage", "inputs": {"image": name}}, f"IMAGE{i + 1}")
        g[vi] = L.title({"class_type": "VAEEncode", "inputs": {"pixels": [li, 0], "vae": ["3", 0]}}, f"ENCODE{i + 1}")
        g[ri] = L.title({"class_type": "ReferenceLatent", "inputs": {"conditioning": cond, "latent": [vi, 0]}}, f"REF{i + 1}")
        cond = [ri, 0]
    g["23"] = L.title({"class_type": "BasicGuider", "inputs": {"model": ["1", 0], "conditioning": cond}}, "GUIDER")
    g["27"] = L.title({"class_type": "SamplerCustomAdvanced", "inputs": {
        "noise": ["24", 0], "guider": ["23", 0], "sampler": ["25", 0], "sigmas": ["26", 0], "latent_image": ["8", 0]}}, "SAMPLER")
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--shots", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--size", default=None, help="WxH (default lab.json keyframe)")
    ap.add_argument("--dump-workflows", action="store_true", help="write ref1/2/3 example graphs to workflows/")
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
        wf = os.path.join(HERE, "workflows")
        os.makedirs(wf, exist_ok=True)
        import json
        for n in (1, 2, 3):
            g = build_graph(kc, w, h, 12345, "PROMPT PLACEHOLDER", [f"ref{i + 1}.png" for i in range(n)], "stills/klein")
            with open(os.path.join(wf, f"klein4b_ref{n}.json"), "w", encoding="utf-8") as f:
                json.dump(g, f, indent=2)
        log.info("dumped klein4b_ref1..3.json to %s", wf)

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
