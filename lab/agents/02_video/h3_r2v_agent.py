"""V2: MiniMax-H3 Reference-to-Video mit korrekt verdrahteten Refs (ref_images.ref_image_N).

    python h3_r2v_agent.py --run proof01 [--shots s01,s02] [--resume] [--probe] [--guide K1|K2] [--dump-workflows]

--probe   : kleine Probe (lab.json h3.probe: 864x480x39) fuer s01 MIT und OHNE Refs -> clips_V2/probe_*.mp4
--guide E : zusaetzlich Gewinner-Keyframe der Engine E als Frame-0-Guide (MiniMaxH3AddGuide)
Ausgabe: runs/<id>/clips_V2/<shot>_V2.mp4 (bzw. <shot>_V2guide.mp4)
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "common"))
import comfy_lab as L  # noqa: E402
import shots as S  # noqa: E402

ENGINE = "V2"


def build_graph(hc, w, h, frames, seed, prompt, staged_refs, prefix, negative=S.NEGATIVE_SHORT, guide_image=None):
    g = {
        "1": L.title({"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": hc["unet_gguf"]}}, "DIFFUSION_MODEL"),
        "2": L.title({"class_type": "CLIPLoaderGGUF", "inputs": {"clip_name": hc["clip_gguf"], "type": "minimax"}}, "TEXT_ENCODER"),
        "3": L.title({"class_type": "VAELoader", "inputs": {"vae_name": hc["video_vae"]}}, "VIDEO_VAE"),
        "4": L.title({"class_type": "VAELoader", "inputs": {"vae_name": hc["audio_vae"]}}, "AUDIO_VAE"),
        "5": L.title({"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": hc["lora"], "strength_model": 1.0}}, "LORA_TURBO"),
        "7": L.title({"class_type": "PrimitiveStringMultiline", "inputs": {"value": prompt}}, "PROMPT"),
        "8": L.title({"class_type": "PrimitiveStringMultiline", "inputs": {"value": negative}}, "NEGATIVE"),
        "9": L.title({"class_type": "StringConcatenate", "inputs": {"string_a": ["7", 0], "string_b": ["8", 0], "delimiter": "\n\nDo not show: "}}, "PROMPT_FULL"),
        "10": L.title({"class_type": "PrimitiveInt", "inputs": {"value": int(w)}}, "WIDTH"),
        "11": L.title({"class_type": "PrimitiveInt", "inputs": {"value": int(h)}}, "HEIGHT"),
        "12": L.title({"class_type": "PrimitiveInt", "inputs": {"value": int(frames)}}, "FRAMES"),
        "14": L.title({"class_type": "RandomNoise", "inputs": {"noise_seed": int(seed)}}, "SEED"),
        "15": L.title({"class_type": "KSamplerSelect", "inputs": {"sampler_name": hc["sampler"]}}, "SAMPLER_SELECT"),
        "16": L.title({"class_type": "PrimitiveInt", "inputs": {"value": int(hc["steps"])}}, "STEPS"),
        "17": L.title({"class_type": "BasicScheduler", "inputs": {"model": ["5", 0], "scheduler": hc["scheduler"], "steps": ["16", 0], "denoise": 1.0}}, "SCHEDULER"),
        "20": L.title({"class_type": "VAEDecode", "inputs": {"samples": ["19", 0], "vae": ["3", 0]}}, "DECODE"),
        "21": L.title({"class_type": "VAEDecodeAudio", "inputs": {"samples": ["19", 0], "vae": ["4", 0]}}, "DECODE_AUDIO"),
        "22": L.title({"class_type": "CreateVideo", "inputs": {"images": ["20", 0], "fps": 24, "audio": ["21", 0]}}, "CREATE_VIDEO"),
        "23": L.title({"class_type": "SaveVideo", "inputs": {"video": ["22", 0], "filename_prefix": prefix, "format": "auto", "codec": "auto"}}, "FILENAME_PREFIX"),
    }
    cond = {
        "clip": ["2", 0], "vae": ["3", 0], "audio_vae": ["4", 0], "prompt": ["9", 0],
        "width": ["10", 0], "height": ["11", 0], "length": ["12", 0],
        "ref_image_size": hc["ref_image_size"],
    }
    for i, name in enumerate(staged_refs):
        li = str(24 + i)
        g[li] = L.title({"class_type": "LoadImage", "inputs": {"image": name}}, f"REF{i + 1}")
        cond[f"ref_images.ref_image_{i}"] = [li, 0]   # Autogrow: gepunktete Keys, NICHT Liste
    g["13"] = L.title({"class_type": "MiniMaxH3ReferenceToVideo", "inputs": cond}, "H3_CONDITION")
    positive, latent = ["13", 0], ["13", 1]
    if guide_image:
        g["30"] = L.title({"class_type": "LoadImage", "inputs": {"image": guide_image}}, "GUIDE_IMAGE")
        g["31"] = L.title({"class_type": "MiniMaxH3AddGuide", "inputs": {
            "positive": positive, "latent": latent, "vae": ["3", 0], "image": ["30", 0], "frame_idx": 0}}, "GUIDE")
        positive = ["31", 0]   # AddGuide liefert nur positive; latent bleibt vom R2V-Node
    g["18"] = L.title({"class_type": "BasicGuider", "inputs": {"model": ["5", 0], "conditioning": positive}}, "GUIDER")
    g["19"] = L.title({"class_type": "SamplerCustomAdvanced", "inputs": {
        "noise": ["14", 0], "guider": ["18", 0], "sampler": ["15", 0], "sigmas": ["17", 0], "latent_image": latent}}, "SAMPLER")
    return g


def pick_keyframe(paths, engine, shot_id):
    d = os.path.join(paths["keyframes"], engine)
    if not os.path.isdir(d):
        return None
    for f in sorted(os.listdir(d)):
        if f.startswith(shot_id + "_") and f.endswith(".png"):
            return os.path.join(d, f)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--shots", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--guide", default=None, help="keyframe engine (K1/K2) whose keyframe anchors frame 0")
    ap.add_argument("--dump-workflows", action="store_true")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--size", default=None, help="WxH override")
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--lora", default=None, help="turbo lora filename override")
    ap.add_argument("--tag", default=None, help="output filename tag override")
    a = ap.parse_args()

    cfg = L.lab_config()
    hc = cfg["h3"]
    if a.steps: hc = dict(hc, steps=a.steps)
    if a.lora: hc = dict(hc, lora=a.lora)
    paths = L.run_paths(cfg, a.run)
    log = L.setup_logging(paths["root"], ENGINE)
    cast, by_id = S.load_cast(cfg["lab_dir"])
    S.write_default_shots(paths["shots"])
    all_shots = S.load_shots(paths["shots"])
    only = a.shots.split(",") if a.shots else None

    if a.dump_workflows:
        import json
        wf = os.path.join(HERE, "workflows")
        os.makedirs(wf, exist_ok=True)
        for n in (1, 2, 3):
            g = build_graph(hc, hc["width"], hc["height"], hc["frames"], 12345, "PROMPT PLACEHOLDER",
                            [f"ref{i + 1}.png" for i in range(n)], "kidsong/h3_r2v")
            with open(os.path.join(wf, f"h3_r2v_ref{n}.json"), "w", encoding="utf-8") as f:
                json.dump(g, f, indent=2)
        log.info("dumped h3_r2v_ref1..3.json to %s", wf)

    if a.probe:
        p = hc["probe"]
        w, h, frames = p["width"], p["height"], p["frames"]
        base = next(s for s in all_shots if s["id"] == (only[0] if only else "s01"))
        todo = [dict(base, id=base["id"] + "_probe_refs"), dict(base, id=base["id"] + "_probe_norefs", norefs=True)]
        tag = "probe" + ("_guide" if a.guide else "")
    else:
        w, h, frames = hc["width"], hc["height"], hc["frames"]
        if a.size: w, h = map(int, a.size.lower().split("x"))
        if a.frames: frames = a.frames
        todo = [s for s in all_shots if ENGINE in s.get("engines", []) and (not only or s["id"] in only)]
        tag = a.tag or (ENGINE + ("guide" if a.guide else ""))

    client = L.make_client(cfg, hc["render_timeout"])
    client._graph_dump_dir = paths["graphs"]
    client.ensure_up()
    guard = L.gpu_guard(cfg, log)
    guard.__enter__()
    try:
        for shot in todo:
            out = os.path.join(paths["clips"], f"{shot['id']}_{tag}.mp4")
            if a.resume and L.done(out):
                log.info("%s: exists, skip", shot["id"])
                continue
            seed = S.shot_seed(dict(shot, id=shot["id"].replace("_probe_refs", "").replace("_probe_norefs", "")), cast, by_id, all_shots)
            with_refs = not shot.get("norefs")
            prompt = S.h3_prompt(shot, by_id, with_refs=with_refs)
            staged = [L.stage_ref(client, os.path.join(cfg["lab_dir"], by_id[c]["ref_hi"])) for c in shot["cast"]] if with_refs else []
            guide = None
            if a.guide:
                base_id = shot["id"].replace("_probe_refs", "").replace("_probe_norefs", "").replace("_norefs", "")
                kf = pick_keyframe(paths, a.guide, base_id)
                if kf:
                    guide = L.stage_ref(client, kf)
                else:
                    log.warning("%s: no %s keyframe found, rendering without guide", shot["id"], a.guide)
            g = build_graph(hc, w, h, frames, seed, prompt, staged, f"lab/{a.run}/{tag}_{shot['id']}", guide_image=guide)
            log.info("%s: rendering %dx%dx%d seed=%d refs=%d guide=%s", shot["id"], w, h, frames, seed, len(staged), bool(guide))
            path, secs = client.render_graph(g, out, dump_name=f"{tag}_{shot['id']}")
            log.info("%s: done in %.0fs -> %s", shot["id"], secs, path)
            L.mark_state(paths, engine=tag, shot=shot["id"], seed=seed, seconds=round(secs), file=path,
                         refs=len(staged), guide=bool(guide), size=f"{w}x{h}x{frames}", prompt=prompt)
            client.cleanup_staged()
    finally:
        client.free()
        guard.__exit__(None, None, None)


if __name__ == "__main__":
    main()
