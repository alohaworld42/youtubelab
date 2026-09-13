"""V2 phased: verlustfreier Episoden-Batch fuer MiniMax-H3 R2V.

Statt pro Shot alles neu zu laden (--cache-none laedt Encoder+DiT je Clip neu,
~35% der Zeit), wird die Episode in 3 Phasen gerendert, jede als EIN Graph mit
allen Shots als Zweige, sodass das teure Modell je Phase nur EINMAL laedt:

  encode : CLIP + VAEs einmal -> je Shot MiniMaxH3ReferenceToVideo -> H3SaveConditioning
  sample : DiT einmal        -> je Shot H3LoadConditioning + EmptyLatent -> Sampler -> H3SaveLatentAV
  decode : VAEs einmal       -> je Shot H3LoadLatentAV -> VAEDecode(+Audio) -> CreateVideo -> SaveVideo

Bit-identisch zum Einzel-Render (H3SaveConditioning speichert die exakte
Conditioning; das leere AV-Latent ist deterministisch). Cache liegt in
ComfyUI/output/h3_cache/. Ausgabe wie gehabt: runs/<id>/clips_V2/<shot>_V2.mp4

    python h3_phased_agent.py --run proof01 [--shots s01,s02,s03] [--phase all|encode|sample|decode] [--resume]
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "common"))
import comfy_lab as L  # noqa: E402
import shots as S  # noqa: E402

ENGINE = "V2"


def cache_key(run, shot_id):
    return f"{run}_{shot_id}"


def build_encode_graph(hc, run, todo, cfg, by_id, client):
    """One graph: shared CLIP+VAEs, one encode->save branch per shot."""
    g = {
        "2": L.title({"class_type": "CLIPLoaderGGUF", "inputs": {"clip_name": hc["clip_gguf"], "type": "minimax"}}, "TEXT_ENCODER"),
        "3": L.title({"class_type": "VAELoader", "inputs": {"vae_name": hc["video_vae"]}}, "VIDEO_VAE"),
        "4": L.title({"class_type": "VAELoader", "inputs": {"vae_name": hc["audio_vae"]}}, "AUDIO_VAE"),
    }
    nid = 100
    for shot in todo:
        with_refs = not shot.get("norefs")
        prompt = S.h3_prompt(shot, by_id, with_refs=with_refs)
        pnode, nnode = str(nid), str(nid + 1)
        g[pnode] = L.title({"class_type": "PrimitiveStringMultiline", "inputs": {"value": prompt}}, f"PROMPT_{shot['id']}")
        g[nnode] = L.title({"class_type": "PrimitiveStringMultiline", "inputs": {"value": S.NEGATIVE_SHORT}}, f"NEG_{shot['id']}")
        full = str(nid + 2)
        g[full] = L.title({"class_type": "StringConcatenate", "inputs": {"string_a": [pnode, 0], "string_b": [nnode, 0], "delimiter": "\n\nDo not show: "}}, f"FULL_{shot['id']}")
        cond = {"clip": ["2", 0], "vae": ["3", 0], "audio_vae": ["4", 0], "prompt": [full, 0],
                "width": hc["width"], "height": hc["height"], "length": hc["frames"], "ref_image_size": hc["ref_image_size"]}
        base = nid + 3
        if with_refs:
            for i, c in enumerate(shot["cast"]):
                li = str(base + i)
                staged = L.stage_ref(client, os.path.join(cfg["lab_dir"], by_id[c]["ref_hi"]))
                g[li] = L.title({"class_type": "LoadImage", "inputs": {"image": staged}}, f"REF_{shot['id']}_{i}")
                cond[f"ref_images.ref_image_{i}"] = [li, 0]
        r2v = str(base + 5)
        g[r2v] = L.title({"class_type": "MiniMaxH3ReferenceToVideo", "inputs": cond}, f"R2V_{shot['id']}")
        sav = str(base + 6)
        g[sav] = L.title({"class_type": "H3SaveConditioning", "inputs": {"conditioning": [r2v, 0], "filename": cache_key(run, shot["id"])}}, f"SAVE_{shot['id']}")
        nid += 20
    return g


def build_sample_graph(hc, run, todo):
    g = {
        "1": L.title({"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": hc["unet_gguf"]}}, "DIFFUSION_MODEL"),
        "5": L.title({"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": hc["lora"], "strength_model": 1.0}}, "LORA_TURBO"),
        "16": L.title({"class_type": "PrimitiveInt", "inputs": {"value": int(hc["steps"])}}, "STEPS"),
    }
    nid = 100
    for shot in todo:
        seed = shot["_seed"]
        lc = str(nid)
        g[lc] = L.title({"class_type": "H3LoadConditioning", "inputs": {"file_name": cache_key(run, shot["id"]) + ".cond.pt"}}, f"LOADC_{shot['id']}")
        el = str(nid + 1)
        g[el] = L.title({"class_type": "EmptyMiniMaxH3LatentAV", "inputs": {"width": hc["width"], "height": hc["height"], "length": hc["frames"]}}, f"EMPTY_{shot['id']}")
        rn = str(nid + 2)
        g[rn] = L.title({"class_type": "RandomNoise", "inputs": {"noise_seed": int(seed)}}, f"SEED_{shot['id']}")
        ss = str(nid + 3)
        g[ss] = L.title({"class_type": "KSamplerSelect", "inputs": {"sampler_name": hc["sampler"]}}, f"SS_{shot['id']}")
        sch = str(nid + 4)
        g[sch] = L.title({"class_type": "BasicScheduler", "inputs": {"model": ["5", 0], "scheduler": hc["scheduler"], "steps": ["16", 0], "denoise": 1.0}}, f"SCH_{shot['id']}")
        gd = str(nid + 5)
        g[gd] = L.title({"class_type": "BasicGuider", "inputs": {"model": ["5", 0], "conditioning": [lc, 0]}}, f"GUIDE_{shot['id']}")
        sm = str(nid + 6)
        g[sm] = L.title({"class_type": "SamplerCustomAdvanced", "inputs": {"noise": [rn, 0], "guider": [gd, 0], "sampler": [ss, 0], "sigmas": [sch, 0], "latent_image": [el, 0]}}, f"SAMP_{shot['id']}")
        sv = str(nid + 7)
        g[sv] = L.title({"class_type": "H3SaveLatentAV", "inputs": {"samples": [sm, 0], "filename": cache_key(run, shot["id"])}}, f"SAVEL_{shot['id']}")
        nid += 20
    return g


def build_decode_graph(hc, run, todo):
    g = {
        "3": L.title({"class_type": "VAELoader", "inputs": {"vae_name": hc["video_vae"]}}, "VIDEO_VAE"),
        "4": L.title({"class_type": "VAELoader", "inputs": {"vae_name": hc["audio_vae"]}}, "AUDIO_VAE"),
    }
    nid = 100
    for shot in todo:
        ll = str(nid)
        g[ll] = L.title({"class_type": "H3LoadLatentAV", "inputs": {"file_name": cache_key(run, shot["id"]) + ".lat.pt"}}, f"LOADL_{shot['id']}")
        vd = str(nid + 1)
        g[vd] = L.title({"class_type": "VAEDecode", "inputs": {"samples": [ll, 0], "vae": ["3", 0]}}, f"VD_{shot['id']}")
        va = str(nid + 2)
        g[va] = L.title({"class_type": "VAEDecodeAudio", "inputs": {"samples": [ll, 0], "vae": ["4", 0]}}, f"VA_{shot['id']}")
        cv = str(nid + 3)
        g[cv] = L.title({"class_type": "CreateVideo", "inputs": {"images": [vd, 0], "fps": 24, "audio": [va, 0]}}, f"CV_{shot['id']}")
        sv = str(nid + 4)
        g[sv] = L.title({"class_type": "SaveVideo", "inputs": {"video": [cv, 0], "filename_prefix": f"lab/{run}/phased_{shot['id']}", "format": "auto", "codec": "auto"}}, f"SAVE_{shot['id']}")
        nid += 20
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--shots", default=None)
    ap.add_argument("--phase", default="all", choices=["all", "encode", "sample", "decode"])
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    cfg = L.lab_config()
    hc = cfg["h3"]
    paths = L.run_paths(cfg, a.run)
    log = L.setup_logging(paths["root"], "PHASED")
    cast, by_id = S.load_cast(cfg["lab_dir"])
    S.write_default_shots(paths["shots"])
    all_shots = S.load_shots(paths["shots"])
    only = a.shots.split(",") if a.shots else None
    todo = [s for s in all_shots if ENGINE in s.get("engines", []) and (not only or s["id"] in only)]
    for s in todo:
        s["_seed"] = S.shot_seed(s, cast, by_id, all_shots)
    cache_dir = os.path.join(cfg["comfy_dir"], "output", "h3_cache")

    client = L.make_client(cfg, hc["render_timeout"])
    client._graph_dump_dir = paths["graphs"]
    client.ensure_up()
    guard = L.gpu_guard(cfg, log)
    guard.__enter__()
    import time
    try:
        phases = ["encode", "sample", "decode"] if a.phase == "all" else [a.phase]
        for ph in phases:
            if ph == "encode":
                pend = [s for s in todo if not (a.resume and L.done(os.path.join(cache_dir, cache_key(a.run, s["id"]) + ".cond.pt")))]
                if not pend:
                    log.info("encode: all cached, skip"); continue
                g = build_encode_graph(hc, a.run, pend, cfg, by_id, client)
                log.info("encode: %d shots in one graph (CLIP loads once)", len(pend))
                t0 = time.time()
                client.render_headless(g, dump_name="phase_encode")
                log.info("encode: done in %.0fs for %d shots (%.0fs/shot)", time.time() - t0, len(pend), (time.time() - t0) / len(pend))
                client.cleanup_staged()
                client.free()
            elif ph == "sample":
                pend = [s for s in todo if not (a.resume and L.done(os.path.join(cache_dir, cache_key(a.run, s["id"]) + ".lat.pt")))]
                if not pend:
                    log.info("sample: all cached, skip"); continue
                g = build_sample_graph(hc, a.run, pend)
                log.info("sample: %d shots in one graph (DiT loads once)", len(pend))
                t0 = time.time()
                client.render_headless(g, dump_name="phase_sample")
                log.info("sample: done in %.0fs for %d shots (%.0fs/shot)", time.time() - t0, len(pend), (time.time() - t0) / len(pend))
                client.free()
            elif ph == "decode":
                pend = [s for s in todo if not (a.resume and L.done(os.path.join(paths["clips"], f"{s['id']}_V2.mp4")))]
                if not pend:
                    log.info("decode: all done, skip"); continue
                g = build_decode_graph(hc, a.run, pend)
                log.info("decode: %d shots", len(pend))
                t0 = time.time()
                # decode graph saves via SaveVideo to ComfyUI/output/lab/<run>/; then copy into runs/clips
                client.render_headless(g, dump_name="phase_decode")
                log.info("decode: done in %.0fs", time.time() - t0)
                # retrieve each SaveVideo output
                _collect_decoded(cfg, paths, a.run, pend, log)
                client.free()
    finally:
        client.free()
        guard.__exit__(None, None, None)


def _collect_decoded(cfg, paths, run, todo, log):
    """SaveVideo wrote to ComfyUI/output/lab/<run>/phased_<shot>_NNNNN.mp4; copy to runs/clips_V2/<shot>_V2.mp4."""
    import glob
    import shutil
    src_dir = os.path.join(cfg["comfy_dir"], "output", "lab", run)
    for shot in todo:
        matches = sorted(glob.glob(os.path.join(src_dir, f"phased_{shot['id']}_*.mp4")))
        if not matches:
            log.warning("decode: no output for %s in %s", shot["id"], src_dir); continue
        dst = os.path.join(paths["clips"], f"{shot['id']}_V2.mp4")
        shutil.copyfile(matches[-1], dst)
        L.mark_state(paths, engine="V2phased", shot=shot["id"], file=dst)
        log.info("decode: %s -> %s", os.path.basename(matches[-1]), dst)


if __name__ == "__main__":
    main()
