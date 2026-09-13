# -*- coding: utf-8 -*-
"""H3-Ref2VA-Modelle laden (funktionierender Satz, portabel).

Nur der verifizierte Satz wird geladen:
  - GGUF:  Abiray/MiniMax-H3-Pruned-GGUF  MiniMax-H3-Ref2VA-Pruned-Q4_K_M.gguf
  - TE:    Comfy-Org/MiniMax-H3           qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors (safetensors, KEIN gguf)
  - VAEs:  Comfy-Org/MiniMax-H3           minimax_h3_video_vae_fp16 / minimax_h3_audio_vae_fp32
  - LoRA:  Comfy-Org/MiniMax-H3           minimax_h3_ref2v_turbo_4step_v0.1 (optional)

Aufruf:
  python dl_h3.py --models-dir <ComfyUI>/models
"""
import argparse
import os
import shutil
import sys
import tempfile
import time
from huggingface_hub import hf_hub_download

DEFAULT_COMFY = r"C:\Users\admin\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI"

JOBS = [
    ("Abiray/MiniMax-H3-Pruned-GGUF",
     "MiniMax-H3-Ref2VA-Pruned-Q4_K_M.gguf",
     "diffusion_models"),
    ("Comfy-Org/MiniMax-H3",
     "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
     "text_encoders"),
    ("Comfy-Org/MiniMax-H3",
     "vae/minimax_h3_video_vae_fp16.safetensors",
     "vae"),
    ("Comfy-Org/MiniMax-H3",
     "vae/minimax_h3_audio_vae_fp32.safetensors",
     "vae"),
    ("Comfy-Org/MiniMax-H3",
     "loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
     "loras"),
]


def log(msg):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


def fetch_flat(repo_id, filename, dest_dir):
    """Laedt eine Datei flach nach dest_dir (loest Unterordner auf)."""
    os.makedirs(dest_dir, exist_ok=True)
    target = os.path.join(dest_dir, os.path.basename(filename))
    if os.path.exists(target) and os.path.getsize(target) > 0:
        log("EXISTS %s" % os.path.basename(target))
        return True
    staging = tempfile.mkdtemp(prefix="h3dl_")
    try:
        remote = hf_hub_download(repo_id=repo_id, filename=filename,
                                 local_dir=staging, revision="main")
        shutil.move(remote, target)
        log("OK %s -> %s (%.1f GB)" % (os.path.basename(target),
                                        target, os.path.getsize(target) / 1e9))
        return True
    except Exception as e:
        log("FEHLER %s: %s" % (os.path.basename(filename), e))
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default="",
                    help="ComfyUI models-Verzeichnis (Default: Comfy-Desktop-Standard)")
    args = ap.parse_args()
    models = args.models_dir
    if not models:
        models = os.path.join(DEFAULT_COMFY, "models")
    if not os.path.isdir(models):
        print("FEHLER: models-Verzeichnis nicht gefunden: %s" % models)
        print("       --models-dir <ComfyUI>/models angeben.")
        sys.exit(1)

    rc = 0
    for repo, fname, sub in JOBS:
        dest = os.path.join(models, sub)
        if not fetch_flat(repo, fname, dest):
            rc = 2
    log("ENDE rc=%d (%s)" % (rc, models))
    sys.exit(rc)


if __name__ == "__main__":
    main()