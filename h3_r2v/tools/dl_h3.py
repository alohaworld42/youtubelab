# -*- coding: utf-8 -*-
"""H3-Modelle laden (Ref2VA GGUF + TE GGUF + H3 VAEs + Turbo-LoRA)."""
import os
import sys
import time
from huggingface_hub import snapshot_download

MODELS = r"C:\Users\admin\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\models"

JOBS = [
    ("unsloth/MiniMax-H3-GGUF", ["minimax_h3_ref2va_pruned-Q4_K.gguf"],
     os.path.join(MODELS, "diffusion_models")),
    ("unsloth/MiniMax-H3-GGUF", ["qwen3vl_32b_minimax_h3-Q2_K_M.gguf"],
     os.path.join(MODELS, "text_encoders")),
    ("Comfy-Org/MiniMax-H3", ["vae/minimax_h3_video_vae_fp16.safetensors",
                               "vae/minimax_h3_audio_vae_fp32.safetensors"],
     os.path.join(MODELS, "vae")),
    ("Comfy-Org/MiniMax-H3", ["loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"],
     os.path.join(MODELS, "loras")),
]


def log(msg):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


def main():
    rc = 0
    for repo, allow, dest in JOBS:
        os.makedirs(dest, exist_ok=True)
        log("START %s -> %s (%s)" % (repo, os.path.basename(dest), ", ".join(allow)))
        try:
            snapshot_download(
                repo_id=repo,
                allow_patterns=allow,
                local_dir=dest,
                local_dir_use_symlinks=False,
                max_workers=1,
            )
            log("OK %s" % repo)
        except Exception as e:
            rc = 2
            log("FEHLER %s: %s" % (repo, e))
    log("ENDE rc=%d" % rc)
    sys.exit(rc)


if __name__ == "__main__":
    main()