"""Standalone ACE-Step 1.5 one-shot generator (runs in the isolated ace-venv).

This script is NOT part of the pipeline package: it imports only the standard
library and `acestep`, and it is executed by the official ACE-Step environment's
Python interpreter (D:/brainrot/ace-venv), invoked by file path with a single
argv -- the path to a job JSON.

The repo's own venv (Python 3.13 / torch 2.11 / transformers 5.x) produced
corrupted, noise-like audio when it drove ACE-Step in-process. The verified-good
recipe lives in the pinned official env (Python 3.12 / torch 2.7.1+cu128 /
transformers 4.57.6 / ACE-Step 1.5), so `sing_song` shells out to this script
there instead.

Job JSON schema (all keys required except lm_model, which may be null):
    {
      "caption":  "<style/genre prompt>",
      "lyrics":   "<[Verse]-tagged lyrics>",
      "duration": <float seconds>,
      "seed":     <int>,
      "lm_model": "<5Hz LM checkpoint id>" | null,
      "out_dir":  "<dir for the raw wav + result.json>"
    }

On success writes <out_dir>/result.json:
    {"audio_path": "<raw wav ACE saved>", "sample_rate": <int>, "metas": {...}}
and prints DONE_OK. On failure prints an ASCII marker and exits non-zero.

This mirrors the ground-truth gen_official.py: acestep's own get_gpu_config tier
defaults, the turbo DiT (acestep-v15-turbo), inference_steps=8, and the MANDATORY
turbo shift=3.0. Stdout is ASCII only (the parent runs it with PYTHONUTF8=1).
"""
import json
import os
import sys
import time

# Env MUST be set before importing torch / acestep so all weights resolve on D:.
# The parent passes these in the subprocess env; setdefault keeps the parent's
# values while still working if the script is ever run by hand.
os.environ.setdefault("ACESTEP_CHECKPOINTS_DIR", "D:/brainrot/acestep")
os.environ.setdefault("HF_HOME", "D:/brainrot/hf")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

DIT_CONFIG = "acestep-v15-turbo"
STEPS = 8            # turbo default
SHIFT = 3.0          # OFFICIAL turbo recommendation (INFERENCE.md) -- MANDATORY
VOCAL_LANGUAGE = "en"
VRAM_AUTO_OFFLOAD_THRESHOLD_GB = 20.0
DEFAULT_LM_MODEL = "acestep-5Hz-lm-0.6B"


def _extract_metas(result):
    """Best-effort bpm/key readout from the LM's planning metadata."""
    extra = getattr(result, "extra_outputs", None) or {}
    lm_meta = extra.get("lm_metadata")
    metas = {}
    if isinstance(lm_meta, dict):
        for key in ("bpm", "key"):
            if key in lm_meta and lm_meta[key] is not None:
                metas[key] = lm_meta[key]
        # Keep whatever the LM planned even if the field names differ, so the
        # caller (and future debugging) can see it. Small dict, safe to embed.
        if not metas:
            metas = {k: v for k, v in lm_meta.items()
                     if isinstance(v, (str, int, float, bool))}
    return metas


def main():
    if len(sys.argv) < 2:
        print("USAGE: ace_runner.py <job.json>")
        sys.exit(64)
    job_path = sys.argv[1]
    with open(job_path, "r", encoding="utf-8") as fh:
        job = json.load(fh)

    caption = str(job["caption"])
    lyrics = str(job["lyrics"])
    duration = float(job["duration"])
    seed = int(job["seed"])
    out_dir = os.path.abspath(job["out_dir"])
    os.makedirs(out_dir, exist_ok=True)

    checkpoints_dir = os.path.abspath(os.environ["ACESTEP_CHECKPOINTS_DIR"])
    project_root = os.path.dirname(checkpoints_dir) or checkpoints_dir

    import torch  # noqa: F401  (import side effects / parity with gen_official)
    from acestep.gpu_config import (
        get_gpu_config,
        set_global_gpu_config,
        resolve_lm_backend,
    )
    from acestep.handler import AceStepHandler
    from acestep.llm_inference import LLMHandler
    from acestep.inference import (
        GenerationParams,
        GenerationConfig,
        generate_music,
    )

    gpu_config = get_gpu_config()
    set_global_gpu_config(gpu_config)
    mem = gpu_config.gpu_memory_gb
    auto_offload = 0 < mem < VRAM_AUTO_OFFLOAD_THRESHOLD_GB
    # Official recommended backend for this tier; on Windows the handler's vllm
    # preflight (no Triton) transparently falls back to 'pt'.
    backend = resolve_lm_backend(gpu_config.recommended_backend, gpu_config)

    # A null lm_model in the job means "use the tier default" -- that is exactly
    # what produced the verified-good song (0.6B on this 12GB card).
    lm_model = job.get("lm_model")
    if not lm_model:
        lm_model = (gpu_config.available_lm_models or [DEFAULT_LM_MODEL])[0]

    print("=" * 60)
    print("GPU %.2fGB tier=%s recommended_backend=%s -> backend=%s"
          % (mem, gpu_config.tier, gpu_config.recommended_backend, backend))
    print("auto_offload(<20GB)=%s lm_model=%s DiT=%s steps=%s shift=%s seed=%s"
          % (auto_offload, lm_model, DIT_CONFIG, STEPS, SHIFT, seed))
    print("=" * 60)

    # ---- DiT (+ VAE + text encoder) ----
    print("Loading DiT/VAE/text-encoder ...")
    t = time.time()
    dit = AceStepHandler()
    status, ok = dit.initialize_service(
        project_root=project_root,
        config_path=DIT_CONFIG,
        device="cuda",
        offload_to_cpu=auto_offload,
        offload_dit_to_cpu=False,
        use_mlx_dit=False,
    )
    print("  DiT init ok=%s (%.1fs): %s" % (ok, time.time() - t, status))
    if not ok:
        print("DIT_INIT_FAILED")
        sys.exit(2)

    # ---- 5Hz "thinking" LM ----
    print("Loading 5Hz LM %s (backend=%s) ..." % (lm_model, backend))
    t = time.time()
    llm = LLMHandler()
    lm_status, lm_ok = llm.initialize(
        checkpoint_dir=checkpoints_dir,
        lm_model_path=lm_model,
        backend=backend,
        device="cuda",
        offload_to_cpu=auto_offload,
    )
    print("  LM init ok=%s (%.1fs): %s" % (lm_ok, time.time() - t, lm_status))
    print("  LM actual backend attr: %s" % getattr(llm, "llm_backend", "?"))
    if not lm_ok:
        print("LM_INIT_FAILED")
        sys.exit(3)

    # ---- Generation ----
    params = GenerationParams(
        task_type="text2music",
        caption=caption,
        lyrics=lyrics,
        vocal_language=VOCAL_LANGUAGE,
        duration=duration,
        inference_steps=STEPS,
        shift=SHIFT,
        guidance_scale=1.0,     # turbo auto-corrects CFG to 1.0
        seed=seed,
        thinking=True,
        use_cot_metas=True,     # let the LM plan bpm/key (official)
        use_cot_caption=False,  # keep the exact caption
        use_cot_lyrics=False,   # keep the exact lyrics
        use_cot_language=False,
    )
    config = GenerationConfig(
        batch_size=1,
        audio_format="wav",
        use_random_seed=False,
        seeds=[seed],
    )

    print("Generating ...")
    t = time.time()
    result = generate_music(dit, llm, params=params, config=config, save_dir=out_dir)
    print("GEN_TIME=%.1fs success=%s" % (time.time() - t, getattr(result, "success", None)))

    if not getattr(result, "success", False) or not result.audios:
        print("GEN_FAILED:", getattr(result, "error", None) or getattr(result, "status_message", None))
        sys.exit(4)

    audio = result.audios[0]
    raw_path = audio.get("path")
    sample_rate = audio.get("sample_rate")
    metas = _extract_metas(result)

    result_json = {
        "audio_path": raw_path,
        "sample_rate": sample_rate,
        "metas": metas,
    }
    result_path = os.path.join(out_dir, "result.json")
    with open(result_path, "w", encoding="utf-8") as fh:
        json.dump(result_json, fh)

    print("RAW_PATH:", raw_path)
    print("RAW_SR:", sample_rate)
    print("METAS:", metas)
    print("RESULT_JSON:", result_path)
    print("DONE_OK")


if __name__ == "__main__":
    main()
