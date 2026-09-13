"""
kidsong.scenes — Render one still image per verse for a children's song video.

Uses Stable Diffusion XL (via diffusers) to paint a Cocomelon-style 3D-cartoon
frame for each verse of the song. A single character description is prepended to
every prompt so the recurring toddler cast looks consistent scene to scene.

VRAM strategy (RTX 5070, 12 GB, often ~10 GB already in use):
  - Never `.to("cuda")`. We use `enable_model_cpu_offload()` so each sub-model is
    streamed onto the GPU only while it runs and pushed back to CPU afterwards.
  - VAE slicing + tiling keep the decode step's peak memory tiny.
  - `torch.cuda.empty_cache()` between scenes and a full teardown at the end,
    because the caller keeps running (Whisper + video encoding) inside one
    long-lived Flask process.
  - Out-of-memory is caught per image: we free memory and retry once, then raise
    a clear, user-facing error telling them to close other GPU apps.

torch/diffusers are imported lazily inside `render_scenes` so that merely
`import pipeline.kidsong` stays cheap and the rest of the app runs without the
heavy GPU deps installed.
"""
import gc
import os


# ------------------------------------------------------------------ settings ---
_DEFAULTS = {
    "image_model": "stabilityai/stable-diffusion-xl-base-1.0",
    "image_width": 832,
    "image_height": 1216,
    "steps": 28,
    "guidance": 6.5,
    "seed": 20260717,
    "hf_home": "D:/brainrot/hf",
}

# Kept short on purpose: CLIP truncates prompts at 77 tokens, and everything —
# style, cast, scene action — must fit inside that budget or SDXL silently
# ignores the tail. The character description is compacted to its lead words.
STYLE = (
    "adorable 3D animated preschool cartoon render, big expressive eyes, "
    "soft rounded features, glossy 3D style, bright cheerful colors, "
    "one single scene"
)

# Word budget for the cast description inside the prompt; identity anchors
# (skin, hair) come first in lyrics' `characters` sentence, outfits later,
# so a head-truncation keeps what matters for consistency.
_CHARACTER_WORDS = 20


def _compact(text, max_words=_CHARACTER_WORDS):
    words = str(text).split()
    return " ".join(words[:max_words]).rstrip(",;:") if words else ""

NEGATIVE = (
    "photorealistic, live action, photo, text, watermark, logo, scary, creepy, "
    "dark, deformed, extra limbs, extra fingers, blurry, lowres, adult, "
    "crowd, many children, duplicate characters, collage, grid, split frame"
)


# ------------------------------------------------------------------ helpers ---
def _is_oom(exc, oom_types):
    """True if `exc` is a CUDA out-of-memory error (typed or message-based)."""
    if oom_types and isinstance(exc, oom_types):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


# ------------------------------------------------------------------- public ---
def render_scenes(scene_descriptions, characters, out_dir, cfg, on_progress=None):
    """
    Render one PNG per scene description with SDXL and return their absolute paths.

    scene_descriptions: list of short visual descriptions (one per verse).
    characters:         one-sentence visual description of the recurring cast;
                        prepended to every prompt for character consistency.
    out_dir:            directory to write scene_00.png, scene_01.png, ...
    cfg:                the loaded config dict (reads cfg["kidsong"]).
    on_progress:        optional callable(str); printed if None.
    """
    def log(msg):
        if on_progress:
            on_progress(msg)
        else:
            print(msg)

    settings = cfg.get("kidsong", {})

    def opt(key):
        return settings.get(key, _DEFAULTS[key])

    # HF_HOME must be set BEFORE torch/diffusers import so downloads land on D:.
    hf_home = opt("hf_home")
    hf_drive = os.path.splitdrive(os.path.abspath(hf_home))[0]
    if not hf_drive or os.path.isdir(hf_drive + os.sep):
        os.environ.setdefault("HF_HOME", hf_home)
    # else: target drive missing — let HuggingFace use its default cache.

    # Lazy imports (repo convention): keep `import pipeline.kidsong` GPU-free.
    import torch
    from diffusers import StableDiffusionXLPipeline

    model = opt("image_model")
    width = int(opt("image_width"))
    height = int(opt("image_height"))
    steps = int(opt("steps"))
    guidance = float(opt("guidance"))
    seed = int(opt("seed"))

    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Which exception classes count as OOM (varies across torch versions).
    oom_types = tuple(
        t for t in (
            getattr(torch, "OutOfMemoryError", None),
            getattr(torch.cuda, "OutOfMemoryError", None),
        ) if isinstance(t, type)
    )

    log("Loading SDXL image model (first run downloads several GB)…")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    # Stream sub-models to GPU on demand instead of pinning everything in VRAM.
    pipe.enable_model_cpu_offload()

    # VAE slicing + tiling — API moved between diffusers generations, so try the
    # pipeline-level helpers first and fall back to the vae-level ones.
    try:
        pipe.enable_vae_slicing()
    except Exception:
        try:
            pipe.vae.enable_slicing()
        except Exception:
            pass
    try:
        pipe.enable_vae_tiling()
    except Exception:
        try:
            pipe.vae.enable_tiling()
        except Exception:
            pass

    paths = []
    try:
        total = len(scene_descriptions)
        for i, desc in enumerate(scene_descriptions):
            log(f"Rendering scene {i + 1}/{total} on GPU…")
            prompt = STYLE + ", " + _compact(characters) + ", " + desc

            def _render():
                # Deterministic but distinct per scene; CPU generator works with
                # model-cpu-offload (the unet still runs on the GPU).
                generator = torch.Generator(device="cpu").manual_seed(seed + i)
                result = pipe(
                    prompt=prompt,
                    negative_prompt=NEGATIVE,
                    width=width,
                    height=height,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    generator=generator,
                )
                return result.images[0]

            try:
                image = _render()
            except Exception as exc:
                if not _is_oom(exc, oom_types):
                    raise
                # Free everything we can and retry exactly once.
                log("  VRAM ran out — clearing cache and retrying scene once…")
                torch.cuda.empty_cache()
                gc.collect()
                try:
                    image = _render()
                except Exception as exc2:
                    if not _is_oom(exc2, oom_types):
                        raise
                    raise RuntimeError(
                        "Not enough free VRAM — close GPU-heavy apps "
                        "(browser, llama-server) and retry."
                    ) from exc2

            path = os.path.join(out_dir, f"scene_{i:02d}.png")
            image.save(path)
            paths.append(path)

            # Release the decoded image and reclaim VRAM before the next scene.
            del image
            torch.cuda.empty_cache()
    finally:
        # Full teardown — the caller keeps working (Whisper + encoding) in the
        # same long-lived process, so give the GPU memory back now.
        try:
            del pipe
        except Exception:
            pass
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    return paths


# --------------------------------------------------------------------- main ---
if __name__ == "__main__":
    import sys

    from pipeline.config import load_config

    test_characters = (
        "two cheerful Black toddler siblings, a little girl with two puff pigtails "
        "in a yellow dress and a little boy with short curly hair in blue overalls"
    )

    if len(sys.argv) > 1:
        descriptions = sys.argv[1:]
    else:
        descriptions = [
            "the two toddlers waving hello in a sunny green backyard with balloons",
            "the two toddlers clapping and dancing in a colorful playroom",
        ]

    cfg = load_config()
    out = os.path.join(cfg["_root"], "output", "kidsong_scene_test")
    results = render_scenes(descriptions, test_characters, out, cfg)
    print("\nRendered scenes:")
    for p in results:
        print(" ", p)
