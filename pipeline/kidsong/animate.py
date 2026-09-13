"""
kidsong.animate — Turn each still keyframe into a short animated clip.

`kidsong.scenes` paints one SDXL keyframe PNG per verse. This stage brings those
frames to life: every keyframe is fed to an image-to-video diffusion model along
with a short *motion prompt* (the verse's action, e.g. "the kids clap and bounce")
and rendered into a few-second MP4. The downstream assembler then concatenates the
clips (or falls back to a Ken-Burns still for any scene that failed to animate).

Two backends are supported, selected by cfg["kidsong"]["video_backend"]:
  - "ltx"  (default): Lightricks LTX-Video via `LTXImageToVideoPipeline`. Fast,
    T5-conditioned, bfloat16. Good motion at low VRAM.
  - "wan"           : `WanImageToVideoPipeline` (Wan2.2-TI2V-5B). Heavier; the VAE
    is kept in float32 (as the diffusers docs recommend) while the transformer /
    text encoder run in bfloat16.

VRAM strategy (RTX 5070, 12 GB, frequently ~10 GB already spoken for by other
apps — video models are far larger than SDXL, so offloading is not optional):
  - Never `.to("cuda")`. `enable_model_cpu_offload()` streams each sub-model onto
    the GPU only while it runs; an optional `sequential_offload` flag switches to
    `enable_sequential_cpu_offload()` (slower, but survives the tightest budgets).
  - VAE tiling + slicing keep the temporal VAE decode's peak memory small — this
    is where video pipelines usually blow up.
  - `torch.cuda.empty_cache()` between scenes and a full teardown at the end,
    because the caller (a long-lived Flask process) keeps using the GPU afterwards.
  - Out-of-memory is caught *per scene*: free memory, retry once, and if it still
    fails, log a warning and yield `None` for that clip so one bad scene can't sink
    the whole song — the assembler falls back to the still keyframe.

torch/diffusers are imported lazily inside `animate_scenes` so that merely
`import pipeline.kidsong.animate` stays cheap and GPU-free.
"""
import gc
import os


# ------------------------------------------------------------------ settings ---
_DEFAULTS = {
    "video_backend": "ltx",
    "video_model": "Lightricks/LTX-Video-0.9.1",
    "wan_model": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    "clip_width": 480,          # divisible by 32
    "clip_height": 832,         # divisible by 32
    "clip_frames": 49,          # (frames - 1) divisible by 8; short clips loop cleanly, long ones drift/smear
    "clip_fps": 24,
    "video_steps": 30,
    "video_guidance": 5.0,
    "seed": 20260717,           # mirrors scenes.py so a scene's still and clip share a seed
    "sequential_offload": False,
    "hf_home": "D:/brainrot/hf",
}

# Short style guard prepended to every motion prompt: nudges the model toward a
# calm, character-consistent cartoon animation instead of wild reinterpretation.
_STYLE_GUARD_PREFIX = "smooth gentle 3D cartoon animation, "
_STYLE_GUARD_SUFFIX = ", consistent characters, steady camera"

# Shortened form of scenes.NEGATIVE plus motion-specific artifacts. Video models
# are especially prone to morphing/jitter, so those are called out explicitly.
NEGATIVE = (
    "photorealistic, live action, text, watermark, deformed, extra limbs, "
    "extra fingers, blurry, lowres, adult, crowd, duplicate characters, "
    "jitter, morphing, warping, static image"
)


# ------------------------------------------------------------------ helpers ---
def _is_oom(exc, oom_types):
    """True if `exc` is a CUDA out-of-memory error (typed or message-based)."""
    if oom_types and isinstance(exc, oom_types):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def _prepare_conditioning_image(png_path, width, height):
    """Load a keyframe and resize+center-crop it to exactly (width, height).

    The i2v pipelines condition on the first frame, so the aspect ratio of the
    input must match the target clip or the model letterboxes / stretches it.
    `ImageOps.fit` does a cover-fit (scale to fill, crop the overflow, centered).
    """
    from PIL import Image, ImageOps

    img = Image.open(png_path).convert("RGB")
    return ImageOps.fit(img, (width, height), method=Image.LANCZOS, centering=(0.5, 0.5))


# ------------------------------------------------------------------- public ---
def animate_scenes(scene_pngs, motion_prompts, out_dir, cfg, on_progress=None):
    """
    Animate each keyframe PNG into a short MP4 and return their absolute paths.

    scene_pngs:     list of keyframe PNG paths (one per verse).
    motion_prompts: same-length list of short action descriptions, e.g.
                    "the three kids clap their hands together, smiling and bouncing".
    out_dir:        directory to write scene_00.mp4, scene_01.mp4, ... (created).
    cfg:            the loaded config dict (reads cfg["kidsong"]).
    on_progress:    optional callable(str); printed if None.

    Returns a list the same length as `scene_pngs`; each entry is the absolute
    path of the rendered clip, or `None` if that scene failed (caller falls back
    to the still keyframe). Raises only if the pipeline itself cannot be loaded.
    """
    if len(scene_pngs) != len(motion_prompts):
        raise ValueError(
            f"scene_pngs ({len(scene_pngs)}) and motion_prompts "
            f"({len(motion_prompts)}) must be the same length"
        )

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

    # Lazy imports (repo convention): keep `import pipeline.kidsong.animate` cheap.
    import torch
    from diffusers.utils import export_to_video

    backend = str(opt("video_backend")).lower()
    width = int(opt("clip_width"))
    height = int(opt("clip_height"))
    num_frames = int(opt("clip_frames"))
    fps = int(opt("clip_fps"))
    steps = int(opt("video_steps"))
    guidance = float(opt("video_guidance"))
    seed = int(opt("seed"))
    sequential = bool(opt("sequential_offload"))

    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # The i2v pipelines add conditioning/decode noise directly against the (CUDA)
    # latents, so — unlike SDXL — the generator must live on the same device or
    # torch raises "Expected a 'cuda' device type for generator". With cpu-offload
    # the compute device is still cuda:0, so seed on cuda when it's available.
    gen_device = "cuda" if torch.cuda.is_available() else "cpu"

    # Which exception classes count as OOM (varies across torch versions).
    oom_types = tuple(
        t for t in (
            getattr(torch, "OutOfMemoryError", None),
            getattr(torch.cuda, "OutOfMemoryError", None),
        ) if isinstance(t, type)
    )

    # ---- Load the requested image-to-video pipeline -------------------------
    if backend == "wan":
        from diffusers import AutoencoderKLWan, WanImageToVideoPipeline

        model = opt("wan_model")
        log(f"Loading Wan i2v model '{model}' (first run downloads many GB)…")
        # Wan's VAE is numerically sensitive — the diffusers docs keep it in fp32
        # while the transformer / text encoder run in bf16.
        vae = AutoencoderKLWan.from_pretrained(
            model, subfolder="vae", torch_dtype=torch.float32
        )
        pipe = WanImageToVideoPipeline.from_pretrained(
            model, vae=vae, torch_dtype=torch.bfloat16
        )
    else:
        from diffusers import LTXImageToVideoPipeline

        model = opt("video_model")
        log(f"Loading LTX i2v model '{model}' (first run downloads several GB)…")
        pipe = LTXImageToVideoPipeline.from_pretrained(
            model, torch_dtype=torch.bfloat16
        )

    # Stream sub-models to the GPU on demand instead of pinning everything in VRAM.
    if sequential:
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.enable_model_cpu_offload()

    # VAE slicing + tiling — the temporal decode is the usual OOM culprit. The
    # helper API moved between diffusers generations, so try pipeline-level first
    # and fall back to the vae-level methods.
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
        total = len(scene_pngs)
        for i, (png_path, motion) in enumerate(zip(scene_pngs, motion_prompts)):
            log(f"Animating scene {i + 1}/{total} on GPU…")
            prompt = _STYLE_GUARD_PREFIX + str(motion).strip() + _STYLE_GUARD_SUFFIX
            cond_image = _prepare_conditioning_image(png_path, width, height)
            out_path = os.path.join(out_dir, f"scene_{i:02d}.mp4")

            def _render():
                # Deterministic but distinct per scene; generator on the compute
                # device (cuda) because LTX/Wan sample noise against cuda latents.
                generator = torch.Generator(device=gen_device).manual_seed(seed + i)
                call_kwargs = dict(
                    image=cond_image,
                    prompt=prompt,
                    negative_prompt=NEGATIVE,
                    width=width,
                    height=height,
                    num_frames=num_frames,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    generator=generator,
                )
                # LTX takes an explicit source frame_rate; Wan does not.
                if backend != "wan":
                    call_kwargs["frame_rate"] = fps
                result = pipe(**call_kwargs)
                # Both LTXPipelineOutput and WanPipelineOutput expose `.frames`,
                # a batch of clips; we render one clip per call.
                return result.frames[0]

            try:
                frames = _render()
            except Exception as exc:
                if not _is_oom(exc, oom_types):
                    log(f"  Scene {i + 1} failed to animate ({exc!r}); using still fallback.")
                    paths.append(None)
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                # Free everything we can and retry exactly once.
                log("  VRAM ran out — clearing cache and retrying scene once…")
                torch.cuda.empty_cache()
                gc.collect()
                try:
                    frames = _render()
                except Exception as exc2:
                    # Even the retry lost the race for memory — warn and fall back
                    # to the still for this scene rather than aborting the song.
                    if _is_oom(exc2, oom_types):
                        log(
                            "  Still out of VRAM after retry — skipping animation "
                            "for this scene (falling back to still). Close GPU-heavy "
                            "apps (browser, llama-server) for full animation."
                        )
                    else:
                        log(f"  Scene {i + 1} failed on retry ({exc2!r}); using still fallback.")
                    paths.append(None)
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue

            export_to_video(frames, out_path, fps=fps)
            paths.append(out_path)

            # Release the decoded frames and reclaim VRAM before the next scene.
            del frames
            torch.cuda.empty_cache()
    finally:
        # Full teardown — the caller keeps working in the same long-lived process,
        # so give the (substantial) GPU memory back now.
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
    from pipeline.config import load_config

    cfg = load_config()
    root = cfg["_root"]

    scene_dir = os.path.join(root, "output", "kidsong_scene_test")
    test_pngs = [
        os.path.join(scene_dir, "scene_00.png"),
        os.path.join(scene_dir, "scene_01.png"),
    ]
    test_motions = [
        "the two toddlers wave hello and bounce happily, balloons drifting gently",
        "the two toddlers clap their hands and sway side to side, smiling",
    ]

    out = os.path.join(root, "output", "kidsong_animate_test")
    results = animate_scenes(test_pngs, test_motions, out, cfg)
    print("\nAnimated clips:")
    for p in results:
        print(" ", p)
