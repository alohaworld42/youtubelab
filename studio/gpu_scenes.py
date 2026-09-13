"""studio.gpu_scenes — render one LTX-2.3 t2v clip per scene via the mature
`ComfyClient`, for the NON-KIDS 2D video types (facts/reddit/scary/conversation).

Today those types either loop gameplay/stock B-roll under the whole video, or
(behind the legacy `video.scene_mode` flag) call `studio.scenes.generate_scene_clips`,
which drives the OLDER `studio/comfy.py` client (positional/class_type patching,
no graph validation, no autostart). This module instead drives the SAME
title-based `ComfyClient` (`pipeline/kidsong/comfy.py`) and render-style registry
(`pipeline/kidsong/render_style.py`) the kidsong channel already proved on GPU:
validated graphs, autostart, `/free` cleanup, and patches addressed by node
title instead of guessing at a CLIPTextEncode node.

Content-safety note: this is explicitly a NON-KIDS path. A render style's
`identity_clause` ("Every child on screen is one of these Black toddlers...")
is kids-specific and is NEVER read here — only `workflow_base`, `style_lora`,
`style_strength`, `style_trigger` and `prompt_skeleton` are borrowed from the
resolved style. `is_kidsong_type` / the kidsong content gate in `studio/ideas.py`
are untouched; this module is only ever reached downstream of that gate, for
video types the gate already classified as non-kids.

Fail-soft by design: `build_scene_clips_gpu` NEVER raises. Any failure — GPU
scenes disabled, ComfyUI down/misconfigured, a scene render failing partway —
returns None so the caller (`pipeline/generate.py`) falls back to the legacy
`scene_mode` path and ultimately to a normal background, exactly like
`studio.scenes.generate_scene_clips` already does today.

Locking: this module does NOT acquire `pipeline.gpu_lock` — the caller
(the scheduler tick / CLI entry point) already holds it for the whole
`pipeline.generate.generate()` call, same contract the kidsong pipeline uses.

Config lives under `cfg["studio"]["gpu_scenes"]` (mirrors the existing
`(cfg.get("studio") or {}).get(key, default)` idiom used by
`studio/assets.py` and `studio/scheduler.py`)::

    enabled          false                 master on/off switch
    style            (kidsong.render_style  which render-style entry to borrow
                       or "pixar_toon")      workflow/LoRA/skeleton/trigger from
    prompt_template  see DEFAULT_PROMPT_TEMPLATE
    negative         see DEFAULT_NEGATIVE
    width            (video.width, snapped) latent width, multiple of 32
    height           (video.height, snapped) latent height, multiple of 32
    fps              (video.fps or 24)
    max_scenes       6                     upper bound on scenes per video
    max_frames       161 (studio.scenes.MAX_FRAMES)  per-scene frame cap
"""
import hashlib
import logging
import os
import re

from studio.scenes import MIN_FRAMES, build_scenes

log = logging.getLogger("studio.gpu_scenes")

# No {trigger}/{skeleton} clamp for children here — this is the non-kids path.
DEFAULT_PROMPT_TEMPLATE = "{trigger} {skeleton} {visual}. Cinematic, high quality, smooth motion."

DEFAULT_NEGATIVE = (
    "blurry, low quality, distorted, deformed, extra limbs, bad anatomy, "
    "watermark, text, logo, static image, still frame, jump cut, flicker"
)

_MAX_FRAMES_DEFAULT = 161  # studio.scenes.MAX_FRAMES; kept local so a future
                            # change to that constant doesn't silently change
                            # this module's default too.


def _conf(cfg):
    return ((cfg or {}).get("studio") or {}).get("gpu_scenes") or {}


def _snap32(value, minimum=64):
    """Round a latent dimension DOWN to the nearest multiple of 32 (LTX's
    EmptyLTXVLatentVideo step size — see pipeline/kidsong/generate.py::_shot_dims
    for the same rule and the "why" of a non-multiple silently truncating)."""
    return max(minimum, (int(value) // 32) * 32)


def _dims(cfg, conf):
    video = (cfg or {}).get("video") or {}
    width = int(conf.get("width") or video.get("width", 1080))
    height = int(conf.get("height") or video.get("height", 1920))
    return _snap32(width), _snap32(height)


def _frames_for(duration, fps, max_frames):
    """LTX frame count for a scene: n*8+1, clamped to [MIN_FRAMES, max_frames].
    Same math as studio.scenes.frames_for, with a configurable cap."""
    raw = int(duration * fps)
    return max(MIN_FRAMES, min(max_frames, (raw // 8) * 8 + 1))


def _resolve_borrowed_style(cfg, conf):
    """Resolve the render style named by `studio.gpu_scenes.style` (default:
    the active `kidsong.render_style`, else "pixar_toon"), WITHOUT mutating
    `cfg`. Only workflow_base/style_lora/style_strength/style_trigger/
    prompt_skeleton are ever read off the result by this module — the kids
    `identity_clause` it also carries is never used here."""
    from pipeline.kidsong.render_style import resolve_style

    kidsong_cfg = (cfg or {}).get("kidsong") or {}
    style_name = conf.get("style") or kidsong_cfg.get("render_style") or "pixar_toon"
    probe_cfg = {"kidsong": {**kidsong_cfg, "render_style": style_name}}
    return resolve_style(probe_cfg)


def _scene_prompt(scene, conf, style):
    """Prompt = template({trigger, skeleton, visual}), whitespace-normalized so
    an empty trigger (e.g. a neutralized style) never leaves a stray double
    space or leading space. Deliberately mirrors studio/scenes.py::scene_prompt
    but sources the look from the borrowed render style instead of a fixed
    "3d cartoon animation for children" template, and never touches
    identity_clause."""
    template = conf.get("prompt_template", DEFAULT_PROMPT_TEMPLATE)
    trigger = style.get("style_trigger") or ""
    skeleton = style.get("prompt_skeleton") or ""
    visual = scene["visuals"][0] if scene.get("visuals") else " ".join(scene.get("texts") or [])[:160]
    prompt = template.format(trigger=trigger, skeleton=skeleton, visual=visual)
    return re.sub(r"\s+", " ", prompt).strip()


def _base_seed(script):
    """One deterministic seed for the whole video, from its title — identical
    approach to studio/scenes.py::generate_scene_clips so the same title
    always renders the same look across scenes/runs."""
    title = (script or {}).get("title") or "video"
    return int(hashlib.sha1(title.encode("utf-8")).hexdigest()[:8], 16) % 2_147_483_647


def build_scene_clips_gpu(script, segments, cfg, cache_dir, on_progress=None):
    """Render one AI clip per scene via the mature ComfyClient (LTX-2.3 t2v).

    Returns a list of {"path", "start", "end", "duration"} dicts (the exact
    shape `pipeline.assemble.build_video` consumes as `scene_clips`), or None
    when GPU scenes are disabled, no scenes can be built, or anything fails —
    the caller then falls back to the legacy `studio.scenes` path and
    ultimately to a normal background. NEVER raises.
    """
    conf = _conf(cfg)
    if not conf.get("enabled", False):
        return None

    try:
        max_scenes = int(conf.get("max_scenes", 6))
        scenes = build_scenes(script, segments, max_scenes=max_scenes)
        if not scenes:
            return None

        from pipeline.kidsong.comfy import ComfyClient

        style = _resolve_borrowed_style(cfg, conf)
        workflow = style["workflow_base"]
        width, height = _dims(cfg, conf)
        fps = int(conf.get("fps") or (cfg.get("video") or {}).get("fps", 24))
        max_frames = int(conf.get("max_frames", _MAX_FRAMES_DEFAULT))
        negative = conf.get("negative", DEFAULT_NEGATIVE)
        seed = _base_seed(script)
    except Exception as e:
        log.warning("gpu_scenes: could not prepare scenes/style (%s) — falling back.", e)
        return None

    os.makedirs(cache_dir, exist_ok=True)
    prefix_dir = os.path.basename(os.path.normpath(cache_dir)) or "gpu_scenes"

    client = ComfyClient(cfg)
    try:
        client.ensure_up()
    except Exception as e:
        log.warning("gpu_scenes: ComfyUI unavailable (%s) — falling back.", e)
        return None

    clips = []
    try:
        for i, sc in enumerate(scenes):
            prompt = _scene_prompt(sc, conf, style)
            if on_progress:
                on_progress(f"GPU-Szene {i + 1}/{len(scenes)}: {prompt[:80]}…")
            frames = _frames_for(sc["duration"], fps, max_frames)
            out_path = os.path.join(cache_dir, f"scene_{i:02d}.mp4")
            patches = {
                "PROMPT": {"text": prompt},
                "NEGATIVE": {"text": negative},
                "SEED": {"noise_seed": seed},
                "WIDTH": {"value": width},
                "HEIGHT": {"value": height},
                "FRAMES": {"value": frames},
                "FILENAME_PREFIX": {"filename_prefix": f"gpu_scenes/{prefix_dir}/scene_{i:02d}"},
                "LORA_STYLE": {
                    "lora_name": style["style_lora"],
                    "strength_model": style["style_strength"],
                },
            }
            client.render(workflow, patches, out_path)
            clips.append({
                "path": out_path, "start": sc["start"], "end": sc["end"],
                "duration": sc["duration"],
            })
    except Exception as e:
        log.warning("gpu_scenes: scene render failed (%s) — aborting GPU scene mode.", e)
        return None
    finally:
        try:
            client.free()
        except Exception:
            pass

    return clips or None
