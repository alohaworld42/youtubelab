"""Scene engine: turn a script + TTS timing into per-scene AI video clips.

This is what separates a real CoComelon-style short from "slop": every few
lines get their OWN generated 3D clip matching what is being said (the LLM
provides a "visual" per line and a recurring character description), instead
of one looped background under the whole video.
"""
import logging
import os

log = logging.getLogger("studio.scenes")

# LTX latent constraint: frame count must be n*8+1; keep within 8GB-VRAM range.
MIN_FRAMES = 49
MAX_FRAMES = 161


def build_scenes(script, segments, max_scenes=6, target_seconds=5.0):
    """Group consecutive TTS segments into scenes.

    segments are 1:1 with script lines (tts.synthesize iterates lines in
    order), so each scene knows its lines' visuals. Greedy grouping up to
    target_seconds; target grows if the video would exceed max_scenes.
    """
    lines = script.get("lines") or []
    if not segments:
        return []
    total = segments[-1]["end"]
    target = max(target_seconds, total / max_scenes)

    scenes = []
    cur = None
    for i, seg in enumerate(segments):
        visual = (lines[i].get("visual") or "").strip() if i < len(lines) else ""
        if cur is None:
            cur = {"start": seg["start"], "end": seg["end"],
                   "texts": [seg["text"]], "visuals": [visual] if visual else []}
        else:
            cur["end"] = seg["end"]
            cur["texts"].append(seg["text"])
            if visual:
                cur["visuals"].append(visual)
        if cur["end"] - cur["start"] >= target:
            scenes.append(cur)
            cur = None
    if cur:
        # merge a tiny tail into the previous scene instead of a 1s stub
        if scenes and (cur["end"] - cur["start"]) < 2.0:
            scenes[-1]["end"] = cur["end"]
            scenes[-1]["texts"].extend(cur["texts"])
            scenes[-1]["visuals"].extend(cur["visuals"])
        else:
            scenes.append(cur)

    for sc in scenes:
        sc["duration"] = round(sc["end"] - sc["start"], 3)
    return scenes


def scene_prompt(scene, script, cfg, conf=None):
    """Build the generation prompt for one scene: style prefix + recurring
    characters + this scene's visual (LLM-provided), falling back to the
    spoken text. `conf` is the active provider's config block (defaults to
    comfyui for backward compatibility)."""
    if conf is None:
        conf = cfg.get("comfyui") or {}
    template = conf.get(
        "scene_prompt_template",
        "3d cartoon animation for children, {characters}, {visual}, bright "
        "colorful colors, cute, friendly, soft studio lighting, smooth gentle "
        "motion, high quality render",
    )
    characters = (script.get("characters") or "").strip() or "cute cartoon characters"
    visual = scene["visuals"][0] if scene["visuals"] else " ".join(scene["texts"])[:160]
    return template.format(characters=characters, visual=visual)


def frames_for(duration, fps=24):
    """LTX frame count for a scene: n*8+1, clamped to the 8GB-VRAM safe range."""
    raw = int(duration * fps)
    n8 = max(MIN_FRAMES, min(MAX_FRAMES, (raw // 8) * 8 + 1))
    return n8


def pick_provider(cfg):
    """Choose the scene provider: local ComfyUI first (free), then the
    Higgsfield cloud API (paid). `video.scene_provider` pins one explicitly."""
    from studio import comfy, higgsfield

    pref = ((cfg.get("video") or {}).get("scene_provider") or "auto").lower()
    if pref in ("auto", "comfyui") and comfy.is_available(cfg):
        return "comfyui"
    if pref in ("auto", "higgsfield") and higgsfield.is_available(cfg):
        return "higgsfield"
    return None


def generate_scene_clips(script, segments, cfg, cache_dir, on_progress=None):
    """Render one AI clip per scene (local ComfyUI/LTX or Higgsfield cloud).
    Returns list of {path, start, end, duration} or None if no provider is
    available/any scene fails (caller falls back to the single-background
    path)."""
    provider = pick_provider(cfg)
    if not provider:
        return None
    conf = cfg.get(provider) or {}
    scenes = build_scenes(script, segments, max_scenes=int(conf.get("max_scenes", 6)))
    if not scenes:
        return None

    os.makedirs(cache_dir, exist_ok=True)
    if provider == "comfyui":
        from studio import comfy

        fps = conf.get("frame_rate", 24)
        # One seed for the whole video: similar prompts + same seed keep the
        # look (and the recurring character) as consistent as LTX allows
        # across scenes.
        import hashlib

        base_seed = int(hashlib.sha1(
            (script.get("title") or "video").encode("utf-8")
        ).hexdigest()[:8], 16) % 2_147_483_647
    else:
        from studio import higgsfield

    clips = []
    for i, sc in enumerate(scenes):
        prompt = scene_prompt(sc, script, cfg, conf=conf)
        if on_progress:
            on_progress(f"AI-Szene {i + 1}/{len(scenes)}: {prompt[:80]}…")
        out = os.path.join(cache_dir, f"scene_{i:02d}.mp4")
        if provider == "comfyui":
            path = comfy.generate_clip(
                cfg, None, script, out,
                seed=base_seed,
                prompt_text=prompt,
                length_frames=frames_for(sc["duration"], fps),
            )
        else:
            path = higgsfield.generate_clip(
                cfg, prompt, out,
                duration=sc["duration"],
                on_progress=on_progress,
            )
        if not path:
            log.warning("scene %s failed (%s), aborting scene mode", i, provider)
            return None
        clips.append({"path": path, "start": sc["start"], "end": sc["end"],
                      "duration": sc["duration"]})
    return clips
