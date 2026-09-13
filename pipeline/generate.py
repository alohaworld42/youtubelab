"""
generate.py — Run the whole pipeline: script -> voices -> captions -> video [-> upload].

Use from the web UI (app.py) or the command line:
    python -m pipeline.generate facts
    python -m pipeline.generate conversation --topic "cats vs dogs" --upload
"""
import os
import re
import time

from pipeline.config import abspath, load_config


def _slug(text):
    s = re.sub(r"[^\w\s-]", "", text).strip().lower()
    return re.sub(r"[\s_-]+", "-", s)[:50] or "video"


def _hook_spec(cfg, title, duration):
    """Motion-graphic spec for the intro title hook, or None when disabled /
    no room. Rides the same graphics pipeline as the LLM specs."""
    hook = cfg.get("hook") or {}
    if not hook.get("enabled", True) or not (title or "").strip():
        return None
    start = float(hook.get("delay_seconds", 0.1))
    end = min(start + float(hook.get("seconds", 1.4)), float(duration))
    if end - start < 0.5:
        return None
    return {
        "type": "title",
        "text": title,
        "start": start,
        "end": end,
        "y_ratio": float(hook.get("vertical_position", 0.18)),
    }


def _build_graphics(cfg, script, segments, words, duration):
    """Merge LLM per-line graphics with deterministic number stat-counters into
    one time-ordered, non-overlapping, capped list of motion-graphic specs."""
    from pipeline.highlights import extract_highlights, to_stat_specs

    m = cfg.get("motion") or {}
    if not m.get("enabled", True):
        return []

    hold = float(m.get("hold_seconds", 0.5))
    max_count = int(m.get("max_count", 5))
    min_gap = float(m.get("min_gap_seconds", 0.8))

    # LLM graphics: each carried on a script line, timed to that line's segment.
    primary = []
    for ln, seg in zip(script["lines"], segments):
        g = ln.get("graphic")
        if g:
            primary.append({**g, "start": seg["start"], "end": seg["end"] + hold})

    # Deterministic stat counters for spoken numbers (fills in when the LLM is
    # sparse); only added if they don't clash with an LLM graphic's window.
    extra = to_stat_specs(extract_highlights(words, cfg)) if m.get("auto_stats", True) else []

    def clashes(g, chosen):
        return any(
            not (g["end"] + min_gap <= e["start"] or g["start"] >= e["end"] + min_gap)
            for e in chosen
        )

    merged = sorted(primary, key=lambda g: g["start"])[:max_count]
    for g in sorted(extra, key=lambda g: g["start"]):
        if len(merged) >= max_count:
            break
        if not clashes(g, merged):
            merged.append(g)
    merged.sort(key=lambda g: g["start"])

    # Clamp to the (possibly truncated) video duration.
    out = []
    for g in merged:
        if g["start"] < duration:
            g["end"] = min(g["end"], duration)
            out.append(g)
    return out


def generate(video_type, topic=None, do_upload=False, cfg=None, on_progress=None,
             channel_overrides=None, extra_context=None,
             background_path=None, music_path=None, style=None, resume_base=None):
    if cfg is None:
        cfg = load_config()
    if style:
        from pipeline.config import apply_style

        cfg = apply_style(cfg, style)
    if channel_overrides:
        from pipeline.config import merge_overrides

        cfg = merge_overrides(cfg, channel_overrides)

    # The kids' song type has its own pipeline (GPU scene images instead of
    # gameplay footage) — dispatch early so the shared flow below stays simple.
    # resume_base (Contract 3) picks up an interrupted run instead of always
    # starting a fresh episode: generate_kidsong reuses its song, shot ledger
    # and every take already on disk, so a scheduler retry after a mid-render
    # infra failure doesn't re-render shots that already passed QC.
    if video_type == "kidsong":
        from pipeline.kidsong.generate import generate_kidsong

        return generate_kidsong(topic, do_upload, cfg, on_progress, resume_base=resume_base)

    def step(msg):
        if on_progress:
            on_progress(msg)
        else:
            try:
                print(f"  → {msg}")
            except UnicodeEncodeError:
                # Windows console with cp1252 (e.g. redirected output)
                print(("  -> " + msg).encode("ascii", "replace").decode())

    out_dir = abspath(cfg, cfg["paths"]["output_dir"])
    os.makedirs(out_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")

    # 1. Script
    step("Writing script with the LLM…")
    from pipeline.script_gen import generate_script

    script = generate_script(video_type, topic, cfg, extra_context=extra_context)
    base = f"{stamp}-{video_type}-{_slug(script['title'])}"
    video_path = os.path.join(out_dir, base + ".mp4")

    # 1b. Alternative render engine: short-video-maker (Remotion + scene B-roll).
    # Falls back to the built-in MoviePy engine if it fails / isn't running.
    engine = (cfg.get("render_engine") or "moviepy").lower()
    if engine == "svm":
        from studio import svm

        try:
            step("Rendering via short-video-maker…")
            svm.render_via_svm(cfg, script, video_path, style=style, on_progress=step)
            return _finish(cfg, step, script, video_path, do_upload)
        except Exception as e:
            step(f"short-video-maker unavailable ({e}); falling back to built-in engine.")

    # 2. Voices
    step("Generating voices (edge-tts)…")
    from pipeline.tts import synthesize

    voice_path = os.path.join(out_dir, base + ".wav")
    tts = synthesize(script["lines"], cfg, voice_path)

    duration = min(tts["duration"], cfg["video"]["max_seconds"])

    # 3. Captions
    step("Aligning word-level captions (Whisper)…")
    from pipeline.captions import get_word_timestamps

    words = get_word_timestamps(voice_path, cfg, uppercase=cfg["captions"].get("uppercase", True))
    words = [w for w in words if w["start"] < duration]
    for w in words:
        w["end"] = min(w["end"], duration)

    # 3a. Motion graphics: animated overlays (stat counters, statement cards,
    # comparison bars) timed to the transcript. Two sources, merged:
    #   - LLM per-line "graphic" specs, mapped to that line's spoken window
    #   - deterministic stat counters for spoken numbers not already covered
    graphics = _build_graphics(cfg, script, tts["segments"], words, duration)

    # Intro hook: stamp the title on screen for the first beat.
    hook = _hook_spec(cfg, script.get("title"), duration)
    if hook:
        graphics.insert(0, hook)

    # Virtual camera: LLM per-line "camera" hints ride on the tts segments
    # (1:1 with lines) so the beat planner can pin moves to spoken windows.
    for ln, seg in zip(script["lines"], tts["segments"]):
        if ln.get("camera"):
            seg["camera_hint"] = ln["camera"]

    # 3b. Scene mode: one generated AI clip per scene (CoComelon-style),
    # instead of a single background under the whole video. Kids style enables
    # it; falls back silently to the normal background path if ComfyUI is off.
    #
    # GPU scene mode (video.gpu_scene_mode, default False) tries FIRST: it
    # renders real animated visuals for the NON-kids 2D types (facts/reddit/
    # scary/conversation) via the same mature ComfyClient + render-style
    # registry the kidsong channel uses (see studio/gpu_scenes.py). Off by
    # default, so every existing facts/reddit/scary/conversation run is
    # unaffected; when it returns None (disabled, ComfyUI down, a scene
    # failed) this falls through to the legacy scene_mode path below exactly
    # as if gpu_scene_mode had never been set.
    scene_clips = None
    if cfg["video"].get("gpu_scene_mode"):
        try:
            from studio.gpu_scenes import build_scene_clips_gpu

            gpu_scene_dir = os.path.join(out_dir, base + "_gpu_scenes")
            scene_clips = build_scene_clips_gpu(
                script, tts["segments"], cfg, gpu_scene_dir, on_progress=step
            )
            if scene_clips:
                step(f"{len(scene_clips)} GPU-Szenen generiert.")
        except Exception as e:
            step(f"GPU-Szenen-Modus fehlgeschlagen ({e}); nutze Fallback.")
            scene_clips = None

    if not scene_clips and cfg["video"].get("scene_mode"):
        try:
            from studio.scenes import generate_scene_clips

            scene_dir = os.path.join(out_dir, base + "_scenes")
            scene_clips = generate_scene_clips(
                script, tts["segments"], cfg, scene_dir, on_progress=step
            )
            if scene_clips:
                step(f"{len(scene_clips)} AI-Szenen generiert.")
        except Exception as e:
            step(f"Szenen-Modus fehlgeschlagen ({e}); nutze normalen Hintergrund.")
            scene_clips = None

    # 4. Video
    from pipeline.assemble import build_video

    build_video(
        cfg,
        voice_path,
        tts["segments"],
        words,
        duration,
        video_path,
        on_progress=step,
        background_path=background_path,
        music_path=music_path,
        scene_clips=scene_clips,
        graphics=graphics,
    )

    # tidy: drop the intermediate wav
    try:
        os.remove(voice_path)
    except OSError:
        pass

    return _finish(cfg, step, script, video_path, do_upload, duration=duration)


def _probe_duration(video_path):
    import json
    import subprocess

    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "json", video_path],
            capture_output=True, text=True, timeout=15,
        )
        return round(float(json.loads(out.stdout)["format"]["duration"]), 3)
    except Exception:
        return None


def _finish(cfg, step, script, video_path, do_upload, duration=None):
    """Shared tail: thumbnail, result dict, optional upload. Used by both the
    built-in engine and the short-video-maker engine."""
    if duration is None:
        duration = _probe_duration(video_path)
    step("Generating thumbnail…")
    from pipeline.thumbnail import generate_thumbnail

    thumb_path = generate_thumbnail(video_path, script["title"], cfg)

    result = {
        "video_path": video_path,
        "title": script["title"],
        "description": script["description"],
        "tags": script["tags"],
        "duration": duration,
        "script": script,
        "thumb_path": thumb_path,
    }

    if do_upload:
        step("Uploading to YouTube…")
        from pipeline.youtube_upload import upload

        up = upload(video_path, script["title"], script["description"], script["tags"], cfg)
        result["youtube"] = up

    step("Done.")
    return result


if __name__ == "__main__":
    import argparse
    import sys

    p = argparse.ArgumentParser()
    p.add_argument("video_type", help="reddit | conversation | facts | scary | kids")
    p.add_argument("--topic", default=None)
    p.add_argument("--style", default=None, help="style preset from config: kids | brainrot | clean")
    p.add_argument("--engine", default=None, help="render engine: moviepy | svm")
    p.add_argument("--upload", action="store_true")
    args = p.parse_args()

    _cfg = load_config()
    if args.engine:
        _cfg["render_engine"] = args.engine
    out = generate(args.video_type, args.topic, args.upload, cfg=_cfg, style=args.style)
    # Reconfigure stdout so emoji in titles don't crash the Windows cp1252 console.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    print("\nResult:")
    for k, v in out.items():
        print(f"  {k}: {v}")
