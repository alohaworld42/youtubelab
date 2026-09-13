"""
assemble.py — Put everything together into a 1080x1920 vertical MP4.

Layers (bottom to top):
  1. Gameplay background (random clip, random start, cropped to 9:16, muted)
  2. Optional character avatars that switch with whoever's speaking
  3. Animated motion graphics (stat counters, statement cards, comparison bars)
  4. Word-by-word captions burned in, timed from Whisper
  5. Audio = the TTS voice track (+ optional quiet background music)

Built for MoviePy 2.x  (pip install "moviepy>=2.1").
"""
import glob
import os
import random

from moviepy import (
    AudioFileClip,
    CompositeAudioClip,
    CompositeVideoClip,
    ImageClip,
    VideoFileClip,
    concatenate_videoclips,
)

from pipeline.camera import (
    apply_camera,
    camera_enabled,
    get_camera_cfg,
    make_rng,
    plan_for_scenes,
    plan_segments,
    whip_times,
)
from pipeline.caption_render import group_words, render_group
from pipeline.config import abspath
from pipeline.grade import apply_grade
from pipeline.motion import build_flash_clips, build_motion_clips, build_progress_clip
from pipeline.sfx import whip_sfx_clips


# --------------------------------------------------------------- background ---
def _pick_gameplay(cfg):
    d = abspath(cfg, cfg["paths"]["gameplay_dir"])
    clips = []
    for ext in ("*.mp4", "*.mov", "*.mkv", "*.webm"):
        clips.extend(glob.glob(os.path.join(d, ext)))
    if not clips:
        raise FileNotFoundError(
            f"No gameplay clips found in {d}. Drop at least one .mp4 there "
            "(e.g. a Subway Surfers or Minecraft parkour clip), or use the one-click "
            "gameplay download on the Setup page. See README."
        )
    # Prefer real clips; the bundled _placeholder is only a last resort so videos
    # never render with the boring fractal background when real footage exists.
    real = [c for c in clips if not os.path.basename(c).startswith("_")]
    return random.choice(real or clips)


def _cover_crop(clip, W, H):
    """Scale to cover WxH then center-crop. Works for landscape or vertical sources."""
    scale = max(W / clip.w, H / clip.h)
    clip = clip.resized((round(clip.w * scale), round(clip.h * scale)))
    x1 = (clip.w - W) / 2
    y1 = (clip.h - H) / 2
    return clip.cropped(x1=x1, y1=y1, x2=x1 + W, y2=y1 + H)


GENERATED_BG = "__generated__"


def _generated_background(cfg, duration):
    """Colorful animated gradient (slow vertical pan over blended color bands).

    Local fallback so videos never need gameplay footage — used by the kids
    style and whenever no clip source delivers.
    """
    import numpy as np
    from PIL import Image, ImageDraw, ImageFilter

    W, H = cfg["video"]["width"], cfg["video"]["height"]
    palette = cfg["video"].get("background_palette") or [[255, 94, 158], [83, 196, 255], [255, 214, 90]]
    colors = [tuple(c) for c in palette]
    if len(colors) < 2:
        colors = colors * 2

    # Tall strip: soft vertical bands between palette colors, repeated so the
    # pan can loop seamlessly, plus soft circles for depth.
    strip_h = H * 2
    img = Image.new("RGB", (W, strip_h), colors[0])
    draw = ImageDraw.Draw(img)
    stops = colors + [colors[0]]
    band = strip_h // (len(stops) - 1)
    for i in range(len(stops) - 1):
        c0, c1 = stops[i], stops[i + 1]
        for y in range(band):
            t = y / band
            col = tuple(int(c0[k] * (1 - t) + c1[k] * t) for k in range(3))
            draw.line([(0, i * band + y), (W, i * band + y)], fill=col)
    rng = random.Random(42)
    for _ in range(14):
        r = rng.randint(90, 240)
        x, y = rng.randint(-r, W), rng.randint(-r, strip_h)
        overlay = Image.new("RGB", img.size, (255, 255, 255))
        mask = Image.new("L", img.size, 0)
        ImageDraw.Draw(mask).ellipse([x, y, x + 2 * r, y + 2 * r], fill=46)
        img = Image.composite(overlay, img, mask.filter(ImageFilter.GaussianBlur(60)))
    frame = np.asarray(img)

    speed = (strip_h - H) / max(duration, 1)

    def make_frame(t):
        off = int(t * speed) % (strip_h - H)
        return frame[off:off + H, :, :]

    from moviepy import VideoClip

    return VideoClip(make_frame, duration=duration)


def _fit_clip_to(clip, seg_duration):
    """Make one scene clip exactly seg_duration long: trim if longer, slow down
    a bit if slightly shorter, loop if far shorter."""
    if clip.duration >= seg_duration:
        return clip.subclipped(0, seg_duration)
    factor = clip.duration / seg_duration
    if factor >= 0.55:
        # gentle slow-motion reads fine on cartoon backgrounds
        try:
            from moviepy import vfx

            slowed = clip.with_effects([vfx.MultiplySpeed(factor)])
        except Exception:
            slowed = clip.time_transform(lambda t: t * factor).with_duration(seg_duration)
        return slowed.with_duration(seg_duration)
    reps = int(seg_duration // clip.duration) + 1
    return concatenate_videoclips([clip] * reps).subclipped(0, seg_duration)


def _scene_background(cfg, scene_clips, duration):
    """Concatenate per-scene AI clips into one background timeline."""
    W, H = cfg["video"]["width"], cfg["video"]["height"]
    seg_ds = [max(0.5, sc["end"] - sc["start"]) for sc in scene_clips]
    cuts = []  # whip-pan boundary times, for the whoosh SFX
    # Every VideoFileClip opened here holds an ffmpeg subprocess and a pipe, and
    # NOTHING downstream closes them: MoviePy's CompositeVideoClip.close()
    # deliberately leaves member clips alone ("it remains the job of whoever
    # created it") and the base Clip.close() is a no-op. So `bg.close()` in
    # build_video released nothing, and a 16-scene render leaked 16 readers —
    # in the long-running Flask server, once per generated video. Carried out on
    # the returned clip, the same way whip_times is.
    opened = []

    if camera_enabled(cfg):
        cam = get_camera_cfg(cfg)
        fps = cfg["video"]["fps"]
        rng = make_rng(cam, scene_clips[0]["path"])
        tau = float(cam.get("transition_seconds", 0.24))
        crossfade = str(cam.get("transition")) == "crossfade" and len(scene_clips) > 1
        # With crossfades each scene renders tau longer so the overlap doesn't
        # steal time from the narration-aligned scene windows.
        extra = tau if crossfade else 0.0
        plans = plan_for_scenes([d + extra for d in seg_ds], cam, rng)
        parts = []
        for sc, seg_d, plan in zip(scene_clips, seg_ds, plans):
            src = VideoFileClip(sc["path"])
            opened.append(src)
            src = src.without_audio()
            fitted = _fit_clip_to(src, seg_d + extra)
            parts.append(apply_camera(fitted, plan, cam, (W, H), fps))
        if crossfade:
            from moviepy import vfx

            comps, start = [parts[0]], 0.0
            for prev_d, part in zip(seg_ds, parts[1:]):
                start += prev_d
                comps.append(part.with_start(start).with_effects([vfx.CrossFadeIn(tau)]))
            bg = CompositeVideoClip(comps, size=(W, H))
        else:
            bg = concatenate_videoclips(parts)
            if str(cam.get("transition")) == "whip_pan" and len(seg_ds) > 1:
                acc = 0.0
                for d in seg_ds[:-1]:
                    acc += d
                    cuts.append(acc)
    else:
        parts = []
        for sc, seg_d in zip(scene_clips, seg_ds):
            src = VideoFileClip(sc["path"])
            opened.append(src)
            parts.append(_fit_clip_to(_cover_crop(src.without_audio(), W, H), seg_d))
        bg = concatenate_videoclips(parts)

    if bg.duration < duration:
        reps = int(duration // bg.duration) + 1
        bg = concatenate_videoclips([bg] * reps)
    bg = bg.subclipped(0, duration)
    bg.whip_times = [t for t in cuts if t < duration]
    bg.source_clips = opened
    return bg


def _background(cfg, duration, background_path=None, segments=None, impulse_times=None):
    W, H = cfg["video"]["width"], cfg["video"]["height"]
    if background_path == GENERATED_BG:
        return _generated_background(cfg, duration)
    if background_path and os.path.exists(background_path):
        path = background_path
    else:
        try:
            path = _pick_gameplay(cfg)
        except FileNotFoundError:
            return _generated_background(cfg, duration)
    src = VideoFileClip(path)
    opened = [src]  # see _scene_background: nothing downstream closes these

    if cfg["video"].get("mute_gameplay", True):
        src = src.without_audio()

    # Take a random window of `duration` seconds; loop if the clip is too short.
    if src.duration >= duration:
        max_start = max(0.0, src.duration - duration)
        start = random.uniform(0, max_start)
        seg = src.subclipped(start, start + duration)
    else:
        reps = int(duration // src.duration) + 1
        seg = concatenate_videoclips([src] * reps).subclipped(0, duration)

    if camera_enabled(cfg):
        cam = get_camera_cfg(cfg)
        if impulse_times and cam.get("impulses", True):
            cam["_impulse_times"] = sorted(float(t) for t in impulse_times)
        rng = make_rng(cam, os.path.basename(path))
        plan = plan_segments(segments, duration, cam, rng)
        clip = apply_camera(seg, plan, cam, (W, H), cfg["video"]["fps"])
        clip.whip_times = whip_times(plan)
        clip.source_clips = opened
        return clip
    cropped = _cover_crop(seg, W, H)
    cropped.source_clips = opened
    return cropped


# ------------------------------------------------------------------ avatars ---
def _avatar_clips(cfg, segments):
    if not cfg["video"].get("show_character_avatars", True):
        return []
    char_dir = abspath(cfg, cfg["paths"]["characters_dir"])
    H = cfg["video"]["height"]
    W = cfg["video"]["width"]
    av_h = int(H * cfg["video"]["avatar_height_ratio"])
    y = int(H * cfg["video"]["avatar_vertical_position"])

    clips = []
    for seg in segments:
        png = os.path.join(char_dir, f"{seg['speaker']}.png")
        if not os.path.exists(png):
            continue
        dur = seg["end"] - seg["start"]
        if dur <= 0:
            continue
        try:
            av = (
                ImageClip(png, transparent=True)
                .resized(height=av_h)
                .with_start(seg["start"])
                .with_duration(dur)
                .with_position(("center", y))
            )
            clips.append(av)
        except Exception as e:
            print(f"[assemble] skipped avatar {png}: {e}")
    return clips


# ----------------------------------------------------------------- captions ---
def _caption_pop(pop_seconds):
    """Scale-over-time for a caption pop-in: starts at 75%, overshoots a few
    percent (ease-out-back), settles at 100% after pop_seconds."""
    def scale(t):
        p = min(max(t / pop_seconds, 0.0), 1.0)
        q = p - 1.0
        return max(0.75 + 0.25 * (1.0 + 2.70158 * q ** 3 + 1.70158 * q ** 2), 0.05)

    return scale


def _caption_clips(cfg, words):
    if not words:
        return []
    H = cfg["video"]["height"]
    y = int(H * cfg["captions"]["vertical_position"])
    pop_s = float(cfg["captions"].get("pop_seconds", 0.14)) if cfg["captions"].get("pop_in", True) else 0.0
    groups = group_words(words, cfg["captions"]["words_per_group"])
    clips = []
    for i, g in enumerate(groups):
        dur = max(0.12, g["end"] - g["start"])
        png = render_group(g["text"], cfg, i)
        clip = (
            ImageClip(png, transparent=True)
            .with_start(g["start"])
            .with_duration(dur)
            .with_position(("center", y))
        )
        if pop_s > 0:
            clip = clip.resized(_caption_pop(pop_s))
        clips.append(clip)
    return clips


# -------------------------------------------------------------------- audio ---
def _build_audio(cfg, voice_path, duration, music_path=None, sfx_times=None):
    voice = AudioFileClip(voice_path)
    music_dir = abspath(cfg, cfg["paths"]["music_dir"])
    vol = cfg["video"].get("music_volume", 0)
    tracks = [voice]
    if vol and vol > 0:
        songs = []
        if music_path and os.path.exists(music_path):
            songs = [music_path]
        elif os.path.isdir(music_dir):
            for ext in ("*.mp3", "*.wav", "*.m4a", "*.ogg"):
                songs.extend(glob.glob(os.path.join(music_dir, ext)))
        if songs:
            try:
                music = AudioFileClip(random.choice(songs))
                if music.duration < duration:
                    reps = int(duration // music.duration) + 1
                    music = concatenate_audio_loop(music, reps)
                music = music.subclipped(0, duration)
                music = _scale_volume(music, vol)
                tracks.append(music)
            except Exception as e:
                print(f"[assemble] music skipped: {e}")
    if sfx_times:
        cam = get_camera_cfg(cfg)
        if cam.get("sfx", True):
            try:
                tracks.extend(whip_sfx_clips(sfx_times, duration, float(cam.get("sfx_volume", 0.4))))
            except Exception as e:
                print(f"[assemble] whip sfx skipped: {e}")
    if len(tracks) == 1:
        return voice
    # CompositeAudioClip.close() does not close its members either, so the voice
    # and music readers have to be carried out for the caller to release.
    mixed = CompositeAudioClip(tracks)
    mixed.source_clips = tracks
    return mixed


def concatenate_audio_loop(clip, reps):
    from moviepy import concatenate_audioclips

    return concatenate_audioclips([clip] * reps)


def _scale_volume(clip, factor):
    # Try the convenience method first, fall back to the effect.
    try:
        return clip.with_volume_scaled(factor)
    except Exception:
        from moviepy.audio.fx import MultiplyVolume

        return clip.with_effects([MultiplyVolume(factor)])


# -------------------------------------------------------------------- public ---
def build_video(cfg, voice_path, segments, words, duration, out_path, on_progress=None,
                background_path=None, music_path=None, scene_clips=None, graphics=None):
    def log(msg):
        if on_progress:
            on_progress(msg)
        else:
            print(msg)

    W, H = cfg["video"]["width"], cfg["video"]["height"]
    fps = cfg["video"]["fps"]

    # graphic-reveal times drive the camera zoom impulses and the white flashes
    reveal_times = [float(g["start"]) for g in (graphics or [])]

    if scene_clips:
        log(f"Building background from {len(scene_clips)} AI scenes…")
        bg = _scene_background(cfg, scene_clips, duration)
    else:
        log("Loading background gameplay…")
        bg = _background(cfg, duration, background_path=background_path,
                         segments=segments, impulse_times=reveal_times)

    # whip boundary times must be read before grading wraps the clip
    sfx_times = getattr(bg, "whip_times", None)
    bg = apply_grade(bg, cfg, fps)

    log("Placing character avatars…")
    avatars = _avatar_clips(cfg, segments)

    log("Rendering captions…")
    captions = _caption_clips(cfg, words)

    log("Animating motion graphics…")
    motion_clips = build_motion_clips(cfg, graphics, (W, H), fps)
    flashes = build_flash_clips(cfg, reveal_times, (W, H), fps, duration)
    overlays = []
    progress = build_progress_clip(cfg, duration, (W, H), fps)
    if progress is not None:
        overlays.append(progress)

    log("Compositing layers…")
    final = CompositeVideoClip(
        [bg, *avatars, *flashes, *motion_clips, *captions, *overlays], size=(W, H)
    ).with_duration(duration)

    log("Mixing audio…")
    audio = _build_audio(cfg, voice_path, duration, music_path=music_path, sfx_times=sfx_times)
    final = final.with_audio(audio)

    try:
        log("Encoding MP4 (this is the slow part)…")
        crf = str(cfg["video"].get("crf", 18))
        final.write_videofile(
            out_path,
            fps=fps,
            codec="libx264",
            audio_codec="aac",
            preset=cfg["video"].get("x264_preset", "medium"),
            threads=os.cpu_count() or 4,
            # CRF instead of MoviePy's default low bitrate — the main quality lever
            ffmpeg_params=["-crf", crf, "-pix_fmt", "yuv420p"],
            logger=None,
        )
    finally:
        # Close every clip so ffmpeg reader/writer handles are released.
        # On Windows an open handle keeps the intermediate .wav locked, so the
        # caller's os.remove() would silently fail; leaked handles also pile up
        # across repeated generations in the long-running Flask server.
        #
        # `source_clips` is what actually matters here: MoviePy's
        # CompositeVideoClip/CompositeAudioClip.close() deliberately leave their
        # member clips alone, and the base Clip.close() is a no-op, so closing
        # `bg`/`audio` released NOTHING. The readers opened per scene in
        # _scene_background (one ffmpeg subprocess each) leaked once per render.
        sources = list(getattr(bg, "source_clips", ())) + list(
            getattr(audio, "source_clips", ())
        )
        for clip in (final, audio, bg, *avatars, *captions, *motion_clips,
                     *flashes, *overlays, *sources):
            try:
                clip.close()
            except Exception:
                pass
    return out_path
