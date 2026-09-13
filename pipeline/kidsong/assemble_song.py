"""
assemble_song.py — Ken Burns / animated-clip slideshow assembler for
children's-song videos.

Layers (bottom to top):
  1. Per-verse scene media: either a still image with a slow Ken Burns zoom,
     or an animated video clip (looped/subclipped to fill the verse window),
     covering the 1080x1920 frame
  2. Word-by-word captions burned in (reused from pipeline.assemble)
  3. Audio = the sung voice track (+ optional quiet background music bed)

Built for MoviePy 2.x (pip install "moviepy>=2.1"), mirroring the idioms used
in pipeline/assemble.py (`.resized/.subclipped/.with_start/.with_duration/
.with_position`, the try/finally close pattern, `_scale_volume`).
"""
import os

from moviepy import (
    AudioFileClip,
    CompositeAudioClip,
    CompositeVideoClip,
    ImageClip,
    VideoFileClip,
    concatenate_audioclips,
    vfx,
)

from pipeline.assemble import _caption_clips

_VIDEO_EXTS = (".mp4", ".webm", ".mov", ".mkv")


# ------------------------------------------------------------- ken burns ---
_KB_LO = 1.02
_KB_HI = 1.10


def _kb_scale_fn(zoom_in, dur):
    """Return a time -> scale-factor callable for the slow Ken Burns drift.

    zoom_in=True drifts _KB_LO -> _KB_HI (slow push-in), False drifts the
    other way (slow pull-back). `dur` is the clip's own (local) duration.
    """
    span = _KB_HI - _KB_LO

    def f(t):
        frac = min(max(t / dur, 0.0), 1.0) if dur > 0 else 0.0
        return (_KB_LO + span * frac) if zoom_in else (_KB_HI - span * frac)

    return f


def _cover_size(img_w, img_h, W, H):
    """Like assemble._cover_crop's scale math: smallest size that covers WxH."""
    scale = max(W / img_w, H / img_h)
    return max(1, round(img_w * scale)), max(1, round(img_h * scale))


def _is_video(path):
    return os.path.splitext(path)[1].lower() in _VIDEO_EXTS


def _resolve_media(scene_media):
    """Fill None entries with the nearest non-None neighbor.

    Earlier Nones borrow from the most recent preceding element; leading
    Nones (nothing precedes them yet) borrow from the first later element.
    Raises if every entry is None.
    """
    n = len(scene_media)
    if n == 0:
        raise ValueError("scene_media must not be empty")

    resolved = list(scene_media)
    last = None
    for i in range(n):
        if resolved[i] is not None:
            last = resolved[i]
        elif last is not None:
            resolved[i] = last

    nxt = None
    for i in range(n - 1, -1, -1):
        if resolved[i] is not None:
            nxt = resolved[i]
        else:
            resolved[i] = nxt

    if all(m is None for m in resolved):
        raise ValueError("scene_media: every entry is None (no usable images/videos)")
    return resolved


# --------------------------------------------------------------- scenes ---
def _scene_clips(cfg, scene_media, verse_times, duration):
    """One Ken-Burns/animated clip per verse, timed to the verse boundaries.

    Each element of `scene_media` may be a still image path, a video path
    (.mp4/.webm/.mov/.mkv), or None (filled in from a neighboring element by
    `_resolve_media`). Still images get the existing slow Ken Burns zoom;
    video clips are cover-scaled+centered and looped (if shorter than the
    verse window) or subclipped (if longer) to exactly fill it — no extra
    zoom is applied on top of video.

    Verse 0 always starts at 0 (covers the lead-in even if the first sung
    word starts later); each verse clip lasts until the next verse's start,
    and the last verse extends to `duration`. Adjacent clips get a small
    CrossFadeIn overlap for a soft transition (falls back to a hard cut if
    there isn't enough room).

    Returns (clips, video_sources) — `video_sources` holds the raw
    VideoFileClip readers opened for video elements, so the caller can
    close them explicitly (transformed/resized clips lose the `.close()`
    hook to the underlying reader).
    """
    W, H = cfg["video"]["width"], cfg["video"]["height"]
    n = len(verse_times)
    if n == 0:
        raise ValueError("verse_times must not be empty")
    if not scene_media:
        raise ValueError("scene_media must not be empty")

    scene_media = _resolve_media(scene_media)

    starts = [0.0 if i == 0 else verse_times[i][0] for i in range(n)]
    ends = [starts[i + 1] for i in range(n - 1)] + [duration]

    fade = 0.35
    clips = []
    video_sources = []
    for i in range(n):
        media = scene_media[min(i, len(scene_media) - 1)]
        seg_start, seg_end = starts[i], ends[i]

        overlap = fade if i > 0 else 0.0
        overlap = max(0.0, min(overlap, (seg_end - seg_start) * 0.4, seg_start))
        disp_start = seg_start - overlap
        disp_dur = seg_end - disp_start
        if disp_dur <= 0:
            disp_dur = max(seg_end - seg_start, 0.05)
            disp_start = seg_end - disp_dur
            overlap = 0.0

        if _is_video(media):
            raw = VideoFileClip(media).without_audio()
            video_sources.append(raw)
            cover_w, cover_h = _cover_size(raw.w, raw.h, W, H)
            base = raw.resized((cover_w, cover_h)).with_position("center")
            if base.duration < disp_dur:
                base = base.with_effects([vfx.Loop(duration=disp_dur)])
            else:
                base = base.subclipped(0, disp_dur)
            clip = base.with_duration(disp_dur)
        else:
            img = ImageClip(media)
            cover_w, cover_h = _cover_size(img.w, img.h, W, H)
            base = img.resized((cover_w, cover_h))
            kb = _kb_scale_fn(zoom_in=(i % 2 == 0), dur=disp_dur)
            clip = base.resized(kb).with_position("center").with_duration(disp_dur)

        if overlap > 0:
            clip = clip.with_effects([vfx.CrossFadeIn(overlap)])
        clip = clip.with_start(disp_start)
        clips.append(clip)
    return clips, video_sources


# ---------------------------------------------------------------- audio ---
def _scale_volume(clip, factor):
    # Try the convenience method first, fall back to the effect.
    try:
        return clip.with_volume_scaled(factor)
    except Exception:
        from moviepy.audio.fx import MultiplyVolume

        return clip.with_effects([MultiplyVolume(factor)])


def _build_audio(cfg, voice_path, music_path, duration):
    voice = AudioFileClip(voice_path)
    vol = cfg.get("kidsong", {}).get("music_volume", 0.16)
    tracks = [voice]
    if music_path and vol and vol > 0 and os.path.exists(music_path):
        try:
            music = AudioFileClip(music_path)
            if music.duration < duration:
                reps = int(duration // music.duration) + 1
                music = concatenate_audioclips([music] * reps)
            music = music.subclipped(0, duration)
            music = _scale_volume(music, vol)
            tracks.append(music)
        except Exception as e:
            print(f"[assemble_song] music skipped: {e}")
    if len(tracks) == 1:
        return voice
    return CompositeAudioClip(tracks)


# --------------------------------------------------------------- public ---
def build_song_video(
    cfg,
    voice_path,
    music_path,
    scene_media,
    verse_times,
    words,
    duration,
    out_path,
    on_progress=None,
):
    def log(msg):
        if on_progress:
            on_progress(msg)
        else:
            print(msg)

    W, H = cfg["video"]["width"], cfg["video"]["height"]
    fps = cfg["video"]["fps"]

    log("Building scene slideshow…")
    scenes, video_sources = _scene_clips(cfg, scene_media, verse_times, duration)

    log("Rendering captions…")
    captions = _caption_clips(cfg, words)

    log("Compositing layers…")
    final = CompositeVideoClip([*scenes, *captions], size=(W, H)).with_duration(duration)

    log("Mixing audio…")
    audio = _build_audio(cfg, voice_path, music_path, duration)
    final = final.with_audio(audio)

    try:
        log("Encoding MP4 (this is the slow part)…")
        final.write_videofile(
            out_path,
            fps=fps,
            codec="libx264",
            audio_codec="aac",
            preset="medium",
            threads=os.cpu_count() or 4,
            logger=None,
        )
    finally:
        # Close every clip so ffmpeg reader/writer handles are released.
        # On Windows an open handle keeps the intermediate .wav locked, so the
        # caller's os.remove() would silently fail; leaked handles also pile up
        # across repeated generations in the long-running Flask server.
        for clip in (final, audio, *scenes, *captions, *video_sources):
            try:
                clip.close()
            except Exception:
                pass

    # The legacy slideshow path publishes straight to out_path, so the
    # disclosure is stamped here rather than on a .part sibling. Tagging is a
    # stream copy through its own temp file, so an interrupted tag leaves the
    # untagged-but-complete episode in place.
    from pipeline import ai_disclosure

    ai_disclosure.tag_file(out_path, cfg, log)
    return out_path


# ------------------------------------------------------------------ test ---
if __name__ == "__main__":
    import math
    import wave

    import numpy as np
    from PIL import Image, ImageDraw

    from pipeline.config import abspath, load_config

    cfg = load_config()
    out_dir = abspath(cfg, cfg["paths"]["output_dir"])
    os.makedirs(out_dir, exist_ok=True)
    scratch = os.path.join(out_dir, "_kidsong_assemble_test")
    os.makedirs(scratch, exist_ok=True)

    # --- two synthetic 832x1216 scene PNGs (solid color + big circle) ---
    scene_pngs = []
    colors = [(220, 60, 60), (60, 120, 220)]
    for i, color in enumerate(colors):
        img = Image.new("RGB", (832, 1216), color)
        d = ImageDraw.Draw(img)
        cx, cy, r = 416, 608, 300
        d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(255, 255, 255))
        path = os.path.join(scratch, f"scene_{i}.png")
        img.save(path)
        scene_pngs.append(path)

    # --- fake 6-second 440Hz sine voice track ---
    voice_path = os.path.join(scratch, "voice.wav")
    sr = 24000
    dur = 6.0
    n_samples = int(sr * dur)
    t = np.linspace(0, dur, n_samples, endpoint=False)
    tone = (0.3 * np.sin(2 * math.pi * 440 * t) * 32767).astype(np.int16)
    with wave.open(voice_path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(tone.tobytes())

    words = [{"word": "LA", "start": 0.5 * k, "end": 0.5 * k + 0.4} for k in range(10)]
    verse_times = [(0, 3), (3, 6)]
    out_path = os.path.join(out_dir, "kidsong_assemble_test.mp4")

    result = build_song_video(
        cfg,
        voice_path=voice_path,
        music_path=None,
        scene_media=scene_pngs,
        verse_times=verse_times,
        words=words,
        duration=dur,
        out_path=out_path,
    )
    print(f"[assemble_song] wrote {result}")

    # --- second test: mixed media, one entry is an animated (looping) clip ---
    # A tiny synthetic 2-second, 480x832 mp4 (a rectangle sliding across 20
    # frames) that must be looped by _scene_clips to fill its 3s verse window.
    from moviepy import ImageSequenceClip

    n_frames = 20
    clip_fps = n_frames / 2.0  # 2-second source clip
    frames = []
    for k in range(n_frames):
        frame = Image.new("RGB", (480, 832), (30, 30, 30))
        d = ImageDraw.Draw(frame)
        x = int((k / n_frames) * 400)
        d.rectangle((x, 380, x + 80, 460), fill=(240, 200, 40))
        frames.append(np.array(frame))
    video_path = os.path.join(scratch, "scene_clip.mp4")
    seq = ImageSequenceClip(frames, fps=clip_fps)
    seq.write_videofile(video_path, fps=clip_fps, codec="libx264", audio=False, logger=None)
    seq.close()

    scene_media = [video_path, scene_pngs[1]]
    out_path2 = os.path.join(out_dir, "kidsong_assemble_test2.mp4")

    result2 = build_song_video(
        cfg,
        voice_path=voice_path,
        music_path=None,
        scene_media=scene_media,
        verse_times=verse_times,
        words=words,
        duration=dur,
        out_path=out_path2,
    )
    print(f"[assemble_song] wrote {result2}")
