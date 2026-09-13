"""
sfx.py — Tiny synthesized sound effects, no asset files needed.

Currently one effect: a filtered-noise "whoosh" played under each whip-pan
camera transition. Synthesized once per process into a temp wav and reused.
"""
import math
import os
import struct
import tempfile
import wave

_SR = 22050
_WHOOSH_SECONDS = 0.35
_whoosh_path = None


def whoosh_wav():
    """Synthesize (once) and return the path of the whoosh sample."""
    global _whoosh_path
    if _whoosh_path and os.path.exists(_whoosh_path):
        return _whoosh_path
    import numpy as np

    n = int(_SR * _WHOOSH_SECONDS)
    rng = np.random.default_rng(23)
    noise = rng.standard_normal(n + 64)
    # low-pass the noise so it reads as air, not hiss
    kernel = np.hanning(64)
    kernel /= kernel.sum()
    body = np.convolve(noise, kernel, mode="valid")[:n]
    t = np.linspace(0.0, 1.0, n)
    env = (t ** 0.6) * ((1.0 - t) ** 1.6)
    env /= env.max()
    sample = body / np.max(np.abs(body)) * env

    fd, path = tempfile.mkstemp(prefix="brainrot_whoosh_", suffix=".wav")
    os.close(fd)
    with wave.open(path, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(_SR)
        w.writeframes(b"".join(
            struct.pack("<h", int(max(-1.0, min(1.0, s)) * 32000)) for s in sample
        ))
    _whoosh_path = path
    return path


def whip_sfx_clips(times, duration, volume):
    """AudioFileClips of the whoosh, one per whip transition, volume-scaled.

    Each starts slightly before its boundary so the swell peaks on the cut.
    """
    if not times or volume <= 0:
        return []
    from moviepy import AudioFileClip

    path = whoosh_wav()
    clips = []
    for t in times:
        start = max(0.0, float(t) - _WHOOSH_SECONDS / 2.0)
        if start >= duration:
            continue
        clip = AudioFileClip(path).with_start(start)
        try:
            clip = clip.with_volume_scaled(volume)
        except Exception:
            from moviepy.audio.fx import MultiplyVolume

            clip = clip.with_effects([MultiplyVolume(volume)])
        clips.append(clip)
    return clips
