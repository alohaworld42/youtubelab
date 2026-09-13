"""
svm.py — client for short-video-maker (gyoridavid/short-video-maker).

An alternative render ENGINE to the built-in MoviePy pipeline: a Remotion-based
renderer that does scene-based Pexels B-roll, Kokoro TTS and animated captions,
exposed as an HTTP API (default :3123, run as a Docker container — see
scripts/start-svm.ps1). The studio stays the brain (channels, ideas, schedule,
review, upload); svm is just one way to turn a script into an mp4.

If the server isn't reachable, render_via_svm() raises so the caller can fall
back to the MoviePy engine.
"""
import logging
import os
import re
import time

import requests

log = logging.getLogger("studio.svm")

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "is",
    "are", "was", "were", "that", "this", "it", "you", "your", "my", "at", "by",
    "from", "about", "into", "how", "why", "what", "when", "who", "which", "we",
    "der", "die", "das", "und", "oder", "ein", "eine", "mit", "von", "für",
}

# Map our three speakers to Kokoro voices per style handled by the caller;
# these are safe defaults known to exist in short-video-maker.
DEFAULT_VOICE = "am_adam"
KIDS_VOICE = "af_bella"


def _conf(cfg):
    return cfg.get("svm") or {}


def host(cfg):
    return _conf(cfg).get("host", "http://127.0.0.1:3123")


def is_available(cfg):
    if not _conf(cfg).get("enabled"):
        return False
    try:
        r = requests.get(f"{host(cfg)}/api/voices", timeout=4)
        return r.status_code == 200
    except Exception:
        return False


def _keywords(text, fallback_tags, n=3):
    words = []
    for w in re.findall(r"\w+", (text or "").lower()):
        if w not in STOPWORDS and len(w) > 2 and w not in words:
            words.append(w)
    if len(words) < 2:
        for t in fallback_tags or []:
            tw = re.sub(r"[^\w]", "", t.lower())
            if tw and tw not in words:
                words.append(tw)
    return words[:n] or ["abstract", "background"]


def script_to_scenes(script):
    """Map our {lines:[{speaker,text}], tags} into svm scenes with B-roll terms."""
    tags = script.get("tags") or []
    scenes = []
    for ln in script.get("lines") or []:
        text = (ln.get("text") or "").strip()
        if not text:
            continue
        scenes.append({"text": text, "searchTerms": _keywords(text, tags)})
    if not scenes:
        scenes = [{"text": script.get("title") or "Hello", "searchTerms": ["abstract"]}]
    return scenes


def build_config(cfg, style=None):
    conf = _conf(cfg)
    orientation = conf.get("orientation", "portrait")
    voice = conf.get("voice") or (KIDS_VOICE if style == "kids" else DEFAULT_VOICE)
    music = conf.get("music") or ("happy" if style == "kids" else "chill")
    music_volume = conf.get("music_volume") or ("high" if style == "kids" else "low")
    return {
        "paddingBack": int(conf.get("padding_back_ms", 1500)),
        "captionPosition": conf.get("caption_position", "center"),
        "captionBackgroundColor": conf.get("caption_bg", "#ff5ea0" if style == "kids" else "#0b0b12"),
        "voice": voice,
        "music": music,
        "musicVolume": music_volume,
        "orientation": orientation,
    }


def render_via_svm(cfg, script, out_path, style=None, on_progress=None):
    """Render a script into out_path via short-video-maker. Returns out_path.

    Raises RuntimeError if the server is unreachable or generation fails, so the
    scheduler can fall back to the MoviePy engine.
    """
    def step(msg):
        if on_progress:
            on_progress(msg)

    h = host(cfg)
    if not is_available(cfg):
        raise RuntimeError("short-video-maker not reachable (is the container running?)")

    payload = {"scenes": script_to_scenes(script), "config": build_config(cfg, style)}
    step("Sending scenes to short-video-maker…")
    resp = requests.post(f"{h}/api/short-video", json=payload, timeout=30)
    resp.raise_for_status()
    video_id = resp.json().get("videoId")
    if not video_id:
        raise RuntimeError(f"short-video-maker returned no videoId: {resp.text[:200]}")

    deadline = time.time() + int(_conf(cfg).get("timeout_seconds", 900))
    last = None
    while time.time() < deadline:
        try:
            st = requests.get(f"{h}/api/short-video/{video_id}/status", timeout=10).json()
        except Exception:
            time.sleep(4)
            continue
        status = st.get("status")
        if status != last:
            step(f"short-video-maker: {status}")
            last = status
        if status == "ready":
            break
        if status == "error":
            raise RuntimeError(f"short-video-maker render error for {video_id}")
        time.sleep(5)
    else:
        raise RuntimeError("short-video-maker timed out")

    step("Downloading rendered video…")
    data = requests.get(f"{h}/api/short-video/{video_id}", timeout=180)
    data.raise_for_status()
    from pipeline.atomicio import atomic_write_bytes

    atomic_write_bytes(out_path, [data.content])
    return out_path
