"""Asset sourcing: pick/download a background clip for a video.

Fallback chain (every network failure falls through to the next tier):
  1. Channel gameplay prefs — local files or yt-dlp cache of asset_prefs.gameplay_urls
  2. Topic-matched stock — Pexels (portrait search), then Pixabay; cached + attributed
  3. None — assemble.py then uses its normal random pick from assets/gameplay/

Free API keys via env: PEXELS_API_KEY, PIXABAY_API_KEY. Missing key = tier skipped.
"""
import glob
import hashlib
import logging
import os
import random
import re
import subprocess
import sys

import requests

from pipeline.config import abspath
from studio import models

log = logging.getLogger("studio.assets")

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "is",
    "are", "was", "were", "that", "this", "it", "you", "your", "my", "at", "by",
    "from", "about", "into", "how", "why", "what", "when", "who", "which",
    "der", "die", "das", "und", "oder", "ein", "eine", "mit", "von", "für",
    "shorts", "brainrot", "fyp", "viral", "video",
}

VIDEO_EXTS = ("*.mp4", "*.mov", "*.mkv", "*.webm")


def _studio_cfg(cfg, key, default):
    return (cfg.get("studio") or {}).get(key, default)


def extract_keywords(topic, script, max_kw=4):
    """script tags first, else topic minus stopwords."""
    words = []
    for tag in (script or {}).get("tags") or []:
        for w in re.findall(r"\w+", tag.lower()):
            if w not in STOPWORDS and len(w) > 2 and w not in words:
                words.append(w)
    if not words:
        for w in re.findall(r"\w+", (topic or "").lower()):
            if w not in STOPWORDS and len(w) > 2 and w not in words:
                words.append(w)
    return words[:max_kw]


# ------------------------------------------------------------- tier 1: yt-dlp --
def _channel_gameplay_dir(channel, cfg):
    return abspath(cfg, os.path.join(cfg["paths"]["gameplay_dir"], f"ch{channel['id']}"))


def _local_clips(directory):
    clips = []
    for ext in VIDEO_EXTS:
        clips.extend(glob.glob(os.path.join(directory, ext)))
    return clips


def _ytdlp_fetch(url, out_dir, max_minutes=8):
    """Download one gameplay clip with yt-dlp (module invocation, uses the venv).

    Only the first `max_minutes` are fetched (a background loop needs no more) and
    h264/mp4 is preferred so the file stays small and MoviePy-friendly — without a
    section cap a single video can be 1-2 GB.
    """
    os.makedirs(out_dir, exist_ok=True)
    stem = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    existing = [
        f for f in glob.glob(os.path.join(out_dir, stem + ".*"))
        if not f.endswith(".part")
    ]
    if existing:
        return existing[0]
    outtmpl = os.path.join(out_dir, stem + ".%(ext)s")
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "-f", "bv*[height<=1080][vcodec^=avc1]/b[height<=1080][vcodec^=avc1]/"
              "bv*[height<=1080]/b[height<=1080]/b",
        "--download-sections", f"*0:00-{max_minutes}:00",
        "--force-keyframes-at-cuts",
        "--no-playlist", "--remux-video", "mp4",
        "-o", outtmpl, url,
    ]
    subprocess.run(cmd, check=True, capture_output=True, timeout=900)
    files = [
        f for f in glob.glob(os.path.join(out_dir, stem + ".*"))
        if not f.endswith(".part")
    ]
    return files[0] if files else None


def _from_channel_prefs(channel, cfg):
    prefs = channel.get("asset_prefs") or {}
    urls = prefs.get("gameplay_urls") or []
    ch_dir = _channel_gameplay_dir(channel, cfg)

    cached = _local_clips(ch_dir) if os.path.isdir(ch_dir) else []
    if cached:
        return random.choice(cached), None
    max_minutes = int(_studio_cfg(cfg, "gameplay_max_minutes", 8))
    for url in urls:
        try:
            path = _ytdlp_fetch(url, ch_dir, max_minutes=max_minutes)
            if path:
                size = os.path.getsize(path)
                asset_id = models.record_asset(
                    "gameplay", "ytdlp", path, source_url=url,
                    keywords="gameplay", size_bytes=size,
                )
                return path, asset_id
        except Exception as e:
            log.warning("yt-dlp failed for %s: %s", url, e)
    return None, None


# -------------------------------------------------------------- tier 2: stock --
def _cache_dir(cfg, source):
    d = os.path.join(cfg["_root"], "assets", "cache", source)
    os.makedirs(d, exist_ok=True)
    return d


def _cached_stock(keywords):
    """Reuse a previously downloaded stock clip whose keywords overlap."""
    for row in models.list_assets(kind="stock"):
        if not os.path.exists(row["path"]):
            models.delete_asset(row["id"])
            continue
        asset_kw = set((row["keywords"] or "").split(","))
        if asset_kw & set(keywords):
            return row["path"], row["id"]
    return None, None


def _download(url, dest, timeout=180):
    """Download `url` to `dest`, atomically.

    The write goes to a `.part` sibling and is renamed into place only after the
    stream completes. Writing straight to `dest` meant a dropped connection, a
    full disk or a killed process left a TRUNCATED mp4 sitting at the final
    path — and every caller here guards with `if not os.path.exists(dest)`, so
    that stump was then reused as the background clip on every subsequent
    render, and `record_asset` stored its partial size as the real one. A
    corrupt background that survives restarts is much worse than a failed
    download the next run retries.
    """
    from pipeline.atomicio import atomic_write_bytes

    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        atomic_write_bytes(dest, r.iter_content(chunk_size=1 << 16))
    return dest


def _search_pexels(query, cfg):
    key = os.environ.get("PEXELS_API_KEY")
    if not key:
        return None
    min_secs = int((cfg.get("assets") or {}).get("min_stock_seconds", 6))
    resp = requests.get(
        "https://api.pexels.com/videos/search",
        headers={"Authorization": key},
        params={"query": query, "orientation": "portrait", "per_page": 15},
        timeout=30,
    )
    resp.raise_for_status()
    videos = resp.json().get("videos") or []
    # Prefer clips at least min_secs long so a short loop doesn't repeat visibly;
    # among those, longest first (fewer loops), then keep API relevance order.
    videos.sort(key=lambda v: (v.get("duration", 0) >= min_secs, v.get("duration", 0)),
                reverse=True)
    for vid in videos:
        files = [
            f for f in vid.get("video_files") or []
            if f.get("file_type") == "video/mp4"
            and (f.get("height") or 0) >= 1080
            and (f.get("height") or 0) >= (f.get("width") or 0)
        ]
        if not files:
            continue
        files.sort(key=lambda f: f.get("height") or 0)
        chosen = files[0]
        dest = os.path.join(_cache_dir(cfg, "pexels"), f"{vid['id']}.mp4")
        if not os.path.exists(dest):
            _download(chosen["link"], dest)
        attribution = f"Video by {vid.get('user', {}).get('name', '?')} on Pexels ({vid.get('url', '')})"
        return dest, vid.get("url"), attribution
    return None


def _search_pixabay(query, cfg):
    key = os.environ.get("PIXABAY_API_KEY")
    if not key:
        return None
    min_secs = int((cfg.get("assets") or {}).get("min_stock_seconds", 6))
    resp = requests.get(
        "https://pixabay.com/api/videos/",
        params={"key": key, "q": query, "per_page": 20, "safesearch": "true"},
        timeout=30,
    )
    resp.raise_for_status()
    hits = resp.json().get("hits") or []

    def _portrait(h):
        v = h.get("videos", {}).get("large", {})
        return v.get("height", 0) > v.get("width", 0)

    # portrait first, then long-enough, then longest
    hits.sort(key=lambda h: (_portrait(h), h.get("duration", 0) >= min_secs,
                             h.get("duration", 0)), reverse=True)
    for hit in hits:
        variants = hit.get("videos") or {}
        chosen = variants.get("large") or variants.get("medium") or variants.get("small")
        if not chosen or not chosen.get("url"):
            continue
        dest = os.path.join(_cache_dir(cfg, "pixabay"), f"{hit['id']}.mp4")
        if not os.path.exists(dest):
            _download(chosen["url"], dest)
        attribution = f"Video by {hit.get('user', '?')} on Pixabay ({hit.get('pageURL', '')})"
        return dest, hit.get("pageURL"), attribution
    return None


def _from_stock(keywords, cfg):
    if not keywords:
        return None, None
    path, asset_id = _cached_stock(keywords)
    if path:
        models.touch_asset(asset_id)
        return path, asset_id

    bias = ((cfg.get("assets") or {}).get("stock_query_bias") or "").strip()
    query = " ".join(keywords[:3] + ([bias] if bias else []))
    for fn, source in ((_search_pexels, "pexels"), (_search_pixabay, "pixabay")):
        try:
            found = fn(query, cfg)
        except Exception as e:
            log.warning("%s search failed: %s", source, e)
            found = None
        if found:
            path, source_url, attribution = found
            asset_id = models.record_asset(
                "stock", source, path, source_url=source_url,
                keywords=",".join(keywords), attribution=attribution,
                size_bytes=os.path.getsize(path),
            )
            return path, asset_id
    return None, None


# ---------------------------------------------------------------- cache prune --
def prune_cache(cfg):
    """Delete oldest downloaded assets when the cache exceeds max_cache_gb."""
    max_bytes = float(_studio_cfg(cfg, "max_cache_gb", 5)) * (1 << 30)
    rows = [r for r in models.list_assets()
            if r["source"] in ("pexels", "pixabay", "ytdlp", "comfyui")]
    total = 0
    for r in rows:
        if os.path.exists(r["path"]):
            total += r.get("size_bytes") or os.path.getsize(r["path"])
    if total <= max_bytes:
        return
    rows.sort(key=lambda r: r.get("last_used_at") or "")  # oldest first
    for r in rows:
        if total <= max_bytes:
            break
        try:
            if os.path.exists(r["path"]):
                total -= r.get("size_bytes") or os.path.getsize(r["path"])
                os.remove(r["path"])
            models.delete_asset(r["id"])
        except OSError as e:
            log.warning("could not prune %s: %s", r["path"], e)


# --------------------------------------------------------------------- public --
def get_background(channel, topic, script, cfg):
    """Return (path, asset_id) for the background clip.

    path may also be assemble.GENERATED_BG (locally generated color animation)
    or None — then assemble.py picks a random local gameplay clip itself.
    Tier order: channel asset_prefs.source_order, else cfg["assets"] (which a
    style preset like 'kids' overrides to stock+generated — no gameplay).
    """
    default_order = (cfg.get("assets") or {}).get(
        "source_order", ["gameplay", "stock", "generated"]
    )
    order = ((channel or {}).get("asset_prefs") or {}).get("source_order") or default_order
    keywords = extract_keywords(topic, script)

    for tier in order:
        if tier == "gameplay":
            if channel:
                path, asset_id = _from_channel_prefs(channel, cfg)
                if path:
                    return path, asset_id
            # No channel (quick-generate / CLI): honor real local gameplay clips
            # by letting assemble pick one itself (None sentinel). Ignore the
            # bundled _placeholder so we fall through to stock/generated instead
            # of the boring fractal when no real clip exists.
            else:
                local = _local_clips(abspath(cfg, cfg["paths"]["gameplay_dir"]))
                if any(not os.path.basename(c).startswith("_") for c in local):
                    return None, None
        elif tier == "stock":
            path, asset_id = _from_stock(keywords, cfg)
            if path:
                return path, asset_id
        elif tier == "aivideo":
            path, asset_id = _from_aivideo(topic, script, cfg)
            if path:
                return path, asset_id
        elif tier == "generated":
            from pipeline.assemble import GENERATED_BG

            return GENERATED_BG, None
    return None, None


def _from_aivideo(topic, script, cfg):
    """Locally generated AI clip via ComfyUI (LTX-Video). Best-effort: returns
    (None, None) when ComfyUI is disabled/unreachable so the chain falls through."""
    from studio import comfy

    if not comfy.is_available(cfg):
        return None, None
    import hashlib

    stem = hashlib.sha1((comfy.build_prompt(cfg, topic, script)).encode("utf-8")).hexdigest()[:12]
    out = os.path.join(_cache_dir(cfg, "aivideo"), stem + ".mp4")
    if os.path.exists(out):
        rows = [r for r in models.list_assets(kind="aivideo") if r["path"] == out]
        return out, (rows[0]["id"] if rows else None)
    path = comfy.generate_clip(cfg, topic, script, out)
    if not path:
        return None, None
    asset_id = models.record_asset(
        "aivideo", "comfyui", path, keywords=",".join(extract_keywords(topic, script)),
        attribution="Locally generated (ComfyUI/LTX-Video)",
        size_bytes=os.path.getsize(path),
    )
    return path, asset_id
