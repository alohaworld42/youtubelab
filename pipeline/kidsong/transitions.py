"""
kidsong.transitions — generated shot-to-shot transitions and shot chaining.

Today's baseline is hard cuts: `kidsong.edit.build_cut_list` snaps shot
boundaries to the beat grid and `assemble()` splices each shot's own t2v/i2v
render straight against the next, with no bridging frame. This module adds
two OPT-IN techniques on top of that baseline, both designed to fail soft
back to the exact same hard cut if anything goes wrong:

  1. SHOT CHAINING (`shot_chaining_enabled`) is free — it costs no extra
     render. Instead of rendering shot N+1 as fresh text-to-video, we feed it
     the LAST FRAME of shot N as its i2v conditioning image. This both saves
     a render and fights the cast/style drift that independent t2v draws
     produce shot-to-shot (faces, colors, proportions wandering). Chaining is
     restricted to runs of shots that stay in the same scene/location — see
     `should_chain` — so a chain never quietly bridges a scene change; it
     stops the moment the location (or, if location isn't tracked, the verse)
     changes, and never chains across a `reuse_of` shot (those already point
     at existing footage, there is nothing fresh to condition).

  2. GENERATED TRANSITIONS (`transitions_enabled`) cost one EXTRA render per
     transition: a short first-last-frame (FLF2V) clip bridging the true
     final frame of one shot to the true first frame of the next, rendered
     against the `ltx23_flf2v_toon` ComfyUI graph (see `pipeline.kidsong.comfy`).
     Because each one is a real render, they are deliberately restricted to
     verse/section boundaries (`transitions_at == "verse"`, the only
     non-"none" mode today) rather than every cut, and capped by
     `transitions_max_per_episode` — spent on the boundaries that matter most
     (spread evenly across the episode) rather than burned on interior cuts
     within a verse.

Both techniques degrade gracefully: `generate_transitions` never lets a
single failed render abort the batch — it logs and simply omits that index,
so the caller's normal hard cut plays instead. Nothing here calls into
ComfyUI at import time; `moviepy`/`PIL`/`requests`-touching work is confined
to function bodies (lazy imports), so `import pipeline.kidsong.transitions`
stays cheap, torch-free and safe to run in tests with no GPU and no network.
"""
import os
import shutil
import subprocess


# ------------------------------------------------------------------- sigmas ---
# The sigma schedule FLF2V samples against. This is NOT cosmetic: it decides
# whether a transition is usable at all.
#
# `workflows/ltx23_flf2v_toon.json` authors the 3-step distilled schedule the
# rest of the channel renders with (t2v/i2v), which starts at 0.85. Both guides
# are pinned by a noise mask, so at 3 steps the FIRST and LAST frames do come
# out correct — but there are not enough steps left to actually *build* the
# frames in between, and the interior collapses into unrelated content.
#
# Measured on a real render (output/_flf2v_smoke/, guides taken from
# 20260720-092939-kidsong-rainbow-friends-color-fun s00->s01):
#   SIGMAS_DISTILLED_3STEP: frames 0 and 24 match their guides, but frames
#     ~3-18 (about 75% of a 25-frame clip) are an unrelated stone courtyard
#     with no children in it. Unusable.
#   SIGMAS_OFFICIAL_9STEP:  frames 0-12 hold the guide-A scene coherently,
#     frames 15-18 are a motion-blurred whip, frames 21-24 land guide B sharply.
#     Reads as a fast camera whip between the two shots. Usable.
#
# So FLF2V defaults to the official 9-step schedule below rather than inheriting
# the channel's distilled one. Override per-config with `transition_sigmas`;
# set it to None to fall back to whatever the workflow JSON authors.
SIGMAS_DISTILLED_3STEP = "0.85, 0.7250, 0.4219, 0.0"
SIGMAS_OFFICIAL_9STEP = (
    "1.0000, 0.9937, 0.9875, 0.9812, 0.9750, 0.9094, 0.7250, 0.4219, 0.0"
)


# ------------------------------------------------------------------ defaults ---
DEFAULTS = {
    "transitions_enabled": False,      # safe default: today's behaviour = hard cuts
    "transitions_at": "verse",         # "verse" = section boundaries only, or "none"
    "transition_frames": 25,           # ~1.0s at 24fps; must be 8*n+1
    "transition_guide_strength": 0.7,
    "transitions_max_per_episode": 4,
    "transition_sigmas": SIGMAS_OFFICIAL_9STEP,
    "shot_chaining_enabled": False,
    "shot_chain_max_len": 3,
}


def options(cfg):
    """DEFAULTS merged with any same-named overrides under cfg["kidsong"]."""
    merged = dict(DEFAULTS)
    kidsong_cfg = (cfg or {}).get("kidsong", {}) if isinstance(cfg, dict) else {}
    if isinstance(kidsong_cfg, dict):
        for key in DEFAULTS:
            if key in kidsong_cfg:
                merged[key] = kidsong_cfg[key]
    return merged


def snap_frames(n):
    """Snap `n` to the nearest valid LTX frame length (8*k + 1), k >= 1."""
    n = int(n)
    if n < 9:
        return 9
    k = round((n - 1) / 8.0)
    if k < 1:
        k = 1
    return 8 * k + 1


# --------------------------------------------------------------- frame grab ---
def _ffmpeg_available():
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _probe_duration_and_fps(path):
    """Return (duration_seconds, fps) for a video via ffprobe."""
    import json

    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=duration,r_frame_rate",
         "-show_entries", "format=duration",
         "-of", "json", path],
        capture_output=True, text=True, check=True,
    ).stdout
    data = json.loads(out)
    streams = data.get("streams") or [{}]
    stream = streams[0]

    duration = stream.get("duration")
    if duration is None:
        duration = (data.get("format") or {}).get("duration")
    duration = float(duration)

    rate = str(stream.get("r_frame_rate") or "24/1")
    num, _, den = rate.partition("/")
    try:
        fps = float(num) / float(den) if den else float(num)
    except (ValueError, ZeroDivisionError):
        fps = 24.0
    if not fps or fps <= 0:
        fps = 24.0
    return duration, fps


def extract_frame(video_path, position, out_png):
    """Grab the first or last frame of `video_path` and save it as a PNG.

    position: "first" or "last". "last" seeks to (duration - 1/fps) so it
    grabs the true final frame rather than landing one frame past the end
    (a common ffmpeg/moviepy off-by-one when seeking straight to `duration`).
    Uses ffmpeg via subprocess when available, else falls back to moviepy
    (lazily imported). Returns `out_png`. Raises on failure.
    """
    if position not in ("first", "last"):
        raise ValueError(f"position must be 'first' or 'last', got {position!r}")

    from pipeline.atomicio import atomic_path

    video_path = os.path.abspath(video_path)
    out_png = os.path.abspath(out_png)
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)

    # Written through a .part sibling. `out_png` is a GUIDE IMAGE: the FLF2V
    # first/last frame, a chain guide, or (via refs.harvest_reference) a cast
    # reference anchor. A killed ffmpeg left a truncated PNG at that path, and
    # every reader here checks only `getsize(...) > 0`, so the stump would be
    # staged into ComfyUI as if it were a real frame. Same discipline as
    # pipeline/atomicio.py's other callers.
    if _ffmpeg_available():
        try:
            with atomic_path(out_png) as tmp_png:
                if position == "first":
                    cmd = ["ffmpeg", "-y", "-v", "error", "-i", video_path,
                           "-vframes", "1", tmp_png]
                else:
                    duration, fps = _probe_duration_and_fps(video_path)
                    seek = max(0.0, duration - (1.0 / fps))
                    cmd = ["ffmpeg", "-y", "-v", "error", "-ss", f"{seek:.3f}",
                           "-i", video_path, "-vframes", "1", tmp_png]
                subprocess.run(cmd, check=True, capture_output=True)
                if not (os.path.exists(tmp_png) and os.path.getsize(tmp_png) > 0):
                    raise RuntimeError("ffmpeg produced no frame")
            return out_png
        except (subprocess.CalledProcessError, OSError, ValueError, RuntimeError,
                KeyError, IndexError, TypeError, FileNotFoundError):
            pass  # fall through to the moviepy fallback below

    # moviepy fallback (lazy import — keeps module import cheap/torch-free).
    from moviepy import VideoFileClip

    with atomic_path(out_png) as tmp_png:
        clip = VideoFileClip(video_path)
        try:
            if position == "first":
                t = 0.0
            else:
                fps = clip.fps or 24.0
                t = max(0.0, clip.duration - 1.0 / fps)
            clip.save_frame(tmp_png, t=t)
        finally:
            clip.close()
        if not (os.path.exists(tmp_png) and os.path.getsize(tmp_png) > 0):
            raise RuntimeError(f"Failed to extract {position} frame from {video_path}")
    return out_png


# ------------------------------------------------------------------- spread ---
def _spread_evenly(items, k):
    """Pick `k` items from `items`, spread across the sequence (endpoints
    included when k > 1) rather than just taking the first k."""
    n = len(items)
    if k <= 0 or n == 0:
        return []
    if k >= n:
        return list(items)
    if k == 1:
        return [items[n // 2]]

    step = (n - 1) / (k - 1)
    used = set()
    idxs = []
    for i in range(k):
        idx = round(i * step)
        while idx in used and idx < n - 1:
            idx += 1
        while idx in used and idx > 0:
            idx -= 1
        used.add(idx)
        idxs.append(idx)
    return [items[j] for j in sorted(set(idxs))]


# --------------------------------------------------------------------- plan ---
def plan_transitions(cut_list, shotlist, cfg):
    """Plan generated transitions at verse/section boundaries in `cut_list`.

    A transition is planned for cut index i (i > 0) whenever the verse of
    cut i's shot differs from the verse of cut i-1's shot, looking verse up
    from `shotlist` by shot id (accepts a bare list of shot dicts or
    {"shots": [...]})  — cuts whose shot id isn't found in the shotlist, or
    whose verse is unknown on either side, are skipped rather than treated
    as a boundary. Returns [] outright when transitions are disabled or
    `transitions_at == "none"`. The candidate list is truncated to
    `transitions_max_per_episode`, preferring evenly spread boundaries over
    just the first N.
    """
    opts = options(cfg)
    if not opts["transitions_enabled"] or opts["transitions_at"] == "none":
        return []
    if not cut_list or len(cut_list) < 2:
        return []

    shots = shotlist.get("shots") if isinstance(shotlist, dict) else shotlist
    by_id = {s["id"]: s for s in (shots or []) if isinstance(s, dict) and "id" in s}

    def verse_of(shot_id):
        s = by_id.get(shot_id)
        return s.get("verse") if s else None

    frames = snap_frames(opts["transition_frames"])

    candidates = []
    for i in range(1, len(cut_list)):
        prev_cut = cut_list[i - 1]
        cur_cut = cut_list[i]
        v_prev = verse_of(prev_cut.get("shot_id"))
        v_cur = verse_of(cur_cut.get("shot_id"))
        if v_prev is None or v_cur is None or v_prev == v_cur:
            continue
        candidates.append({
            "index": i,
            "prev_shot_id": prev_cut.get("shot_id"),
            "next_shot_id": cur_cut.get("shot_id"),
            "prev_src": prev_cut.get("src", prev_cut.get("shot_id")),
            "next_src": cur_cut.get("src", cur_cut.get("shot_id")),
            "frames": frames,
        })

    max_n = int(opts["transitions_max_per_episode"])
    if len(candidates) > max_n:
        candidates = _spread_evenly(candidates, max_n)
    return candidates


# ---------------------------------------------------------------- transition ---
def render_transition(client, first_png, last_png, prompt, seed, cfg, out_path):
    """Render one FLF2V transition bridging `first_png` -> `last_png`.

    Stages both PNGs into ComfyUI's input/ dir via `client.stage_input_image`,
    patches the "ltx23_flf2v_toon" graph (PROMPT, NEGATIVE if configured,
    SEED, WIDTH, HEIGHT, FRAMES, FIRST_IMAGE, LAST_IMAGE, GUIDE_FIRST,
    GUIDE_LAST, FILENAME_PREFIX, and SIGMAS unless `transition_sigmas` is
    None) and calls `client.render(...)`. Returns `out_path`.

    See SIGMAS_OFFICIAL_9STEP above for why the schedule is patched rather
    than inherited from the workflow JSON — at the channel's 3-step distilled
    schedule the guides land but the interior of the clip collapses.
    """
    opts = options(cfg)
    kidsong_cfg = (cfg or {}).get("kidsong", {}) if isinstance(cfg, dict) else {}

    # Shot dimensions come from kidsong.shot — the SAME source the t2v renders
    # use (kidsong.generate._shot_dims), so a transition can never be a
    # different shape than the shots it bridges. This used to read
    # kidsong.width/kidsong.height, keys that do not exist in any shipped
    # config, so it always fell through to a hardcoded portrait 512x896 while
    # the rest of the channel was moving to landscape.
    from pipeline.kidsong.generate import _shot_dims

    width, height = _shot_dims(cfg or {})
    frames = snap_frames(opts["transition_frames"])
    guide = float(opts["transition_guide_strength"])

    first_name = client.stage_input_image(first_png)
    last_name = client.stage_input_image(last_png)

    patches = {
        "PROMPT": prompt,
        "SEED": int(seed),
        "WIDTH": width,
        "HEIGHT": height,
        "FRAMES": frames,
        "FIRST_IMAGE": first_name,
        "LAST_IMAGE": last_name,
        "GUIDE_FIRST": guide,
        "GUIDE_LAST": guide,
        "FILENAME_PREFIX": "kidsong_transition",
    }
    negative = kidsong_cfg.get("negative") or kidsong_cfg.get("transition_negative") \
        if isinstance(kidsong_cfg, dict) else None
    if negative:
        patches["NEGATIVE"] = negative

    # Dict form: "SIGMAS" has no _PRIMARY_INPUT scalar mapping in
    # kidsong.comfy, and the graph's ManualSigmas input is named "sigmas".
    # None means "leave the workflow's own schedule alone".
    sigmas = opts.get("transition_sigmas")
    if sigmas:
        patches["SIGMAS"] = {"sigmas": str(sigmas)}

    client.render("ltx23_flf2v_toon", patches, out_path)
    return out_path


def generate_transitions(plan, renders, cfg, client=None, workdir=None, on_progress=None):
    """Render every transition in `plan`, skipping (not raising on) failures.

    renders: {shot_id: video_path} — same keying as `kidsong.edit.assemble`'s
             renders dict (i.e. by the cut's resolved `src`, see
             `plan_transitions`'s prev_src/next_src).
    Returns {index: transition_video_path}, one entry per transition that
    rendered successfully. A single transition's failure (missing render,
    frame-grab error, ComfyUI error, ...) is caught, reported via
    `on_progress`, and simply omitted — the caller keeps that index's hard
    cut. Never raises for a per-transition failure. Returns {} if `plan` is
    empty (or every transition failed).
    """
    def log(msg):
        if on_progress:
            on_progress(msg)
        else:
            print(msg)

    if not plan:
        return {}

    if client is None:
        from pipeline.kidsong.comfy import ComfyClient

        client = ComfyClient(cfg)

    if workdir is None:
        import tempfile

        workdir = tempfile.mkdtemp(prefix="kidsong_transitions_")
    else:
        workdir = os.path.abspath(workdir)
        os.makedirs(workdir, exist_ok=True)

    kidsong_cfg = (cfg or {}).get("kidsong", {}) if isinstance(cfg, dict) else {}
    seed_base = int(kidsong_cfg.get("seed", 20260717)) if isinstance(kidsong_cfg, dict) else 20260717

    results = {}
    for t in plan:
        idx = t["index"]
        try:
            prev_render = renders.get(t["prev_src"])
            next_render = renders.get(t["next_src"])
            if not prev_render:
                raise FileNotFoundError(
                    f"No render found for transition {idx}'s prev shot "
                    f"'{t['prev_src']}'"
                )
            if not next_render:
                raise FileNotFoundError(
                    f"No render found for transition {idx}'s next shot "
                    f"'{t['next_src']}'"
                )

            prev_last_png = os.path.join(workdir, f"transition_{idx:02d}_prev_last.png")
            next_first_png = os.path.join(workdir, f"transition_{idx:02d}_next_first.png")
            extract_frame(prev_render, "last", prev_last_png)
            extract_frame(next_render, "first", next_first_png)

            out_path = os.path.join(workdir, f"transition_{idx:02d}.mp4")
            prompt = (
                "smooth animated transition, 3D CGI toon style, "
                "consistent characters, gentle continuous motion"
            )
            render_transition(
                client, prev_last_png, next_first_png, prompt,
                seed_base + idx, cfg, out_path,
            )
            results[idx] = out_path
            log(f"Transition {idx}: rendered {out_path}")
        except Exception as exc:
            log(f"Transition {idx} failed ({exc!r}); keeping the hard cut.")
            continue

    return results


# --------------------------------------------------------------- chaining ---
def _norm_location(shot):
    value = shot.get("location") or shot.get("scene")
    if not value:
        return None
    return str(value).strip().lower()


def should_chain(prev_shot, shot, cfg, chain_len):
    """True if `shot` should be rendered i2v, conditioned on `prev_shot`'s
    last frame, instead of a fresh t2v render.

    Requires: shot_chaining_enabled, `shot` is not a reuse_of another shot's
    footage, `prev_shot` exists, the running chain is still under
    shot_chain_max_len, and the two shots share the same scene/location
    (case-insensitive comparison of shot.get("location") or
    shot.get("scene")). If neither shot carries a location/scene, this falls
    back to comparing `verse` instead — but a location on only one side (not
    both) is treated as a scene change, never as a match, so a chain never
    silently bridges past a change we can't actually verify.
    """
    opts = options(cfg)
    if not opts["shot_chaining_enabled"]:
        return False
    if shot.get("reuse_of"):
        return False
    if not prev_shot:
        return False
    if chain_len >= int(opts["shot_chain_max_len"]):
        return False

    prev_loc = _norm_location(prev_shot)
    cur_loc = _norm_location(shot)
    if prev_loc is None and cur_loc is None:
        prev_verse = prev_shot.get("verse")
        return prev_verse is not None and prev_verse == shot.get("verse")
    return prev_loc is not None and prev_loc == cur_loc


def chain_guide_image(prev_render_path, workdir):
    """Extract the previous shot's last frame into `workdir` for use as the
    next shot's i2v conditioning image. Returns the PNG path."""
    workdir = os.path.abspath(workdir)
    os.makedirs(workdir, exist_ok=True)
    base = os.path.splitext(os.path.basename(prev_render_path))[0]
    out_png = os.path.join(workdir, f"{base}_chainguide.png")
    return extract_frame(prev_render_path, "last", out_png)
