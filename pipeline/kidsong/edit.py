"""
kidsong.edit — Beat-synced cut-list editor for children's-song videos.

Given a rendered per-shot clip library (`pipeline.kidsong.animate`/`scenes`
output) and a shotlist describing when each shot plays, this module:

  1. Detects the song's beat grid (`beat_grid`, librosa).
  2. Snaps shot boundaries onto nearby beats to build a `cut_list`
     (`build_cut_list`) — merging cuts that end up too short and splitting
     ones that end up too long so every cut lands in [1.6s, 4.5s].
  3. Assembles the final MP4 from that cut list (`assemble`), reusing the
     cover-scale math, `vfx.Loop`/`subclipped` idiom, and try/finally close
     pattern from `pipeline.kidsong.assemble_song` and the caption renderer
     from `pipeline.assemble`.

Built for MoviePy 2.x, mirroring the idioms used elsewhere in this package
(`.resized/.subclipped/.with_start/.with_duration`, `vfx.Loop`,
`vfx.CrossFadeIn`, closing every clip AND every VideoFileClip reader).
"""
import bisect
import json
import math
import os
import tempfile
import uuid

from moviepy import (
    AudioFileClip,
    CompositeVideoClip,
    ImageClip,
    VideoFileClip,
    vfx,
)

from pipeline.assemble import _caption_clips
from pipeline.kidsong.assemble_song import _cover_size

# --------------------------------------------------------------- constants ---
MIN_CUT = 1.6
MAX_CUT = 4.5
# How far inside the cut GATE's window (cut_qc._MIN_CUT / _MAX_CUT = 1.4 / 5.0)
# the builder aims, so an emitted cut never lands exactly on the gate boundary
# and get rejected by float noise. These are the differences that have always
# existed between the two windows, named so `_cut_window` can preserve them
# when the gate's window is moved by config.
_GATE_MARGIN_MIN = MIN_CUT - 1.4
_GATE_MARGIN_MAX = 5.0 - MAX_CUT
SNAP_TOLERANCE = 0.45
FADE_DURATION = 0.3
# Preschool TV grammar is hard cuts; mid-blend crossfade frames read as double
# exposures. Kept as a switch for other genres.
_CROSSFADES_ENABLED = False
# Over-zoom factor applied on top of cover-scale (crops edges symmetrically).
_EDGE_CROP = 1.04
# A take may be slowed down by up to this factor to fill its cut window
# invisibly; beyond it the fill falls back to the ping-pong (see
# `_extend_to_window`). 1.15 covers the max_frames=81 vs 3.6s-slot gap
# (3.6/3.375 = 1.067) with margin, while a >15% slowdown of a clap/skip
# starts to read as slow motion.
_MAX_SLOW_STRETCH = 1.15
# Caption baseline as a fraction of the WIDE frame height. captions.
# vertical_position (0.62) was tuned for the 9:16 Shorts frame; in 16:9 the same
# fraction sits dead centre over the characters' faces. Lower third instead.
_KIDSONG_CAPTION_VPOS = 0.80


# ------------------------------------------------------------- beat grid ---
# Grid-completion parameters (see `complete_beat_grid`).
#
# librosa's beat tracker only emits beats where it can hold a confident
# pulse. On a song with a quiet intro that means it can emit NOTHING for the
# first several seconds — measured on output/brushing_resing/song_official_
# clean.wav, whose first detected beat is at 13.72s even though the song's
# tempo (129.2 bpm, period 0.4644s) is rock steady from 0s. Cut boundaries
# landing in that dead zone then have no beat to be measured against, and
# cut_qc's ±0.5s check rejects boundaries that are in fact on the beat.
#
# So: fit a period+phase to the beats the tracker DID find and extrapolate a
# complete grid over the whole song, keeping every detected beat and only
# filling the regions the tracker left empty.
_GRID_MIN_BEATS = 3          # need >= 2 intervals before a median means anything
_GRID_MIN_BPM = 40.0         # below this the "period" is not a musical pulse
_GRID_MAX_BPM = 240.0        # above this we are fitting noise, not beats
# How tightly the detected beats must cluster around a single phase, measured
# as the circular resultant length R of their positions modulo the period
# (R = 1 -> a perfectly rigid grid, R -> 0 -> phases scattered at random).
#
# A "median residual as a fraction of the period" test looks natural and is
# almost useless here: for beats with NO grid at all the residuals are
# uniform over [-P/2, P/2], whose median |residual| is 0.25*P, so any
# threshold has to live in a sliver below that. Real trackers wobble more
# than you would guess — on song_official.wav the inter-beat intervals span
# 0.395-0.534s around a 0.4644s median, giving a median residual of 0.15*P
# even though the pulse is unambiguous. So test the phase concentration
# instead, with the Rayleigh statistic n*R^2 guarding against a small-n
# fluke (n*R^2 >= 9 is roughly p < 1e-4 against the uniform null).
#   measured: song_official.wav R=0.325 n=124 (nR^2=13.1)
#             song_official_clean.wav R=0.568 n=95 (nR^2=30.5)
_GRID_MIN_RESULTANT = 0.25
_GRID_MIN_RAYLEIGH = 9.0
_GRID_OCTAVE_TOLERANCE = 0.08  # relative error allowed when matching x2 / x0.5
# A grid point is considered "already covered" by a detected beat when one
# sits within this fraction of a period of it.
_GRID_COVER_FRACTION = 0.5


def _estimate_period(beat_times):
    """Robust inter-beat period from detected beats, or None.

    Uses the MEDIAN inter-beat interval (so a single jittered beat, or a
    handful of beats the tracker skipped — which show up as ~2x intervals —
    cannot drag the estimate), then re-medians over only the intervals close
    to that first estimate to shake off the skipped-beat outliers entirely.
    """
    if len(beat_times) < _GRID_MIN_BEATS:
        return None
    ivals = sorted(
        b - a for a, b in zip(beat_times, beat_times[1:]) if b - a > 1e-9
    )
    if not ivals:
        return None

    def median(xs):
        n = len(xs)
        mid = n // 2
        return xs[mid] if n % 2 else 0.5 * (xs[mid - 1] + xs[mid])

    coarse = median(ivals)
    if coarse <= 0:
        return None
    inliers = [v for v in ivals if 0.6 * coarse <= v <= 1.4 * coarse]
    return median(inliers) if inliers else coarse


def _fit_phase(beat_times, period):
    """(phase, resultant) of the grid, fitted over ALL detected beats.

    Anchoring on a single beat would inherit that beat's jitter (and on this
    song the only available anchor would be 13.72s, ~30 periods from the
    song start, where a small error is amplified). Instead take the circular
    mean of every beat's position modulo the period, which averages the
    jitter out and has no wrap-around discontinuity. The resultant length
    that falls out of the same sum measures how well a single phase actually
    describes the beats (see _GRID_MIN_RESULTANT).
    """
    two_pi = 2.0 * math.pi
    n = len(beat_times)
    sin_sum = sum(math.sin(two_pi * t / period) for t in beat_times)
    cos_sum = sum(math.cos(two_pi * t / period) for t in beat_times)
    resultant = math.hypot(sin_sum, cos_sum) / n if n else 0.0
    if abs(sin_sum) < 1e-12 and abs(cos_sum) < 1e-12:
        return 0.0, resultant
    phase = period * math.atan2(sin_sum, cos_sum) / two_pi
    return phase % period, resultant


def complete_beat_grid(bpm, beat_times, duration=None):
    """Extend a detected beat list into a grid spanning the whole song.

    Returns (beat_times, extrapolated_flags, grid_info) where `beat_times`
    is the merged, sorted grid, `extrapolated_flags` is a parallel list of
    bools (True == this beat was synthesised, not detected), and `grid_info`
    records the fit plus a human-readable `note`.

    Every detected beat is preserved exactly; synthetic beats are only added
    where the tracker left a gap of a full period or more. Degenerate inputs
    are returned UNCHANGED with a note explaining why — this never fabricates
    a grid out of nothing.
    """
    detected = sorted(float(b) for b in (beat_times or []))
    flags = [False] * len(detected)

    def degenerate(note, resultant=None, rayleigh=None):
        # `resultant`/`rayleigh` are included whenever the phase fit actually
        # ran (i.e. every bail-out AFTER `_fit_phase`, even one that fails the
        # module's own quality bar) — callers outside the beat grid (e.g. the
        # kidsong sing-coherence gate) need the raw number to compare renders
        # against each other, not just this grid's pass/fail verdict.
        info = {
            "source": "detected_only",
            "detected": len(detected),
            "extrapolated": 0,
            "note": note,
        }
        if resultant is not None:
            info["resultant"] = round(resultant, 4)
        if rayleigh is not None:
            info["rayleigh"] = round(rayleigh, 2)
        return list(detected), list(flags), info

    if len(detected) < _GRID_MIN_BEATS:
        return degenerate(
            f"only {len(detected)} detected beat(s); need >= {_GRID_MIN_BEATS} "
            "to estimate a period — grid left as detected"
        )

    period = _estimate_period(detected)
    if not period or period <= 0:
        return degenerate("could not estimate an inter-beat period — grid left as detected")

    period_bpm = 60.0 / period
    if not (_GRID_MIN_BPM <= period_bpm <= _GRID_MAX_BPM):
        return degenerate(
            f"implied tempo {period_bpm:.1f} bpm outside "
            f"[{_GRID_MIN_BPM}, {_GRID_MAX_BPM}] — grid left as detected"
        )

    phase, resultant = _fit_phase(detected, period)
    rayleigh = len(detected) * resultant * resultant
    if resultant < _GRID_MIN_RESULTANT or rayleigh < _GRID_MIN_RAYLEIGH:
        return degenerate(
            f"detected beats do not share a consistent phase (resultant "
            f"{resultant:.3f} < {_GRID_MIN_RESULTANT}, Rayleigh {rayleigh:.1f} "
            f"< {_GRID_MIN_RAYLEIGH}) — grid left as detected",
            resultant=resultant, rayleigh=rayleigh,
        )

    # Half/double-time cross-check against the tracker's own tempo estimate.
    # The detected beats' own spacing is what we must extrapolate at, so an
    # octave disagreement is reported, not silently "corrected".
    octave = None
    try:
        reported = float(bpm)
    except (TypeError, ValueError):
        reported = 0.0
    if reported > 0:
        rel = abs(period_bpm - reported) / reported
        if rel > _GRID_OCTAVE_TOLERANCE:
            for factor, name in ((2.0, "double-time"), (0.5, "half-time")):
                if abs(period_bpm - reported * factor) / reported <= _GRID_OCTAVE_TOLERANCE * factor:
                    octave = name
                    break
            else:
                return degenerate(
                    f"implied tempo {period_bpm:.1f} bpm disagrees with the tracker's "
                    f"{reported:.1f} bpm and is not a half/double-time relative — "
                    "grid left as detected",
                    resultant=resultant, rayleigh=rayleigh,
                )

    span_end = float(duration) if duration else detected[-1]
    span_end = max(span_end, detected[-1])

    cover = _GRID_COVER_FRACTION * period
    added = []
    k = math.floor((0.0 - phase) / period)
    k_max = math.ceil((span_end - phase) / period)
    while k <= k_max:
        t = phase + k * period
        k += 1
        if t < 0.0 or t > span_end + 1e-9:
            continue
        idx = bisect.bisect_left(detected, t)
        near = min(
            (abs(detected[j] - t) for j in (idx - 1, idx) if 0 <= j < len(detected)),
            default=float("inf"),
        )
        if near <= cover:
            continue  # the tracker already owns this beat — keep its timing
        added.append(t)

    merged = sorted(
        [(t, False) for t in detected] + [(t, True) for t in added],
        key=lambda p: p[0],
    )
    out_times = [t for t, _ in merged]
    out_flags = [f for _, f in merged]

    note = (
        f"extrapolated {len(added)} beat(s) at {period:.4f}s "
        f"({period_bpm:.1f} bpm) around {len(detected)} detected beat(s)"
        if added
        else f"tracker already covered the whole span at {period:.4f}s — no-op"
    )
    if octave:
        note += f"; NOTE detected spacing is {octave} vs the tracker's {reported:.1f} bpm"

    return out_times, out_flags, {
        "source": "extrapolated" if added else "detected_only",
        "period": round(period, 6),
        "phase": round(phase, 6),
        "bpm_from_period": round(period_bpm, 4),
        "resultant": round(resultant, 4),
        "rayleigh": round(rayleigh, 2),
        "octave_mismatch": octave,
        "detected": len(detected),
        "extrapolated": len(added),
        "span_end": round(span_end, 3),
        "note": note,
    }


def _audio_duration(audio_path):
    """Song length in seconds without decoding the whole file, or None."""
    try:
        import soundfile as sf

        info = sf.info(audio_path)
        if info.samplerate:
            return float(info.frames) / float(info.samplerate)
    except Exception:
        pass
    try:
        import wave

        with wave.open(audio_path, "rb") as wf:
            rate = wf.getframerate()
            if rate:
                return wf.getnframes() / float(rate)
    except Exception:
        pass
    return None


def _apply_grid(bpm, detected, duration):
    """Shape a `beat_grid`-style result dict from raw detected beats."""
    times, flags, info = complete_beat_grid(bpm, detected, duration)
    return {
        "bpm": bpm,
        "beat_times": times,
        # Additive keys — every existing reader (cut_qc._check_cuts,
        # build_cut_list, director, status) only touches "bpm"/"beat_times",
        # and old sidecars that lack these still load fine.
        "detected_beat_times": list(detected),
        "extrapolated": flags,
        "grid": info,
    }


def beat_grid(audio_path, cache=True, duration=None):
    """Detect BPM + beat timestamps for `audio_path`.

    Returns {"bpm": float, "beat_times": [seconds, ...], ...}. `beat_times`
    is a COMPLETE grid over the song: librosa's detected beats plus, where
    the tracker left a gap (typically a quiet intro — see the module
    constants), beats extrapolated at the fitted period and phase. The
    parallel `extrapolated` list flags which is which, `detected_beat_times`
    keeps the tracker's raw output, and `grid` records the fit and a
    human-readable note. Older sidecars carrying only bpm/beat_times still
    load and are upgraded in place on read.

    When `cache` is True, the result is memoized in a `<audio_path>.beats.json`
    sidecar so repeated calls (e.g. review/re-edit iterations) skip re-running
    librosa.
    """
    cache_path = audio_path + ".beats.json"
    if cache and os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = None
        if isinstance(data, dict) and "bpm" in data and "beat_times" in data:
            if "grid" in data:
                return data
            # Pre-grid sidecar: complete it now. This is pure arithmetic on
            # the cached beats, so an existing file gets the fix without
            # paying for another librosa pass.
            detected = data.get("detected_beat_times") or data["beat_times"]
            span = duration if duration is not None else _audio_duration(audio_path)
            upgraded = _apply_grid(data["bpm"], [float(b) for b in detected], span)
            try:
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(upgraded, f)
            except Exception:
                pass
            return upgraded

    import librosa

    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(tempo.item()) if hasattr(tempo, "item") else float(tempo)
    detected = [float(t) for t in librosa.frames_to_time(beat_frames, sr=sr)]
    span = duration if duration is not None else float(librosa.get_duration(y=y, sr=sr))

    result = _apply_grid(bpm, detected, span)
    if cache:
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(result, f)
        except Exception:
            pass
    return result


# ------------------------------------------------------------- cut list ---
def _snap_to_beat(t, beat_times, tolerance):
    """Nearest beat to `t`, if one exists within `tolerance` seconds; else `t`."""
    if not beat_times:
        return t
    idx = bisect.bisect_left(beat_times, t)
    candidates = []
    if idx < len(beat_times):
        candidates.append(beat_times[idx])
    if idx > 0:
        candidates.append(beat_times[idx - 1])
    best = min(candidates, key=lambda b: abs(b - t))
    return best if abs(best - t) <= tolerance else t


def _merge_short_segments(segs, min_cut):
    """Drop boundaries between segments so every surviving segment is >= min_cut.

    A too-short segment is absorbed into its previous neighbor (extending the
    previous segment's end and keeping the previous segment's shot); the very
    first segment, if left too short, absorbs its *next* neighbor instead.
    """
    out = list(segs)
    i = 1
    while i < len(out):
        if out[i]["end"] - out[i]["start"] < min_cut:
            out[i - 1]["end"] = out[i]["end"]
            del out[i]
        else:
            i += 1
    while len(out) > 1 and out[0]["end"] - out[0]["start"] < min_cut:
        out[0]["end"] = out[1]["end"]
        del out[1]
    return out


def _split_long_segments(segs, max_cut, min_cut):
    """Split any segment longer than max_cut into evenly-sized pieces (same
    shot/src reused for every piece — the underlying render is simply looped
    or replayed for each sub-window during assembly)."""
    out = []
    for seg in segs:
        span = seg["end"] - seg["start"]
        if span <= max_cut:
            out.append(seg)
            continue
        n_pieces = max(2, math.ceil(span / max_cut))
        piece_len = span / n_pieces
        if piece_len < min_cut:
            n_pieces = max(1, int(span // min_cut))
            piece_len = span / n_pieces
        cur = seg["start"]
        for k in range(n_pieces):
            piece_end = seg["end"] if k == n_pieces - 1 else cur + piece_len
            piece = dict(seg)
            piece["start"] = cur
            piece["end"] = piece_end
            out.append(piece)
            cur = piece_end
    return out


def _cut_window(cfg):
    """(min_cut, max_cut) the cut LIST is built to, resolved from config.

    This is the third place a cut-duration window exists, and until now it was
    the only one that ignored config — which made the other two partly inert.
    The chain is: `director._shot_bounds` plans the shot slots (config-driven),
    `build_cut_list` merges/splits them into cuts (THIS, previously fixed at
    1.6/4.5), and `cut_qc._check_cuts` gates the result (config-driven).

    So the documented "fewer, longer shots" workflow — raise
    kidsong.director.max_shot_seconds AND kidsong.review.max_cut_seconds —
    could not work: whatever the director planned, this stage split anything
    over 4.5s back down. Measured: four 8.0s shot slots came out as eight 4.0s
    cuts. The knob moved the plan and never the picture.

    Absent config resolves to today's constants exactly, so an unset cfg (and
    every existing caller that passes none) is byte-identical.
    """
    review = ((cfg or {}).get("kidsong", {}) or {}).get("review", {}) or {}
    min_cut, max_cut = MIN_CUT, MAX_CUT
    if review.get("min_cut_seconds") is not None:
        min_cut = max(0.1, float(review["min_cut_seconds"]) + _GATE_MARGIN_MIN)
    if review.get("max_cut_seconds") is not None:
        max_cut = float(review["max_cut_seconds"]) - _GATE_MARGIN_MAX
    # A window the config inverted would make _merge/_split fight each other.
    max_cut = max(max_cut, min_cut)
    return min_cut, max_cut


def build_cut_list(shotlist, beats, duration, cfg=None):
    """Build a beat-synced cut list from a shotlist and a beat grid.

    shotlist: {"shots": [{"id", "start", "end", "reuse_of"?, "render_path"?,
               "verse"?, ...}, ...]} (or a bare list of shot dicts).
    beats:    a `beat_grid(...)` result (or a bare list of beat seconds).
    duration: total song duration in seconds.

    Returns a list of cuts: {"shot_id", "start", "end", "src"[, "transition"]}.
    Every interior boundary is snapped to the nearest beat within
    +/-0.45s; cuts are then merged (if too short) or split (if too long) so
    every cut lands inside `_cut_window(cfg)` (1.6s-4.5s by default, and
    following kidsong.review.min/max_cut_seconds when those are set). The very
    first cut always starts at 0
    and the very last cut always ends exactly at `duration`. A cut is
    flagged `"transition": "fade"` when its shot's verse differs from the
    previous cut's shot's verse (and both are known).
    """
    shots = shotlist.get("shots") if isinstance(shotlist, dict) else shotlist
    if not shots:
        raise ValueError("shotlist must contain at least one shot")

    by_id = {s["id"]: s for s in shots}

    def resolve_src(s):
        ref = s.get("reuse_of")
        return ref if ref and ref in by_id else s["id"]

    ordered = sorted(shots, key=lambda s: float(s["start"]))
    n = len(ordered)

    if isinstance(beats, dict):
        beat_times = beats.get("beat_times") or []
    else:
        beat_times = beats or []
    beat_times = sorted(float(b) for b in beat_times)

    segs = []
    for i, s in enumerate(ordered):
        start = 0.0 if i == 0 else float(s["start"])
        end = duration if i == n - 1 else float(ordered[i + 1]["start"])
        segs.append({
            "shot_id": s["id"],
            "src": resolve_src(s),
            "verse": s.get("verse"),
            "start": start,
            "end": end,
        })

    # Snap interior boundaries (the shared edge between two segments) to the
    # nearest beat, keeping the grid monotonic.
    for i in range(len(segs) - 1):
        snapped = _snap_to_beat(segs[i]["end"], beat_times, SNAP_TOLERANCE)
        snapped = max(snapped, segs[i]["start"])
        segs[i]["end"] = snapped
        segs[i + 1]["start"] = snapped

    min_cut, max_cut = _cut_window(cfg)
    segs = _merge_short_segments(segs, min_cut)
    segs = _split_long_segments(segs, max_cut, min_cut)

    segs[0]["start"] = 0.0
    segs[-1]["end"] = duration

    cuts = []
    prev_verse = None
    for i, seg in enumerate(segs):
        cut = {
            "shot_id": seg["shot_id"],
            "start": round(seg["start"], 3),
            "end": round(seg["end"], 3),
            "src": seg["src"],
        }
        # Crossfades are OFF by default: preschool TV grammar is hard cuts,
        # and a mid-blend frame reads as an ugly double exposure. Re-enable
        # per-project with kidsong.edit_crossfades = true.
        if (
            _CROSSFADES_ENABLED
            and i > 0
            and seg["verse"] is not None
            and prev_verse is not None
            and seg["verse"] != prev_verse
        ):
            cut["transition"] = "fade"
        cuts.append(cut)
        prev_verse = seg["verse"]
    return cuts


def _output_dims(cfg):
    """Kidsong outputs wide 16:9 (real preschool-TV format); classic types
    keep the global vertical Shorts dimensions in cfg["video"]."""
    out_cfg = cfg.get("kidsong", {}).get("output", {})
    return (
        int(out_cfg.get("width", cfg["video"]["width"])),
        int(out_cfg.get("height", cfg["video"]["height"])),
    )


# --------------------------------------------------------------- captions ---
def _caption_layer(cfg, words):
    """Caption clips positioned for the KIDSONG frame, not the Shorts frame.

    DISABLED BY DEFAULT — `kidsong.captions.enabled` must be true for any of
    this to run. The channel does not want burnt-in lyrics on screen. The code
    below is deliberately KEPT rather than deleted: the Whisper word alignment
    that feeds it is still load-bearing for verse timing and the beat-aligned
    cut, and the positioning work documented here was expensive to get right.
    Flip the config key to bring the lyrics back.

    `pipeline.assemble._caption_clips` derives the caption baseline from
    `cfg["video"]["height"]` — the 1920px-tall vertical Shorts frame. Kidsong
    composites WIDE (1920x1080, see `_output_dims`), so with the shipped
    captions.vertical_position = 0.62 that baseline is int(1920 * 0.62) = 1190
    in a frame that is only 1080px tall: every caption is placed entirely
    BELOW the canvas. MEASURED, not theorised — with the shipped config,
    `_caption_clips` returns clips at ("center", 1190) for a 1920x1080
    composite. MoviePy silently drops a fully off-canvas clip, so the direct
    consequence is an episode with NO captions on screen and no error at all.

    SUSPECTED, NOT PROVEN, same root: run 20260720-092939 died inside
    `write_videofile` with

        File moviepy/video/VideoClip.py, in compose_mask
        ValueError: operands could not be broadcast together with
                    shapes (48,272) (0,272)

    i.e. a masked clip blended against a zero-row slice of the background.
    The captions are the only masked clips in this composite, and 272px is a
    real caption width for this song, so the caption layer is where it came
    from — but the exact geometry could NOT be reproduced on MoviePy 2.1.2
    with static or pop-resized captions at any offset, so treat the link as a
    strong inference rather than a demonstrated one. (The audio-only artefact
    itself IS explained: MoviePy muxes the audio before the first video frame,
    so a crash mid-write leaves a valid, playable, video-less MP4 — hence
    `verify_render` below, which catches that class regardless of cause.)

    So: scale the baseline against the frame we actually composite into, and
    clamp every caption so it stays wholly on-canvas even if a config or font
    change makes it taller than expected. `_caption_clips` reads no other
    dimension (`caption_render.render_group` sizes the PNG from the text and
    font alone), so overriding video.width/height is sufficient and leaves the
    caller's cfg untouched.
    """
    if not words:
        return []

    # On-screen lyrics are OFF for this channel (kidsong.captions.enabled).
    # This returns early — it does NOT disturb `words` itself, which is the
    # Whisper alignment the beat grid, verse timing (sing.verse_times_from_
    # words) and the cut list are all derived from upstream in generate.py.
    # Only the visual layer is dropped, so the cut is frame-identical apart
    # from the missing text and captions can be switched back on at any time.
    kid_captions = (cfg.get("kidsong", {}) or {}).get("captions", {}) or {}
    if not kid_captions.get("enabled", False):
        return []

    W, H = _output_dims(cfg)
    caption_cfg = dict(cfg)
    caption_cfg["video"] = {**cfg.get("video", {}), "width": W, "height": H}

    # `captions.vertical_position` was tuned for the 9:16 Shorts frame, where
    # 0.62 reads as lower-middle. In the wide 16:9 kidsong frame the same
    # fraction lands dead centre, over the characters' mouths — measured at
    # y=669 in a 1080px frame. Sing-along lyrics belong in the lower third,
    # so kidsong takes its own fraction (config: kidsong.captions).
    vpos = kid_captions.get("vertical_position", _KIDSONG_CAPTION_VPOS)
    caption_cfg["captions"] = {**cfg.get("captions", {}), "vertical_position": vpos}

    clips = _caption_clips(caption_cfg, words)

    out = []
    for clip in clips:
        height = getattr(clip, "h", None)
        if height:
            # `_caption_clips` positions ("center", y); keep the x rule and
            # only pull y back inside the frame.
            pos = clip.pos(0) if callable(getattr(clip, "pos", None)) else None
            y = pos[1] if isinstance(pos, (tuple, list)) and len(pos) == 2 else None
            if isinstance(y, (int, float)):
                clamped = max(0, min(int(y), H - int(height)))
                if clamped != int(y):
                    clip = clip.with_position(("center", clamped))
        out.append(clip)
    return out


# ------------------------------------------------------------ verification ---
class OutputVerificationError(RuntimeError):
    """A written episode does not contain what the pipeline was asked to write.

    Raised loudly and deliberately: an episode that is missing its video
    stream, or that is the wrong size or length, must never be promoted to
    Final/ or uploaded as if the run had succeeded.
    """


def _probe_json(path):
    import subprocess

    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", path],
        capture_output=True, text=True, check=True,
    ).stdout
    return json.loads(out)


def _video_stream_duration(video_stream):
    """The VIDEO stream's own duration, in seconds, or None if it can't be told.

    Container-level `format.duration` is the max across streams, so a video
    track that dies after a handful of frames is invisible there as long as
    the audio track runs the full length (this is exactly how a 0.2s/6-frame
    video muxed with a complete 60s audio track passed verification before
    this check existed). Prefer the stream's own `duration` field; fall back
    to nb_frames / frame-rate when ffprobe didn't populate it.
    """
    try:
        return float(video_stream.get("duration"))
    except (TypeError, ValueError):
        pass

    try:
        nb_frames = int(video_stream.get("nb_frames") or 0)
    except (TypeError, ValueError):
        nb_frames = 0
    if not nb_frames:
        return None

    rate = video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate")
    try:
        num, den = str(rate).split("/")
        num, den = float(num), float(den)
        if num <= 0 or den <= 0:
            return None
        return nb_frames / (num / den)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def verify_render(path, duration, size, tolerance=1.0, require_audio=True, label="output"):
    """Assert that `path` really is the video we just claimed to write.

    Checks, in order: the file exists and is non-trivial, ffprobe can read it,
    it has at least one VIDEO stream, that stream's resolution matches `size`,
    that VIDEO STREAM's own duration is within `tolerance` of `duration`, the
    container duration is within `tolerance` of `duration`, and (unless
    disabled) an audio stream is present.

    Raises `OutputVerificationError` listing every failed check. This exists
    because the failure it guards against is silent by construction: MoviePy
    muxes the audio before the first video frame is written, so a crash or a
    dropped video pipe leaves behind a perfectly valid, perfectly playable,
    completely video-less MP4 that every downstream size/duration heuristic
    happily accepts. The video-stream-duration check exists for the sibling
    failure: a video track that dies after a handful of frames while the
    audio track runs the full length — container-level `format.duration` is
    the stream max, so it reports the audio length and passes even though the
    video is a fraction of a second long.
    """
    problems = []

    if not os.path.exists(path):
        raise OutputVerificationError(f"{label}: {path} was not written at all")

    try:
        info = _probe_json(path)
    except Exception as exc:
        raise OutputVerificationError(f"{label}: ffprobe could not read {path}: {exc}") from exc

    streams = info.get("streams") or []
    video = [s for s in streams if s.get("codec_type") == "video"]
    audio = [s for s in streams if s.get("codec_type") == "audio"]

    if not video:
        problems.append(
            "NO VIDEO STREAM — the file contains only "
            + (", ".join(sorted({s.get("codec_type", "?") for s in streams})) or "nothing")
        )
    else:
        want_w, want_h = int(size[0]), int(size[1])
        got_w = int(video[0].get("width") or 0)
        got_h = int(video[0].get("height") or 0)
        if (got_w, got_h) != (want_w, want_h):
            problems.append(f"resolution {got_w}x{got_h}, expected {want_w}x{want_h}")
        if int(video[0].get("nb_frames") or 1) == 0:
            problems.append("video stream carries zero frames")
        if duration is not None:
            v_duration = _video_stream_duration(video[0])
            if v_duration is not None and abs(v_duration - float(duration)) > tolerance:
                problems.append(
                    f"video stream duration {v_duration:.2f}s, expected "
                    f"{float(duration):.2f}s (tolerance {tolerance:.2f}s) — "
                    "container duration can mask a truncated video track"
                )

    if require_audio and not audio:
        problems.append("no audio stream")

    if duration is not None:
        try:
            got = float((info.get("format") or {}).get("duration"))
        except (TypeError, ValueError):
            problems.append("container reports no duration")
        else:
            if abs(got - float(duration)) > tolerance:
                problems.append(
                    f"duration {got:.2f}s, expected {float(duration):.2f}s "
                    f"(tolerance {tolerance:.2f}s)"
                )

    if problems:
        raise OutputVerificationError(
            f"{label}: {path} failed verification — " + "; ".join(problems)
        )
    return True


def _post_grade(out_path, cfg, log):
    """Light color grade: the distilled LTX base pass renders low-contrast and
    slightly washed out, so lift saturation/contrast to the vivid preschool
    look. In-place ffmpeg pass; video re-encode only, audio stream copied."""
    # 1080p reference (content shots): sat ~47%, value ~73%. Our graded output
    # measured sat 45.9% (matched) but value 67.2% — lift brightness, keep sat.
    grade = cfg.get("kidsong", {}).get(
        "grade", {"saturation": 1.15, "contrast": 1.06, "brightness": 0.09}
    )
    if not grade:
        return
    import subprocess
    import tempfile

    log("Grading colors…")
    eq = (
        f"eq=saturation={grade.get('saturation', 1.22)}"
        f":contrast={grade.get('contrast', 1.07)}"
        f":brightness={grade.get('brightness', 0.02)}"
    )
    tmp = out_path + ".graded.mp4"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", out_path, "-vf", eq,
             "-c:v", "libx264", "-preset", "medium", "-crf", "18",
             "-c:a", "copy", "-movflags", "+faststart", tmp],
            check=True,
        )
        os.replace(tmp, out_path)
    except Exception as exc:  # pragma: no cover - graceful degradation
        log(f"Grade skipped ({exc})")
        try:
            os.remove(tmp)
        except OSError:
            pass


# -------------------------------------------------------------------- intro ---
def _probe_duration(path):
    import subprocess

    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", path],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return float(out)


def _concat_files(paths, out_path, reencode=False, fps=30):
    """Concatenate MP4s. Stream-copy by default; re-encode on demand.

    The concat *demuxer* (not the filter) is what preserves the episode's
    existing video bitstream untouched — the intro is built by
    `kidsong.intro` in exactly the episode container format precisely so this
    copy path works and the finished video is never re-encoded a second time.
    """
    import subprocess

    if reencode:
        args = ["ffmpeg", "-y", "-v", "error"]
        for p in paths:
            args += ["-i", p]
        n = len(paths)
        streams = "".join(f"[{i}:v:0][{i}:a:0]" for i in range(n))
        args += [
            "-filter_complex", f"{streams}concat=n={n}:v=1:a=1[v][a]",
            "-map", "[v]", "-map", "[a]",
            "-c:v", "libx264", "-profile:v", "high", "-pix_fmt", "yuv420p",
            "-preset", "medium", "-crf", "18", "-r", str(fps),
            "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
            "-movflags", "+faststart", out_path,
        ]
        subprocess.run(args, check=True)
        return out_path

    list_path = out_path + ".concat.txt"
    try:
        with open(list_path, "w", encoding="utf-8") as f:
            for p in paths:
                # ffmpeg's concat demuxer wants forward slashes even on Windows.
                f.write("file '%s'\n" % os.path.abspath(p).replace("\\", "/").replace("'", "'\\''"))
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
             "-i", list_path, "-c", "copy", "-movflags", "+faststart", out_path],
            check=True,
        )
    finally:
        try:
            os.remove(list_path)
        except OSError:
            pass
    return out_path


def prepend_intro(video_path, cfg, on_progress=None):
    """Put the fixed channel branding intro in front of a finished episode.

    Rewrites `video_path` in place and returns
    {"applied": bool, "intro_seconds": float, "reason": str|None}.

    ORDERING — this must run AFTER `_post_grade`. The grade exists to correct
    the LTX render's washed-out look; the intro is a clean, already-final
    Pillow composition, and pushing it through `eq=saturation/contrast/
    brightness` would shift the brand colours. Branding that changes colour
    whenever the song grade is retuned is exactly the drift this feature is
    meant to prevent, so the intro is joined on after grading.

    It must also run AFTER cut QC (`cut_qc.review_cut`), which checks the
    assembled file's duration against the SONG duration and probes for black
    seams at song-time offsets — a prepended intro shifts both. Hence the
    default wiring: `assemble()` leaves this alone unless
    `kidsong.intro.at_assemble` is set, and the caller applies it at promotion
    to Final/. An `<video>.intro.json` sidecar records the offset so anything
    downstream that maps song-time to video-time can add it explicitly instead
    of assuming video-time == song-time. Captions and cut boundaries are baked
    into the episode stream before this point and are simply carried along, so
    they stay in sync with the song.

    Never raises for a missing/disabled intro: the episode ships without it.
    """
    def log(msg):
        if on_progress:
            on_progress(msg)
        else:
            print(msg)

    from pipeline.kidsong import intro as intro_mod

    params = intro_mod.intro_params(cfg)
    if not params["enabled"]:
        return {"applied": False, "intro_seconds": 0.0, "reason": "disabled"}

    intro_path = intro_mod.ensure_intro(cfg, on_progress=on_progress)
    if not intro_path or not os.path.exists(intro_path):
        log("Intro skipped — no intro asset available.")
        return {"applied": False, "intro_seconds": 0.0, "reason": "asset missing"}

    fps = int(cfg.get("video", {}).get("fps", 30))
    try:
        intro_seconds = _probe_duration(intro_path)
        main_seconds = _probe_duration(video_path)
    except Exception as exc:
        log(f"Intro skipped (probe failed: {exc})")
        return {"applied": False, "intro_seconds": 0.0, "reason": f"probe failed: {exc}"}

    expected = intro_seconds + main_seconds
    tmp = video_path + ".intro.mp4"
    try:
        log("Prepending the channel intro…")
        try:
            _concat_files([intro_path, video_path], tmp, reencode=False, fps=fps)
            got = _probe_duration(tmp)
            if abs(got - expected) > 0.35:
                # Stream copy silently produced a wrong timeline (mismatched
                # timebase/SPS): redo it the slow, always-correct way.
                log(f"  stream-copy concat drifted ({got:.2f}s vs {expected:.2f}s) — re-encoding")
                raise ValueError("concat drift")
        except Exception:
            _concat_files([intro_path, video_path], tmp, reencode=True, fps=fps)
            got = _probe_duration(tmp)
            if abs(got - expected) > 0.5:
                raise ValueError(
                    f"concat duration {got:.2f}s, expected {expected:.2f}s"
                )
        os.replace(tmp, video_path)
    except Exception as exc:
        log(f"Intro skipped (concat failed: {exc})")
        try:
            os.remove(tmp)
        except OSError:
            pass
        return {"applied": False, "intro_seconds": 0.0, "reason": f"concat failed: {exc}"}

    try:
        with open(video_path + ".intro.json", "w", encoding="utf-8") as f:
            json.dump({
                "intro_path": intro_path,
                "intro_seconds": round(intro_seconds, 3),
                "song_starts_at": round(intro_seconds, 3),
                "total_seconds": round(got, 3),
            }, f, indent=2)
    except OSError:
        pass

    log(f"  intro applied ({intro_seconds:.2f}s) — song now starts at {intro_seconds:.2f}s")
    return {"applied": True, "intro_seconds": intro_seconds, "reason": None}


# --------------------------------------------------------------- title card ---
def _title_card_clip(cfg):
    """Optional 1.2s PIL-rendered title card, fading in as the first clip.

    Off by default; enabled via cfg["kidsong"]["title_card"] which may be
    either the title text itself (str) or `true` (then
    cfg["kidsong"]["title_card_text"] supplies the text). Returns
    (clip_or_None, tmp_png_path_or_None).
    """
    kidsong_cfg = cfg.get("kidsong", {}) or {}
    raw = kidsong_cfg.get("title_card")
    if not raw:
        return None, None
    text = raw if isinstance(raw, str) else str(kidsong_cfg.get("title_card_text") or "").strip()
    if not text:
        return None, None

    from PIL import Image, ImageDraw, ImageFont

    W, H = _output_dims(cfg)
    img = Image.new("RGBA", (W, H), (12, 12, 24, 255))
    draw = ImageDraw.Draw(img)

    font_path = (cfg.get("captions", {}) or {}).get("font_path")
    font = None
    if font_path and os.path.exists(font_path):
        try:
            font = ImageFont.truetype(font_path, 90)
        except Exception:
            font = None
    if font is None:
        font = ImageFont.load_default()

    bbox = draw.multiline_textbbox((0, 0), text, font=font, align="center")
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.multiline_text(
        ((W - tw) / 2, (H - th) / 2), text, font=font, fill=(255, 255, 255, 255), align="center"
    )

    tmp_path = os.path.join(tempfile.gettempdir(), f"kidsong_title_{uuid.uuid4().hex}.png")
    img.save(tmp_path)

    clip = (
        ImageClip(tmp_path)
        .with_duration(1.2)
        .with_start(0)
        .with_effects([vfx.CrossFadeIn(FADE_DURATION)])
    )
    return clip, tmp_path


# ---------------------------------------------------------------- assemble ---
def _extend_to_window(base, disp_dur):
    """Fit `base` (one shot's take) to a `disp_dur`-second cut window.

    A take that is LONGER than the window is simply trimmed. A take that is
    SHORTER used to be restart-looped (vfx.Loop on the raw take) — which
    replays the scene from frame 0 mid-window, i.e. the same scene visibly
    starts over. At draft quality (frames_scale 0.6) every take is ~40%
    shorter than its window, so essentially EVERY shot restarted; watching
    the cut read as "the same scene twice, back to back" — the exact defect
    the channel's no-repeat rule forbids.

    Now a short take is handled by HOW SHORT it is:

    * A SMALL shortfall (window <= _MAX_SLOW_STRETCH x the take) is covered by
      slowing the take down to exactly the window — a <=15% slowdown of
      gentle toddler motion is invisible, while ANY reversal is not. This is
      the final-quality case since kidsong.shot.max_frames was capped to the
      i2v-hires black-frame envelope (81f = 3.375s) below the director's old
      4.5s slot ceiling: the ping-pong below suddenly fired on most cuts and
      the episode visibly played scenes forward-then-backward (user report,
      2026-07-25 baa-baa).

    * A LARGE shortfall (draft tier, frames_scale 0.6: takes ~40% short) is
      PING-PONGED: TimeSymmetrize plays it forward then backward, ending on
      its own first frame, so the motion reverses smoothly instead of
      jumping, and looping IT (for windows longer than 2x the take) is
      seam-free at the wrap point too. A reversal beats a hard restart, but
      it is a visible compromise — acceptable for drafts, never for finals
      (the director's slot cap keeps finals out of this branch entirely).
    """
    if base.duration >= disp_dur:
        return base.subclipped(0, disp_dur)
    if disp_dur <= base.duration * _MAX_SLOW_STRETCH:
        factor = base.duration / disp_dur
        return base.with_effects([vfx.MultiplySpeed(factor)]).with_duration(disp_dur)
    base = base.with_effects([vfx.TimeSymmetrize()])
    if base.duration < disp_dur:
        return base.with_effects([vfx.Loop(duration=disp_dur)])
    return base.subclipped(0, disp_dur)


def assemble(cfg, cut_list, renders, voice_path, words, duration, out_path, on_progress=None):
    """Render the final MP4 from a beat-synced cut list.

    renders: dict shot_id -> rendered clip path (each render is a fresh shot
             clip, always played back starting at its own t=0). Audio is the
             voice/song track alone (`voice_path` — the full mix for ACE-Step
             songs, or the sung/TTS voice for the tts path).
    """
    def log(msg):
        if on_progress:
            on_progress(msg)
        else:
            print(msg)

    if not cut_list:
        raise ValueError("cut_list must not be empty")

    W, H = _output_dims(cfg)
    fps = cfg["video"]["fps"]

    log("Building cut clips…")
    clips = []
    video_sources = []
    for i, cut in enumerate(cut_list):
        src_id = cut["src"]
        render_path = renders.get(src_id)
        if not render_path or not os.path.exists(render_path):
            raise FileNotFoundError(
                f"No render found for shot '{src_id}' (cut {cut.get('shot_id')!r})"
            )

        seg_start, seg_end = float(cut["start"]), float(cut["end"])
        window = seg_end - seg_start
        if window <= 0:
            continue

        use_fade = cut.get("transition") == "fade" and i > 0
        overlap = FADE_DURATION if use_fade else 0.0
        overlap = max(0.0, min(overlap, window * 0.4, seg_start))
        disp_start = seg_start - overlap
        disp_dur = seg_end - disp_start

        raw = VideoFileClip(render_path).without_audio()
        video_sources.append(raw)
        cover_w, cover_h = _cover_size(raw.w, raw.h, W, H)
        # Slight over-zoom crops the outer ~2% per edge, hiding edge artifacts
        # (e.g. faint LoRA watermark ghosts in corners) without visible loss.
        cover_w, cover_h = int(cover_w * _EDGE_CROP), int(cover_h * _EDGE_CROP)
        base = raw.resized((cover_w, cover_h)).with_position("center")
        base = _extend_to_window(base, disp_dur)
        clip = base.with_duration(disp_dur)
        if overlap > 0:
            clip = clip.with_effects([vfx.CrossFadeIn(overlap)])
        clip = clip.with_start(disp_start)
        clips.append(clip)

    log("Rendering captions…")
    captions = _caption_layer(cfg, words)

    title_clip, title_tmp_path = _title_card_clip(cfg)
    title_clips = [title_clip] if title_clip is not None else []

    log("Compositing layers…")
    final = CompositeVideoClip([*clips, *captions, *title_clips], size=(W, H)).with_duration(duration)

    log("Loading voice/song audio…")
    audio = AudioFileClip(voice_path)
    final = final.with_audio(audio)

    # Encode + verify + grade all happen against a `.part.mp4` sibling, never
    # against `out_path` itself: a crash mid-encode or a failed verification
    # must never leave a broken/truncated file at the name QC, promotion, and
    # every other caller treat as "this episode is done". `out_path` is only
    # ever created by the final os.replace() below, once everything upstream
    # has already passed.
    tmp_out = f"{out_path}.part.mp4"
    try:
        try:
            log("Encoding MP4 (this is the slow part)…")
            final.write_videofile(
                tmp_out,
                fps=fps,
                codec="libx264",
                audio_codec="aac",
                preset="medium",
                threads=os.cpu_count() or 4,
                logger=None,
            )
            # Hard gate. MoviePy muxes the audio BEFORE the first video frame is
            # written, so anything that kills frame generation (a caption placed
            # off-canvas, a dead reader, a broken pipe) leaves a valid, playable,
            # video-less MP4 behind. _post_grade cannot catch that either: its
            # ffmpeg pass fails on a file with no video stream and is swallowed by
            # design as "grade skipped", preserving the broken file. Verify here,
            # loudly, so a video-less episode can never be promoted again.
            verify_render(tmp_out, duration, (W, H), label="assemble")
            _post_grade(tmp_out, cfg, log)
            # Stamp the AI disclosure into the container BEFORE the closing
            # verification, so the file that gets verified is the file that gets
            # published — a stream copy preserves duration and dimensions, and
            # if it somehow did not, verify_render below is what would say so.
            from pipeline import ai_disclosure

            ai_disclosure.tag_file(tmp_out, cfg, log)
            # And again after the grade — it rewrites the file in place.
            verify_render(tmp_out, duration, (W, H), label="post-grade")
        except Exception:
            # Never leave a half-written or failed-verification file at
            # out_path, and never delete the evidence either (repo rule:
            # generated media is channel inventory) — park it under a name
            # that is unambiguously not a finished episode.
            failed_path = f"{out_path}.failed.mp4"
            if os.path.exists(tmp_out):
                try:
                    os.replace(tmp_out, failed_path)
                    log(f"Encode/verify failed — partial output kept at {failed_path}")
                except OSError:
                    pass
            raise

        os.replace(tmp_out, out_path)
        # Intro goes on AFTER the grade (see prepend_intro's docstring: the
        # brand colours must not be pushed through the song's eq pass).
        # Off by default here because cut_qc still has to measure this file
        # against the SONG duration — see kidsong.intro.at_assemble.
        # Runs against the now-promoted out_path (not tmp_out) so its
        # `<out_path>.intro.json` sidecar lands next to the file it actually
        # describes.
        if (cfg.get("kidsong", {}).get("intro") or {}).get("at_assemble", False):
            prepend_intro(out_path, cfg, on_progress=log)
    finally:
        # Close every clip so ffmpeg reader/writer handles are released (an
        # open handle on Windows keeps intermediate files locked for the
        # caller's cleanup, and leaked handles pile up in the long-lived
        # Flask server across repeated generations).
        for clip in (final, audio, *clips, *captions, *title_clips, *video_sources):
            try:
                clip.close()
            except Exception:
                pass
        if title_tmp_path:
            try:
                os.remove(title_tmp_path)
            except OSError:
                pass
    return out_path


# ------------------------------------------------------------------ test ---
if __name__ == "__main__":
    import subprocess
    import wave

    import numpy as np
    from PIL import Image, ImageDraw

    from pipeline.config import abspath, load_config

    cfg = load_config()
    out_dir = abspath(cfg, cfg["paths"]["output_dir"])
    os.makedirs(out_dir, exist_ok=True)
    scratch = os.path.join(out_dir, "_kidsong_edit_test")
    os.makedirs(scratch, exist_ok=True)

    DURATION = 10.0
    BPM = 120.0
    BEAT_PERIOD = 60.0 / BPM  # 0.5s

    # --- a 120bpm click-track WAV: short percussive bursts every beat -------
    audio_path = os.path.join(scratch, "beats.wav")
    sr = 22050
    n_samples = int(sr * DURATION)
    buf = np.zeros(n_samples, dtype=np.float64)
    click_len = int(sr * 0.05)
    decay = np.exp(-np.linspace(0, 12, click_len))
    beat_t = 0.0
    while beat_t < DURATION:
        start_i = int(beat_t * sr)
        end_i = min(n_samples, start_i + click_len)
        n = end_i - start_i
        if n > 0:
            buf[start_i:end_i] += decay[:n]
        beat_t += BEAT_PERIOD
    buf = (buf / (np.max(np.abs(buf)) or 1.0) * 0.9 * 32767).astype(np.int16)
    with wave.open(audio_path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(buf.tobytes())

    print("Detecting beat grid…")
    beats = beat_grid(audio_path, cache=True)
    print(f"  bpm={beats['bpm']:.1f} beats={len(beats['beat_times'])}")
    assert len(beats["beat_times"]) >= 4, "expected several detected beats"

    # --- two synthetic 3s, 480x832 moving-rectangle mp4s --------------------
    from moviepy import ImageSequenceClip

    def make_moving_rect_mp4(path, color):
        n_frames = 60
        clip_fps = n_frames / 3.0  # 3-second source clip
        frames = []
        for k in range(n_frames):
            frame = Image.new("RGB", (480, 832), (25, 25, 25))
            d = ImageDraw.Draw(frame)
            x = int((k / n_frames) * 380)
            d.rectangle((x, 380, x + 90, 470), fill=color)
            frames.append(np.array(frame))
        seq = ImageSequenceClip(frames, fps=clip_fps)
        seq.write_videofile(path, fps=clip_fps, codec="libx264", audio=False, logger=None)
        seq.close()

    mp4_a = os.path.join(scratch, "render_a.mp4")
    mp4_b = os.path.join(scratch, "render_b.mp4")
    make_moving_rect_mp4(mp4_a, (230, 70, 70))
    make_moving_rect_mp4(mp4_b, (70, 120, 230))

    # --- shotlist: 4 shots, one reuse_of, spanning verse 0 then verse 1 ------
    shotlist = {
        "shots": [
            {"id": "s0", "start": 0.0, "end": 2.5, "verse": 0},
            {"id": "s1", "start": 2.5, "end": 5.0, "verse": 0},
            {"id": "s2", "start": 5.0, "end": 7.5, "reuse_of": "s0", "verse": 1},
            {"id": "s3", "start": 7.5, "end": 10.0, "verse": 1},
        ]
    }
    renders = {"s0": mp4_a, "s1": mp4_b, "s3": mp4_b}

    print("Building cut list…")
    cut_list = build_cut_list(shotlist, beats, DURATION)
    for c in cut_list:
        print(" ", c)

    # Every interior boundary should land within 0.46s of a detected beat.
    # (The very first/last boundaries are pinned to 0 / duration by design,
    # not necessarily beat-aligned.)
    beat_times = beats["beat_times"]

    def nearest_beat_dist(t):
        if not beat_times:
            return 0.0
        return min(abs(t - b) for b in beat_times)

    for c in cut_list[:-1]:
        d = nearest_beat_dist(c["end"])
        assert d <= 0.46, f"cut boundary {c['end']} is {d:.3f}s from nearest beat"
    for c in cut_list:
        assert MIN_CUT - 1e-6 <= (c["end"] - c["start"]) <= MAX_CUT + 1e-6, f"cut {c} out of bounds"
    assert cut_list[0]["start"] == 0.0
    assert abs(cut_list[-1]["end"] - DURATION) < 1e-6
    print("  cut list boundaries OK")

    words = [{"word": "LA", "start": 0.5 * k, "end": 0.5 * k + 0.4} for k in range(20)]

    out_path = os.path.join(out_dir, "kidsong_edit_test.mp4")
    print("Assembling…")
    result = assemble(cfg, cut_list, renders, audio_path, words, DURATION, out_path)
    print(f"  wrote {result}")

    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,codec_name",
            "-show_entries", "format=duration",
            "-of", "json", result,
        ],
        capture_output=True, text=True, check=True,
    )
    info = json.loads(probe.stdout)
    stream = info["streams"][0]
    fmt_duration = float(info["format"]["duration"])
    print(f"  ffprobe: {stream} duration={fmt_duration:.2f}")
    assert stream["width"] == 1080 and stream["height"] == 1920, stream
    assert stream["codec_name"] == "h264", stream
    assert abs(fmt_duration - DURATION) < 0.75, fmt_duration
    print("PASS")
