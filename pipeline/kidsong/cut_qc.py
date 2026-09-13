"""
kidsong.cut_qc — Automated quality gate for the assembled, beat-cut MP4.

Runs AFTER `edit.assemble` writes the staging file and BEFORE it's promoted to
Final/: catches a broken container (wrong resolution/codec, drifted duration),
a cut list that ignores the beat grid or overstays a shot, a washing-machine
audio mix (the ACE-Step failure mode this was built to catch — a wall of
near-constant-amplitude noise with no loud/quiet structure and no onsets;
see _check_audio for the calibration behind the current gate), and black
seams at cut boundaries.

Verdict shape matches pipeline.kidsong.review: {"accept": bool, "score": float,
"reasons": [...], "retry_hints": {...}}. Reasons are "<category>: <detail>"
strings; the score is 1 minus the sum of each *category* of failure's weight,
clamped to [0, 1].

Optional external gate mirroring review.py's ExternalReviewer: if
kidsong.review.reviewer == "external", a contact sheet of the edit
(`cut_sheet`) plus the programmatic verdict are written to
`<review_dir>/cut.request.json`, and this blocks (via
review.external_gate_poll) until `cut.response.json` appears or
kidsong.review.external_timeout elapses. `retry_hints` on an external verdict
passes through the reviewer's own `recut`/`reshoot` fields unmodified.

The cut gate is the LAST of a run's four external gates, so by the time it
runs the run's shared `review.ReviewSession` almost always already knows
whether anyone is answering. On an unattended run it therefore writes its
request/cut sheet and returns the programmatic verdict immediately instead of
burning another `external_timeout`. The programmatic checks below run in full
regardless — a cut that fails them is still held back from Final/.
"""
import json
import os
import subprocess

from pipeline.kidsong.review import external_gate_poll

_MIN_CUT, _MAX_CUT = 1.4, 5.0
_BEAT_TOLERANCE = 0.5
_DURATION_TOLERANCE = 0.5
_MIN_CUTS_FOR_LONG_VIDEO = 8
_LONG_VIDEO_SECONDS = 45.0
_BLACK_SEAM_LUMA = 12
_BLACK_SEAM_OFFSET = 0.1

# Secondary/coarse safety nets -- NOT the musicality test. See the primary
# gate (_DYNAMIC_RANGE_MIN_DB / _RMS_CV_MIN below) for why: on 8 calibrated
# real ACE-Step renders (3 confirmed-broken noise takes, 5 confirmed-good
# songs), flatness and chroma concentration both failed to cleanly separate
# broken from good audio.
#
# Tightened from 0.25 -> 0.08: flatness only weakly separated the two groups
# (broken 0.0296-0.1101, good 0.0131-0.0287) with a razor-thin overlap
# between the worst broken sample (0.0296) and the best good one (0.0287).
# 0.08 still catches 2 of the 3 broken samples with comfortable headroom
# above every good sample, but that near-overlap is exactly why this is a
# secondary net rather than the primary gate.
_FLATNESS_MAX = 0.08
# Lowered from 0.40 -> 0.25: chroma top-3 concentration does NOT discriminate
# on calibration data (broken spans 0.307-0.403, good spans 0.359-0.414) --
# at 0.40 this would have falsely rejected a verified-good song (0.359) while
# nearly passing a broken one (0.403). Kept only as a coarse floor for
# genuinely pitchless noise: every one of the 8 calibration samples clears
# 0.30, so at 0.25 it only fires on something far worse than anything
# observed, not as a musicality judgment.
_CHROMA_TOP3_MIN = 0.25
_RMS_MIN = 0.005
_AUDIO_SAMPLE_SECONDS = 30.0

# Primary "audio not musical" discriminators. Calibrated on the same 8 real
# ACE-Step renders: p95/p5 loudness dynamic range (dB) and the coefficient
# of variation of frame RMS both separate broken from good with a clean gap
# and zero overlap:
#   broken (3 samples): dynamic range 2.7-4.7 dB    rms_cv 0.104-0.196
#   good   (5 samples): dynamic range 13.2-45.3 dB  rms_cv 0.416-0.936
# Both thresholds sit in the gap with margin on both sides. _check_audio
# requires BOTH to fail before rejecting -- they're corroborating signals,
# not independent votes, so a real (if unusual) song that happens to be flat
# on just one of the two metrics still passes.
_DYNAMIC_RANGE_MIN_DB = 8.0  # gap: broken maxes at 4.7, next real is 13.2
_RMS_CV_MIN = 0.30  # gap: broken maxes at 0.196, next real is 0.416

# "audio noisy opening" gate -- the backstop for pipeline.kidsong.audio_clean.
# ACE-Step prepends a burst of broadband junk to its renders; audio_clean trims
# it before the song is ever cut. This check exists so that if the trimmer ever
# misses, the video does not reach Final/ with random noise on the front.
#
# Measured on the 5 real renders after cleaning, versus synthetic gross
# failures (a raw noise burst spliced in front of tonal music):
#
#   cleaned real files:      head flatness 0.0001-0.0424   ratio 0.09-2.08
#   0.3s noise + music:      head flatness 0.1197          ratio 213
#   0.5s noise + music:      head flatness 0.1901          ratio 331
#   1.0s noise + music:      head flatness 0.3797          ratio 684
#
# Both thresholds sit in that gap with margin on both sides, and BOTH must
# fail before rejecting -- consistent with the primary gate above, these are
# corroborating signals rather than independent votes. The ratio alone would
# false-reject a song whose body is unusually tonal (song_official's body
# flatness is 0.0135, so small absolute changes swing the ratio a long way);
# the absolute value alone would miss a noisy opening on a song that is noisy
# throughout.
#
# Sensitivity limit, stated plainly: averaged over a 1.5s window, a junk burst
# shorter than ~0.3s does not lift the mean far enough to trip this. That is
# the trimmer's job, and this gate is the net for a gross miss, not a
# replacement for it.
_OPENING_SECONDS = 1.5
_OPENING_FLATNESS_MAX = 0.10  # gap: real cleaned maxes at 0.0424, worst synthetic starts at 0.1197
_OPENING_FLATNESS_RATIO = 3.0  # gap: real cleaned maxes at 2.08, worst synthetic starts at 213

_CUT_WEIGHTS = {
    "container unreadable": 0.5,
    "container duration mismatch": 0.3,
    "container missing h264 video stream": 0.3,
    "container missing aac audio stream": 0.3,
    "resolution mismatch": 0.2,
    "no cuts": 0.5,
    "cut boundary off-beat": 0.15,
    "cut duration out of range": 0.2,
    "too few cuts": 0.2,
    "audio unreadable": 0.4,
    "audio silent": 0.4,
    "audio not musical": 0.35,
    "audio noisy opening": 0.35,
    "visual unreadable": 0.3,
    "black_seam": 0.3,
}


# ------------------------------------------------------------------ utils ---
def _verdict(reasons, weights, hint_key, default_weight=0.2):
    categories = {r.split(":", 1)[0].strip() for r in reasons}
    penalty = sum(weights.get(c, default_weight) for c in categories)
    score = max(0.0, min(1.0, 1.0 - penalty))
    accept = not reasons
    return {
        "accept": accept,
        "score": score,
        "reasons": list(reasons),
        "retry_hints": {hint_key: not accept, "reshoot": []},
    }


# --------------------------------------------------------------- container ---
def _probe_container(video_path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", video_path],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def _check_container(video_path, duration, cfg):
    try:
        info = _probe_container(video_path)
    except Exception as e:
        return [f"container unreadable: {e}"]

    streams = info.get("streams") or []
    vstreams = [s for s in streams if s.get("codec_type") == "video"]
    astreams = [s for s in streams if s.get("codec_type") == "audio"]

    reasons = []
    try:
        fmt_duration = float((info.get("format") or {}).get("duration", 0) or 0)
    except (TypeError, ValueError):
        fmt_duration = 0.0
    if abs(fmt_duration - float(duration)) > _DURATION_TOLERANCE:
        reasons.append(
            f"container duration mismatch: {fmt_duration:.2f}s vs expected {float(duration):.2f}s"
        )

    if not vstreams or vstreams[0].get("codec_name") != "h264":
        got = vstreams[0].get("codec_name") if vstreams else None
        reasons.append(f"container missing h264 video stream: got '{got}'")

    if not astreams or astreams[0].get("codec_name") != "aac":
        got = astreams[0].get("codec_name") if astreams else None
        reasons.append(f"container missing aac audio stream: got '{got}'")

    from pipeline.kidsong.edit import _output_dims

    W, H = _output_dims(cfg)
    if vstreams:
        w, h = vstreams[0].get("width"), vstreams[0].get("height")
        if (w, h) != (W, H):
            reasons.append(f"resolution mismatch: {w}x{h} vs expected {W}x{H}")

    return reasons


# -------------------------------------------------------------------- cuts ---
def _effective_max_cut(cfg):
    """The effective cut-duration ceiling, resolved from
    kidsong.review.max_cut_seconds, defaulting to `_MAX_CUT` when absent or
    when `cfg` is None -- so an unset key (or a caller that never passes cfg,
    like the existing tests in tests/test_beat_grid.py) is byte-identical to
    today's fixed 5.0s ceiling. This is the scene-mode Phase 1 counterpart of
    director.py's `kidsong.director.max_shot_seconds` -- raising the shot
    ceiling there is useless unless the cut gate is told to accept the
    longer cuts it produces."""
    review_cfg = ((cfg or {}).get("kidsong", {}) or {}).get("review", {}) or {}
    return float(review_cfg.get("max_cut_seconds", _MAX_CUT))


def _effective_min_cut(cfg):
    """The effective cut-duration FLOOR, resolved from
    kidsong.review.min_cut_seconds, defaulting to `_MIN_CUT`.

    The ceiling above has been config-driven since scene-mode Phase 1, with a
    docstring explaining that the director's bound and the gate's bound have to
    move together. The floor never got the same treatment, so the coupling only
    held in one direction: lowering `kidsong.director.min_shot_seconds` (a
    documented knob -- "snappier cutting") made the director plan shots the gate
    then rejected as "cut duration out of range", one reason per cut. The
    episode failed its own gate and never reached Final/, with nothing in the
    output pointing at the config key that caused it."""
    review_cfg = ((cfg or {}).get("kidsong", {}) or {}).get("review", {}) or {}
    return float(review_cfg.get("min_cut_seconds", _MIN_CUT))


def _check_cuts(cut_list, beats, duration, cfg=None):
    cut_list = list(cut_list or [])
    if not cut_list:
        return ["no cuts: cut list is empty"]

    max_cut = _effective_max_cut(cfg)
    min_cut = _effective_min_cut(cfg)

    if isinstance(beats, dict):
        beat_times = beats.get("beat_times") or []
    else:
        beat_times = beats or []
    beat_times = sorted(float(b) for b in beat_times)

    ordered = sorted(cut_list, key=lambda c: float(c["start"]))
    reasons = []

    if beat_times:
        for a, b in zip(ordered, ordered[1:]):
            boundary = float(a["end"])
            nearest = min(beat_times, key=lambda b_t: abs(b_t - boundary))
            dist = abs(nearest - boundary)
            if dist > _BEAT_TOLERANCE:
                reasons.append(
                    f"cut boundary off-beat: shots {a.get('shot_id')}/{b.get('shot_id')} "
                    f"boundary {boundary:.2f}s is {dist:.2f}s from nearest beat"
                )

    for c in ordered:
        d = float(c["end"]) - float(c["start"])
        if not (min_cut - 1e-6 <= d <= max_cut + 1e-6):
            reasons.append(
                f"cut duration out of range: shot {c.get('shot_id')} duration {d:.2f}s "
                f"(need {min_cut}-{max_cut}s)"
            )

    if float(duration) >= _LONG_VIDEO_SECONDS and len(ordered) < _MIN_CUTS_FOR_LONG_VIDEO:
        reasons.append(
            f"too few cuts: {len(ordered)} cuts for a {float(duration):.0f}s video "
            f"(need >= {_MIN_CUTS_FOR_LONG_VIDEO})"
        )

    return reasons


# ------------------------------------------------------------------- audio ---
def _spectral_metrics(y, sr):
    """Compute the raw numbers _check_audio thresholds against, isolated from
    file I/O so the threshold logic can be unit-tested against synthetic
    numpy signals.

    Returns a dict:
      rms              -- global RMS over the whole clip (silence check).
      dynamic_range_db -- 20*log10(p95/p5) of frame RMS (primary gate).
      rms_cv           -- std/mean of frame RMS, i.e. coefficient of
                           variation (primary gate, corroborates the above).
      flatness         -- mean spectral flatness (secondary net).
      chroma_top3      -- chroma top-3 concentration ratio (secondary net).
    """
    import numpy as np
    import librosa

    y64 = y.astype(np.float64)
    rms = float(np.sqrt(np.mean(y64 ** 2)))

    frame_rms = librosa.feature.rms(y=y)[0]
    frame_mean = float(np.mean(frame_rms))
    if frame_mean > 1e-12:
        p95 = float(np.percentile(frame_rms, 95))
        p5 = float(np.percentile(frame_rms, 5))
        dynamic_range_db = 20.0 * float(np.log10(p95 / max(p5, 1e-9)))
        rms_cv = float(np.std(frame_rms) / frame_mean)
    else:
        # Effectively silent -- 0/0 in log/ratio terms. Report 0 rather than
        # +/-inf or NaN; the silence check on global `rms` above already
        # gives the correct, readable "audio silent" diagnosis for this
        # case, so this just needs to not also fail loudly/confusingly.
        dynamic_range_db = 0.0
        rms_cv = 0.0

    flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    total = float(np.sum(chroma_mean)) or 1.0
    chroma_top3 = float(np.sum(np.sort(chroma_mean)[-3:])) / total

    return {
        "rms": rms,
        "dynamic_range_db": dynamic_range_db,
        "rms_cv": rms_cv,
        "flatness": flatness,
        "chroma_top3": chroma_top3,
    }


def _opening_metrics(y, sr):
    """Mean spectral flatness of the first `_OPENING_SECONDS` versus the rest.

    Isolated from file I/O like `_spectral_metrics` so the threshold logic can
    be unit-tested against synthetic numpy signals. Returns a dict with
    `opening_flatness`, `body_flatness` and their `ratio`, or None when the
    clip is too short to split into an opening and a body (nothing meaningful
    to compare, so the caller skips the check rather than guessing).
    """
    import numpy as np
    import librosa

    head_n = int(_OPENING_SECONDS * sr)
    # Need a real body to compare against, not a handful of frames.
    if y.size < head_n + sr:
        return None

    opening = float(np.mean(librosa.feature.spectral_flatness(y=y[:head_n])))
    body = float(np.mean(librosa.feature.spectral_flatness(y=y[head_n:])))
    return {
        "opening_flatness": opening,
        "body_flatness": body,
        "ratio": opening / max(body, 1e-9),
    }


def _check_audio(video_path):
    try:
        import librosa
    except Exception as e:
        return [f"audio unreadable: import failed ({e})"]

    try:
        y, sr = librosa.load(video_path, sr=22050, mono=True, duration=_AUDIO_SAMPLE_SECONDS)
    except Exception as e:
        return [f"audio unreadable: {e}"]

    if y is None or y.size == 0:
        return ["audio silent: no samples decoded"]

    m = _spectral_metrics(y, sr)
    reasons = []

    if m["rms"] <= _RMS_MIN:
        reasons.append(f"audio silent: rms {m['rms']:.4f} <= {_RMS_MIN}")

    # Primary gate: reject only when BOTH dynamic range and rms_cv are below
    # their floors (see the constants above for the calibration/rationale).
    dr_fail = m["dynamic_range_db"] < _DYNAMIC_RANGE_MIN_DB
    cv_fail = m["rms_cv"] < _RMS_CV_MIN
    if dr_fail and cv_fail:
        reasons.append(
            "audio not musical: dynamic range "
            f"{_DYNAMIC_RANGE_MIN_DB - m['dynamic_range_db']:.1f} dB below "
            f"{_DYNAMIC_RANGE_MIN_DB}, rms_cv "
            f"{_RMS_CV_MIN - m['rms_cv']:.2f} below {_RMS_CV_MIN}"
        )

    if m["flatness"] >= _FLATNESS_MAX:
        reasons.append(f"audio not musical: spectral flatness {m['flatness']:.3f} >= {_FLATNESS_MAX}")

    if m["chroma_top3"] <= _CHROMA_TOP3_MIN:
        reasons.append(
            f"audio not musical: chroma top-3 concentration {m['chroma_top3']:.3f} <= {_CHROMA_TOP3_MIN}"
        )

    # Backstop for audio_clean: reject a cut that opens on noise (see the
    # constants block). Both signals must fail, as with the primary gate.
    o = _opening_metrics(y, sr)
    if o is not None:
        if (
            o["opening_flatness"] >= _OPENING_FLATNESS_MAX
            and o["ratio"] >= _OPENING_FLATNESS_RATIO
        ):
            reasons.append(
                f"audio noisy opening: first {_OPENING_SECONDS}s spectral flatness "
                f"{o['opening_flatness']:.3f} >= {_OPENING_FLATNESS_MAX} and "
                f"{o['ratio']:.1f}x the body's {o['body_flatness']:.3f} "
                f"(>= {_OPENING_FLATNESS_RATIO}x) — untrimmed lead-in junk?"
            )

    return reasons


# ------------------------------------------------------------------ visual ---
def _check_black_seams(video_path, cut_list):
    import cv2

    # Keep OpenCV off the GPU: during a render the LTX model owns the VRAM and
    # an OpenCL buffer upload here fails outright (see review.py). CPU is fine
    # for reading a handful of frames. Global, idempotent.
    cv2.ocl.setUseOpenCL(False)

    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            return ["visual unreadable: could not open video"]
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        if fps <= 0:
            return []  # can't reliably sample a frame index; skip rather than false-reject

        reasons = []
        for c in cut_list or []:
            t = float(c["start"]) + _BLACK_SEAM_OFFSET
            frame_idx = int(round(t * fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            luma = float(gray.mean())
            if luma < _BLACK_SEAM_LUMA:
                reasons.append(
                    f"black_seam: shot {c.get('shot_id')} at {t:.2f}s luma={luma:.1f} "
                    f"(min {_BLACK_SEAM_LUMA})"
                )
        return reasons
    finally:
        cap.release()


# ---------------------------------------------------------------- contact ---
def cut_sheet(video_path, cut_list, review_dir):
    """Contact sheet of the EDIT (not a single shot): one row per cut, three
    frames each at [start+0.05, midpoint, end-0.05], labeled shot_id + duration."""
    import cv2
    from PIL import Image, ImageDraw, ImageFont

    # Keep OpenCV off the GPU here too (see _check_black_seams above /
    # review.py): this is a second, independent lazy `import cv2` site, so it
    # needs its own call rather than relying on _check_black_seams having
    # already run in this process. Global, idempotent.
    cv2.ocl.setUseOpenCL(False)

    os.makedirs(review_dir, exist_ok=True)
    out_path = os.path.join(review_dir, "cut_sheet.png")

    cut_list = list(cut_list or [])
    cap = cv2.VideoCapture(video_path)
    fps = (cap.get(cv2.CAP_PROP_FPS) or 24.0) if cap.isOpened() else 24.0

    def grab(t):
        idx = max(0, int(round(t * fps)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok and frame is not None:
            return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return Image.new("RGB", (160, 284), (32, 32, 32))

    rows = []
    for c in cut_list:
        start, end = float(c["start"]), float(c["end"])
        mid = (start + end) / 2.0
        times = [start + 0.05, mid, max(start + 0.05, end - 0.05)]
        thumbs = []
        for t in times:
            img = grab(t)
            img.thumbnail((160, 284))
            thumbs.append(img)
        rows.append((c, thumbs))
    cap.release()

    if not rows:
        Image.new("RGB", (160, 284), (32, 32, 32)).save(out_path)
        return out_path

    pad = 6
    tile_w = max(t.width for _, thumbs in rows for t in thumbs)
    tile_h = max(t.height for _, thumbs in rows for t in thumbs)
    label_h = 22
    row_h = tile_h + label_h + pad
    sheet_w = tile_w * 3 + pad * 4
    sheet_h = row_h * len(rows) + pad

    sheet = Image.new("RGB", (sheet_w, sheet_h), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    y = pad
    for c, thumbs in rows:
        x = pad
        for t in thumbs:
            sheet.paste(t, (x, y + (tile_h - t.height) // 2))
            x += tile_w + pad
        dur = float(c["end"]) - float(c["start"])
        draw.text((pad, y + tile_h + 2), f"{c.get('shot_id')} ({dur:.2f}s)", fill=(255, 255, 255), font=font)
        y += row_h

    sheet.save(out_path)
    return out_path


# --------------------------------------------------------------- external ---
def _external_gate(video_path, cut_list, verdict, cfg, review_dir):
    """Offer `verdict` for external review; return `verdict` if nobody answers.

    Wait duration is governed by the RUN's shared `review.ReviewSession` — see
    the "run sessions" section of review.py. Once N requests anywhere in the
    run have gone unanswered this writes the request + cut sheet and returns
    the programmatic verdict immediately rather than idling out
    `external_timeout`. A cut that fails its checks is still held either way.
    """
    review_cfg = (cfg.get("kidsong", {}) or {}).get("review", {}) or {}
    timeout = float(review_cfg.get("external_timeout", 600))

    os.makedirs(review_dir, exist_ok=True)
    sheet_path = cut_sheet(video_path, cut_list, review_dir)
    request_path = os.path.join(review_dir, "cut.request.json")
    response_path = os.path.join(review_dir, "cut.response.json")

    payload = {
        "video_path": os.path.abspath(video_path),
        "cut_list_summary": [
            {
                "shot_id": c.get("shot_id"),
                "start": c.get("start"),
                "end": c.get("end"),
                "src": c.get("src"),
            }
            for c in (cut_list or [])
        ],
        "contact_sheet": sheet_path,
        "programmatic_verdict": verdict,
    }

    response, skipped = external_gate_poll(
        "cut", request_path, response_path, payload, timeout, cfg
    )
    if response is None:
        out = dict(verdict)
        out["reasons"] = list(verdict.get("reasons", [])) + [
            "unattended mode: external review skipped — programmatic verdict used"
            if skipped
            else "external review timed out"
        ]
        return out

    response = dict(response)
    response.setdefault("reasons", [])
    response.setdefault("retry_hints", dict(verdict.get("retry_hints", {})))
    response.setdefault("score", verdict.get("score", 0.5))
    response["accept"] = bool(response.get("accept"))
    return response


# ------------------------------------------------------------------ public ---
def review_cut(video_path, cut_list, beats, words, duration, cfg, review_dir):
    reasons = []
    reasons += _check_container(video_path, duration, cfg)
    reasons += _check_cuts(cut_list, beats, duration, cfg)
    reasons += _check_audio(video_path)
    reasons += _check_black_seams(video_path, cut_list)

    verdict = _verdict(reasons, _CUT_WEIGHTS, "recut")

    review_cfg = (cfg.get("kidsong", {}) or {}).get("review", {}) or {}
    if str(review_cfg.get("reviewer", "heuristic")).lower() == "external":
        verdict = _external_gate(video_path, cut_list, verdict, cfg, review_dir)
    return verdict


# ------------------------------------------------------------------ test ---
if __name__ == "__main__":
    import copy
    import wave

    import numpy as np
    from PIL import Image, ImageDraw

    from pipeline.config import abspath, load_config
    from pipeline.kidsong.edit import assemble

    cfg = load_config()
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("kidsong", {}).setdefault("review", {})["reviewer"] = "heuristic"

    out_dir = abspath(cfg, cfg["paths"]["output_dir"])
    scratch = os.path.join(out_dir, "_kidsong_cut_qc_test")
    review_dir = os.path.join(scratch, "review")
    os.makedirs(scratch, exist_ok=True)
    os.makedirs(review_dir, exist_ok=True)

    DURATION = 8.0
    SR = 22050

    def make_chord_wav(path, duration, silent=False):
        n = int(SR * duration)
        if silent:
            pcm = np.zeros(n, dtype=np.int16)
        else:
            t = np.linspace(0, duration, n, endpoint=False)
            freqs = [261.63, 329.63, 392.00]  # C4 E4 G4 major triad -> a tonal, non-noisy bed
            signal = np.zeros_like(t)
            for f in freqs:
                signal += np.sin(2 * np.pi * f * t)
            signal = signal / len(freqs) * 0.6
            # A sustained chord with no amplitude variation is exactly the
            # flat, unvarying shape the dynamic-range/rms_cv gate (see
            # _DYNAMIC_RANGE_MIN_DB) is built to catch, so gate it into
            # distinct notes with quiet gaps between them -- real music has
            # loud/quiet structure and onsets, and this "good" fixture needs
            # to actually have that structure to stay a valid positive case.
            note_len = 0.5  # seconds
            phase = (t % (2 * note_len)) < note_len
            envelope = np.where(phase, 1.0, 0.05)
            signal = signal * envelope
            pcm = (signal * 32767).astype(np.int16)
        with wave.open(path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SR)
            wf.writeframes(pcm.tobytes())

    def make_moving_rect_mp4(path, color, duration):
        from moviepy import ImageSequenceClip

        n_frames = int(24 * duration)
        clip_fps = n_frames / duration
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
    make_moving_rect_mp4(mp4_a, (230, 70, 70), 3.0)
    make_moving_rect_mp4(mp4_b, (70, 120, 230), 3.0)
    renders = {"s0": mp4_a, "s1": mp4_b}

    # 4 cuts of 2s each over an 8s edit -> exercises the beat/duration checks
    # without tripping the "too few cuts" rule (only applies at >= 45s).
    cut_list = [
        {"shot_id": "c0", "start": 0.0, "end": 2.0, "src": "s0"},
        {"shot_id": "c1", "start": 2.0, "end": 4.0, "src": "s1"},
        {"shot_id": "c2", "start": 4.0, "end": 6.0, "src": "s0"},
        {"shot_id": "c3", "start": 6.0, "end": 8.0, "src": "s1"},
    ]
    beats = {"bpm": 30.0, "beat_times": [0.0, 2.0, 4.0, 6.0, 8.0]}
    words = []

    print("Assembling a musical test edit…")
    chord_wav = os.path.join(scratch, "chord.wav")
    make_chord_wav(chord_wav, DURATION, silent=False)
    good_path = os.path.join(scratch, "good_edit.mp4")
    assemble(cfg, cut_list, renders, chord_wav, words, DURATION, good_path)

    print("Reviewing the musical edit…")
    v_good = review_cut(good_path, cut_list, beats, words, DURATION, cfg, review_dir)
    print("verdict:", v_good)
    assert v_good["accept"] is True, v_good

    print("Assembling a silent-audio test edit…")
    silent_wav = os.path.join(scratch, "silent.wav")
    make_chord_wav(silent_wav, DURATION, silent=True)
    silent_path = os.path.join(scratch, "silent_edit.mp4")
    assemble(cfg, cut_list, renders, silent_wav, words, DURATION, silent_path)

    print("Reviewing the silent edit…")
    v_silent = review_cut(silent_path, cut_list, beats, words, DURATION, cfg, review_dir)
    print("verdict:", v_silent)
    assert v_silent["accept"] is False, v_silent
    assert any("audio silent" in r for r in v_silent["reasons"]), v_silent["reasons"]
    assert v_silent["retry_hints"].get("recut") is True, v_silent

    sheet = cut_sheet(good_path, cut_list, review_dir)
    print("cut sheet:", sheet)
    assert os.path.exists(sheet)

    print("PASS")
