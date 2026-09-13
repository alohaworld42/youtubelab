"""
kidsong.audio_clean — strip ACE-Step's lead-in/lead-out junk and level the song.

WHY THIS EXISTS
---------------
ACE-Step reliably prepends a short burst of non-musical audio to its renders:
a broadband click/DC step in the first 20-90 ms, and sometimes a longer
noise-like smear (up to ~0.4 s) before the arrangement actually settles. On
five real renders measured at sr=22050 (frame 2048 / hop 512):

    file                    t=0 frame flatness   centroid    first tonal frame
    song_official           0.027 -> 0.132        3046-4178   ~0.09 s
    rainbow-color-fun       0.229                 4297        ~0.37 s
    splish-splash           0.136                 3776        ~0.02 s
    counting-fun            0.115                 3888        ~0.05 s
    counting-friends-day    0.083                 3502        ~0.05 s

Every one of them "enters" (reaches 20 % of peak RMS) within 0.02-0.07 s, so
there is no leading silence to trim — the junk is immediate and it is LOUD.
That is exactly why an amplitude gate is the wrong tool here.

HOW JUNK IS TOLD APART FROM A REAL DOWNBEAT
-------------------------------------------
The discriminator is not "how loud" but "does musical content START here and
CONTINUE". A frame is called *musical* when all three hold:

  1. it carries energy relative to the body of the song (not an absolute
     threshold — a quiet tonal intro is still music),
  2. its spectral flatness is low, i.e. the spectrum is peaky/harmonic rather
     than noise-like, and
  3. its chroma mass concentrates on a few pitch classes, i.e. it has definite
     pitch rather than smeared broadband energy.

The musical start is then the first frame from which a *sustain window* is
predominantly musical (`_SUSTAIN_SECONDS` of audio, `_SUSTAIN_RATIO` of whose
frames pass). Two consequences, and they are the whole point:

  * A loud noise burst followed by a drop FAILS: the burst's own frames are
    noise-like (2, 3) and whatever follows does not sustain, so no window
    anchored at t=0 qualifies and the start moves past the burst.
  * A legitimate loud downbeat PASSES and is never trimmed: its frames are
    tonal and pitched, and the music continues, so the window anchored at
    t=0 qualifies and the detected start is 0.

The ratio (rather than "every frame must be musical") matters: real
arrangements contain genuinely noise-like frames — hi-hats and cymbals. In the
rainbow render, mid-music percussion frames hit flatness 0.19, higher than some
of the junk. Requiring every frame to be tonal would trim real music.

Two further guards, both deliberate:

  * `head_is_junk`: the head that would be discarded must be *measurably worse*
    than the body it precedes (noisier, and/or less pitched, and/or quieter).
    If it is not, the detector is second-guessing itself and nothing is
    trimmed. This is the explicit backstop against eating a real intro.
  * `max_head_trim_seconds` (config): a hard cap. A detection beyond the cap is
    clamped to it and reported as such, so a pathological analysis can never
    swallow the song.

The tail is handled by running the same analysis on the reversed signal, which
catches a trailing noise burst, an abrupt mid-phrase cut-off and — the case
that actually dominates these renders — dead air. Every one of the five
measured files ends in digital silence (frame RMS ~0.001 against a body median
of 0.05-0.09):

    song_official 1.51 s   splish-splash 2.60 s   counting-fun 2.51 s
    rainbow 4.16 s         counting-friends-day 13.31 s

13 s of silence on a 60 s render is 22 % of the video, and because the beat
grid, the verse timing and the cut are all derived from the song's duration,
leaving it in means 13 s of footage over nothing. Hence the tail cap defaults
much higher than the head cap: trailing silence is unambiguous and safe to
remove, whereas a long head trim would always be suspicious.

After trimming, a short fade in/out (`fade_ms`) removes the click that cutting
mid-waveform would otherwise introduce, and ffmpeg's EBU R128 `loudnorm`
levels the track so every episode of the channel plays at the same volume.

Design constraints: numpy + librosa + soundfile (all already required) and
ffmpeg via subprocess. No torch, no GPU. librosa is imported lazily inside the
functions that need it so importing this module stays cheap.
"""
import os
import shutil
import subprocess

# --------------------------------------------------------------- analysis ---
#: Analysis sample rate. Detection only needs the spectral envelope, and 22.05
#: kHz matches what cut_qc measures with, so thresholds stay comparable.
_ANALYSIS_SR = 22050
_N_FFT = 2048
_HOP = 512

#: A frame must carry at least this fraction of the body's median RMS to count
#: as musical. Deliberately very low: this test exists ONLY to reject true
#: silence, never to judge whether something is loud enough to be music. The
#: measured trailing silence on real renders sits at 0.7-1.7 % of the body
#: median, so 5 % separates it comfortably while leaving quiet passages,
#: staccato gaps and note decays safely above the line. An earlier value of
#: 10 % was high enough to mark the quiet half of a staccato phrase as
#: non-musical, which stopped the sustain test from ever finding a start.
_ENERGY_FLOOR_RATIO = 0.05

#: Spectral flatness above this is noise-like. Measured junk frames sit at
#: 0.08-0.23; the tonal frames that follow them sit at 0.0002-0.006.
_FLATNESS_MAX = 0.06

#: Fraction of chroma mass in the top 3 pitch classes. Junk frames measured
#: 0.28-0.36 (energy smeared across all 12); tonal frames 0.40-0.71.
_CHROMA_TOP3_MIN = 0.38

#: A candidate start must be followed by this much predominantly-musical audio.
#: Long enough that an isolated burst cannot qualify, short enough that it
#: still resolves a start inside a two-bar intro.
_SUSTAIN_SECONDS = 1.0
#: ...with at least this fraction of the window's frames musical, so that
#: percussion frames inside real music do not veto a correct start.
_SUSTAIN_RATIO = 0.6

#: To be called junk, a region must be noisier than the body it precedes by
#: this factor...
_JUNK_FLATNESS_RATIO = 1.5
#: ...AND be noise-like in absolute terms. Both are required because a purely
#: relative test false-positives on very tonal material: a clean synthesized
#: triad has a body flatness around 0.00002, so *anything* clears 1.5x of it.
#: Measured junk heads sit at 0.07-0.17, tonal music at 0.0002-0.006.
_MIN_JUNK_FLATNESS = 0.02

# ---------------------------------------------------------------- defaults ---
#: Integrated loudness target, LUFS. -14 is the level YouTube normalizes to:
#: upload louder and YouTube simply attenuates playback, so mastering to -14
#: means the channel sounds identical before and after YouTube touches it, and
#: every episode matches every other one.
_DEFAULTS = {
    "enabled": True,
    # Head: deliberately tight. Measured junk lead-ins are 0.02-0.10 s, so 2 s
    # is already ~20x the observed worst case; anything longer is far likelier
    # to be a bad detection than real junk, and this is what stops one from
    # eating the song.
    "max_head_trim_seconds": 2.0,
    # Tail: deliberately generous. What gets removed here is dead air (up to
    # 13.3 s measured), which is unambiguous and safe; the junk classifier,
    # not the cap, is what prevents a quiet outro from being cut.
    "max_tail_trim_seconds": 20.0,
    "fade_ms": 30.0,
    "loudness_lufs": -14.0,
    "loudness_true_peak": -1.0,
    # Loudness range target, LU. Deliberately at loudnorm's maximum rather than
    # the broadcast-typical 11: whenever the target LRA is BELOW the material's
    # own range, loudnorm switches from linear to dynamic mode and compresses.
    # That is wrong here twice over -- it missed the integrated target by 2.1 dB
    # on song_official (input LRA 12.0 -> output -11.9 LUFS instead of -14.0),
    # and squashing dynamics works directly against cut_qc's primary musicality
    # gate, which requires a p95/p5 dynamic range of at least 8 dB. We want one
    # constant gain to a consistent level, not a compressor.
    "loudness_range": 20.0,
}


def settings_from_cfg(cfg):
    """Resolve the kidsong.audio_clean block over `_DEFAULTS`.

    Tolerates a missing cfg, a missing kidsong block and a missing/None
    audio_clean block so callers that predate this feature keep working.
    """
    block = ((cfg or {}).get("kidsong", {}) or {}).get("audio_clean", {}) or {}
    out = dict(_DEFAULTS)
    for key, default in _DEFAULTS.items():
        if key in block and block[key] is not None:
            value = block[key]
            out[key] = bool(value) if isinstance(default, bool) else type(default)(value)
    return out


# ---------------------------------------------------------------- features ---
def frame_features(y, sr):
    """Per-frame (rms, flatness, chroma_top3) for a mono signal.

    Split out from the decision logic so the thresholds can be unit-tested
    against synthetic numpy signals with no file I/O.
    """
    import numpy as np
    import librosa

    y = np.asarray(y, dtype=np.float32)
    if y.size < _N_FFT:
        y = np.pad(y, (0, _N_FFT - y.size))

    S = np.abs(librosa.stft(y, n_fft=_N_FFT, hop_length=_HOP))
    rms = librosa.feature.rms(S=S, frame_length=_N_FFT, hop_length=_HOP)[0]
    flatness = librosa.feature.spectral_flatness(S=S, n_fft=_N_FFT, hop_length=_HOP)[0]

    chroma = librosa.feature.chroma_stft(S=S, sr=sr)
    top3 = np.sort(chroma, axis=0)[-3:, :].sum(axis=0)
    total = np.maximum(chroma.sum(axis=0), 1e-9)
    chroma_top3 = top3 / total

    n = min(len(rms), len(flatness), len(chroma_top3))
    return rms[:n], flatness[:n], chroma_top3[:n]


def _musical_mask(rms, flatness, chroma_top3, body_rms):
    """Boolean per-frame "this frame is musical" — the three-way test from the
    module docstring (energy relative to the body, tonal spectrum, definite
    pitch). All three must hold."""
    import numpy as np

    energy_ok = rms >= (_ENERGY_FLOOR_RATIO * body_rms)
    tonal_ok = flatness <= _FLATNESS_MAX
    pitched_ok = chroma_top3 >= _CHROMA_TOP3_MIN
    return np.asarray(energy_ok & tonal_ok & pitched_ok)


def _first_sustained_index(mask, sustain_frames):
    """First index from which `mask` is predominantly True over the next
    `sustain_frames` frames. Returns 0 when the signal is musical from the
    very first frame — the "legitimate loud downbeat" case — and None when no
    window anywhere qualifies (the whole clip is noise-like)."""
    import numpy as np

    if mask.size == 0:
        return None
    sustain_frames = max(1, min(int(sustain_frames), mask.size))
    # Rolling mean of the boolean mask via a cumulative sum.
    csum = np.concatenate(([0.0], np.cumsum(mask.astype(np.float64))))
    n_windows = mask.size - sustain_frames + 1
    if n_windows <= 0:
        return 0 if mask.mean() >= _SUSTAIN_RATIO else None
    window_mean = (csum[sustain_frames:] - csum[:-sustain_frames]) / float(sustain_frames)
    window_mean = window_mean[:n_windows]
    qualifying = np.flatnonzero(window_mean >= _SUSTAIN_RATIO)
    if qualifying.size == 0:
        return None
    # The window must also START on a musical frame; otherwise a start could
    # land a frame or two before the music, back inside the junk.
    for idx in qualifying:
        if mask[idx]:
            return int(idx)
    return int(qualifying[0])


def _head_is_junk(rms, flatness, chroma_top3, start_idx, body_rms):
    """Is the region [0, start_idx) we are about to discard *measurably worse*
    than the body that follows it?

    The explicit backstop against trimming a real intro. A region only counts
    as junk when it is either

      * noise-like -- both absolutely (>= _MIN_JUNK_FLATNESS) and relative to
        the body (>= _JUNK_FLATNESS_RATIO x the body's flatness), or
      * essentially silent -- below the energy floor relative to the body.

    Note what is deliberately NOT a trigger: low chroma concentration on its
    own. An earlier revision used it and trimmed the first frame off a clean
    synthesized downbeat, whose opening frame measures chroma 0.378 against a
    0.38 floor purely because of STFT boundary padding. Pitch concentration is
    a useful corroborator for finding the start; it is too weak to justify
    destroying audio, so it is reported in the detail string but does not vote.

    Returns (bool, detail_string).
    """
    import numpy as np

    if start_idx <= 0:
        return False, "start at 0"
    # Medians, not means, throughout. The region being judged ends where the
    # music begins, so its last frames are the attack of the first note; on a
    # 2.7s silent tail those few loud frames dragged the MEAN rms to 15x the
    # median and the region was wrongly cleared as "not silent".
    head_flat = float(np.median(flatness[:start_idx]))
    body_flat = float(np.median(flatness[start_idx:])) if start_idx < len(flatness) else head_flat
    head_chroma = float(np.median(chroma_top3[:start_idx]))
    head_rms = float(np.median(rms[:start_idx]))

    noisier = head_flat >= max(_JUNK_FLATNESS_RATIO * body_flat, _MIN_JUNK_FLATNESS)
    too_quiet = head_rms < (_ENERGY_FLOOR_RATIO * body_rms)

    detail = (
        f"flatness {head_flat:.4f} vs body {body_flat:.4f}, "
        f"chroma {head_chroma:.3f}, rms {head_rms:.4f} vs body {body_rms:.4f}"
    )
    # Silence is checked first purely so the logged reason is accurate: digital
    # silence has a degenerate spectral flatness near 1.0, so it satisfies
    # `noisier` too and would otherwise be reported as "noise-like".
    if too_quiet:
        return True, "silent: " + detail
    if noisier:
        return True, "noise-like: " + detail
    return False, detail


def _detect_start(y, sr, max_trim_seconds):
    """Seconds of junk at the head of `y`, plus a human-readable reason.

    Returns (trim_seconds, reason). trim_seconds is always within
    [0, max_trim_seconds].
    """
    import numpy as np

    rms, flatness, chroma_top3 = frame_features(y, sr)
    if rms.size == 0:
        return 0.0, "no frames to analyse"

    # Body reference: median RMS of everything past the cap, so the junk we are
    # measuring against cannot inflate its own reference. Fall back to the whole
    # clip for material shorter than the cap.
    cap_frame = int(max_trim_seconds * sr / _HOP)
    body = rms[cap_frame:] if rms.size > cap_frame + 4 else rms
    body_rms = float(np.median(body)) if body.size else 0.0
    if body_rms <= 1e-6:
        return 0.0, "signal is silent or near-silent; left untouched"

    mask = _musical_mask(rms, flatness, chroma_top3, body_rms)
    sustain_frames = max(1, int(_SUSTAIN_SECONDS * sr / _HOP))
    start_idx = _first_sustained_index(mask, sustain_frames)

    if start_idx is None:
        return 0.0, "no sustained musical window found; left untouched"
    if start_idx == 0:
        return 0.0, "music sustained from the first frame (downbeat kept)"

    detected_s = float(start_idx * _HOP / float(sr))

    # Clamp to the safety cap FIRST, then classify the region we would actually
    # remove. Classifying the un-clamped detection would let a 10s detection
    # justify cutting a 2s region that was never examined.
    capped = detected_s > max_trim_seconds
    trim_idx = min(start_idx, int(max_trim_seconds * sr / _HOP))
    if trim_idx <= 0:
        return 0.0, f"detected start {detected_s:.3f}s but the safety cap leaves nothing to trim"

    is_junk, detail = _head_is_junk(rms, flatness, chroma_top3, trim_idx, body_rms)
    if not is_junk:
        return 0.0, f"not distinguishable from the body; left untouched ({detail})"

    trim_s = float(trim_idx * _HOP / float(sr))
    if capped:
        return trim_s, (
            f"junk until {detected_s:.3f}s exceeds the {max_trim_seconds:.2f}s "
            f"safety cap; trimmed {trim_s:.3f}s ({detail})"
        )
    return trim_s, f"junk until {trim_s:.3f}s ({detail})"


# ------------------------------------------------------------------ shaping ---
def _to_mono(y):
    import numpy as np

    y = np.asarray(y)
    return y.mean(axis=1) if y.ndim > 1 else y


def _apply_fades(y, sr, fade_ms):
    """Linear fade in/out over `fade_ms`, in place on a copy.

    Applied AFTER trimming: the cut lands mid-waveform, and stepping from a
    non-zero sample to silence is exactly what a click is.
    """
    import numpy as np

    y = np.array(y, copy=True)
    n = y.shape[0]
    fade_n = int(max(0.0, float(fade_ms)) * sr / 1000.0)
    fade_n = min(fade_n, n // 2)
    if fade_n <= 0:
        return y
    ramp = np.linspace(0.0, 1.0, fade_n, endpoint=False, dtype=np.float32)
    shape = (fade_n,) + (1,) * (y.ndim - 1)
    y[:fade_n] *= ramp.reshape(shape)
    y[n - fade_n:] *= ramp[::-1].reshape(shape)
    return y


def _measure_loudness(path, lufs, true_peak, lra):
    """Pass 1 of loudnorm: measure the file. Returns the measured dict or None.

    ffmpeg prints the JSON block to stderr after the progress output, so the
    parse looks for the last balanced ``{...}`` in the stream.
    """
    import json

    cmd = [
        "ffmpeg", "-hide_banner", "-i", path,
        "-af", f"loudnorm=I={lufs}:TP={true_peak}:LRA={lra}:print_format=json",
        "-f", "null", "-",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              encoding="utf-8", errors="replace")
    except OSError:
        return None
    err = proc.stderr or ""
    start = err.rfind("{")
    end = err.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        return json.loads(err[start:end + 1])
    except ValueError:
        return None


def _loudnorm(src_path, dst_path, lufs, true_peak, lra):
    """Two-pass EBU R128 normalization via ffmpeg. Returns True on success.

    Two passes, not one, because consistency across episodes is the whole
    point. Single-pass loudnorm is a dynamic normalizer with no lookahead, and
    on this material it missed the target by up to 2.1 dB (measured across the
    five real renders: -11.9 to -14.0 LUFS against a -14 target). The
    measure-then-apply pass runs in `linear=true` mode, which applies one
    constant gain computed from the real measurement — accurate, and it does
    not squash the song's dynamics the way the dynamic mode can.

    If the measurement pass fails, this falls back to a single dynamic pass
    rather than giving up on levelling entirely. A total failure here is
    non-fatal for the caller: trimmed-but-unlevelled audio is still a strict
    improvement over the raw render.
    """
    measured = _measure_loudness(src_path, lufs, true_peak, lra)
    filt = f"loudnorm=I={lufs}:TP={true_peak}:LRA={lra}"
    if measured:
        try:
            filt += (
                f":measured_I={float(measured['input_i'])}"
                f":measured_LRA={float(measured['input_lra'])}"
                f":measured_TP={float(measured['input_tp'])}"
                f":measured_thresh={float(measured['input_thresh'])}"
                ":linear=true:print_format=summary"
            )
        except (KeyError, TypeError, ValueError):
            pass  # fall back to the single dynamic pass

    cmd = [
        "ffmpeg", "-y", "-v", "error", "-i", src_path,
        "-af", filt, "-ar", "44100", "-ac", "2", dst_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except (subprocess.CalledProcessError, OSError):
        return False
    return os.path.exists(dst_path) and os.path.getsize(dst_path) > 0


# ------------------------------------------------------------------- public ---
def clean_song(path, cfg=None, out_path=None, runlog=None):
    """Trim ACE-Step's junk lead-in/lead-out, fade, and loudness-normalize.

    path:     the generated .wav to clean.
    cfg:      loaded config dict; reads cfg["kidsong"]["audio_clean"].
    out_path: destination. Defaults to `path` (cleaned in place, which is what
              sing_song wants). Pass a different path to keep the original.
    runlog:   optional pipeline.kidsong.runlog.RunLog — the decision is logged
              through it when present, and printed otherwise.

    Never raises for an audio reason: if analysis or ffmpeg fails, the original
    audio is preserved and the failure is reported in the returned metadata.
    A failed cleanup must not kill a 40-minute render.

    Returns a metadata dict:
        {"audio_path", "enabled", "applied", "head_trimmed", "tail_trimmed",
         "detected_start", "detected_end", "duration", "duration_before",
         "reason", "loudness_normalized", "loudness_lufs"}
    """
    opts = settings_from_cfg(cfg)
    path = os.path.abspath(path)
    out_path = os.path.abspath(out_path or path)

    meta = {
        "audio_path": out_path,
        "enabled": bool(opts["enabled"]),
        "applied": False,
        "head_trimmed": 0.0,
        "tail_trimmed": 0.0,
        "detected_start": 0.0,
        "detected_end": 0.0,
        "duration": 0.0,
        "duration_before": 0.0,
        "reason": "",
        "loudness_normalized": False,
        "loudness_lufs": float(opts["loudness_lufs"]),
    }

    if not opts["enabled"]:
        meta["reason"] = "audio_clean disabled in config"
        meta["duration"] = meta["duration_before"] = _safe_duration(path)
        _log(runlog, meta)
        return meta

    try:
        import numpy as np
        import soundfile as sf
    except Exception as exc:  # pragma: no cover - dependency guard
        meta["reason"] = f"skipped: audio libraries unavailable ({exc})"
        meta["duration"] = meta["duration_before"] = _safe_duration(path)
        _log(runlog, meta)
        return meta

    try:
        data, sr = sf.read(path, dtype="float32", always_2d=False)
    except Exception as exc:
        meta["reason"] = f"skipped: could not read {path} ({exc})"
        _log(runlog, meta)
        return meta

    total = data.shape[0] / float(sr)
    meta["duration_before"] = total
    meta["detected_end"] = total
    meta["duration"] = total
    if data.shape[0] == 0:
        meta["reason"] = "skipped: empty audio"
        _log(runlog, meta)
        return meta

    # ---- Detect. Analysis runs on a mono downmix at the analysis rate; the
    # trim is then applied to the original full-rate (possibly stereo) data. ----
    try:
        import librosa

        mono = _to_mono(data)
        if sr != _ANALYSIS_SR:
            mono = librosa.resample(np.ascontiguousarray(mono), orig_sr=sr, target_sr=_ANALYSIS_SR)
        head_s, head_reason = _detect_start(
            mono, _ANALYSIS_SR, float(opts["max_head_trim_seconds"])
        )
        tail_s, tail_reason = _detect_start(
            mono[::-1].copy(), _ANALYSIS_SR, float(opts["max_tail_trim_seconds"])
        )
    except Exception as exc:
        meta["reason"] = f"skipped: analysis failed ({type(exc).__name__}: {exc})"
        _log(runlog, meta)
        return meta

    # Never let the two ends meet: keep at least a second of audio.
    if total - head_s - tail_s < 1.0:
        meta["reason"] = (
            f"skipped: head {head_s:.2f}s + tail {tail_s:.2f}s would leave "
            f"under 1s of a {total:.2f}s song"
        )
        _log(runlog, meta)
        return meta

    start_n = int(round(head_s * sr))
    end_n = data.shape[0] - int(round(tail_s * sr))
    trimmed = data[start_n:end_n]
    trimmed = _apply_fades(trimmed, sr, opts["fade_ms"])

    meta["head_trimmed"] = float(head_s)
    meta["tail_trimmed"] = float(tail_s)
    meta["detected_start"] = float(head_s)
    meta["detected_end"] = float(total - tail_s)
    meta["reason"] = f"head: {head_reason}; tail: {tail_reason}"

    # ---- Write. Stage to a temp file so a failure never leaves a half-written
    # wav where the original used to be. ----
    tmp_trim = out_path + ".trim.tmp.wav"
    tmp_norm = out_path + ".norm.tmp.wav"
    try:
        sf.write(tmp_trim, trimmed, sr)
        normalized = _loudnorm(
            tmp_trim, tmp_norm,
            opts["loudness_lufs"], opts["loudness_true_peak"], opts["loudness_range"],
        )
        final_src = tmp_norm if normalized else tmp_trim
        meta["loudness_normalized"] = bool(normalized)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        shutil.move(final_src, out_path)
        meta["applied"] = True
    except Exception as exc:
        meta["reason"] = f"skipped: write failed ({type(exc).__name__}: {exc})"
        meta["head_trimmed"] = meta["tail_trimmed"] = 0.0
        meta["applied"] = False
    finally:
        for tmp in (tmp_trim, tmp_norm):
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass

    meta["duration"] = _safe_duration(out_path) if meta["applied"] else total
    meta["audio_path"] = out_path
    _log(runlog, meta)
    return meta


def _safe_duration(path):
    try:
        import soundfile as sf

        info = sf.info(path)
        return float(info.frames) / float(info.samplerate)
    except Exception:
        return 0.0


def _log(runlog, meta):
    """Report the trim decision through the run logger, or stdout without one."""
    msg = (
        "audio_clean: head -%.3fs tail -%.3fs | %.2fs -> %.2fs | "
        "loudnorm=%s | %s"
        % (
            meta["head_trimmed"], meta["tail_trimmed"],
            meta["duration_before"], meta["duration"],
            "yes" if meta["loudness_normalized"] else "no",
            meta["reason"],
        )
    )
    if runlog is not None and hasattr(runlog, "info"):
        try:
            runlog.info(msg)
            return
        except Exception:
            pass
    print(f"  -> {msg}")
