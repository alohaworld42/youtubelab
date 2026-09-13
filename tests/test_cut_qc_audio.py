"""Tests for the "audio not musical" gate in pipeline/kidsong/cut_qc.py.

Calibrated on 8 real ACE-Step renders (3 confirmed-broken "washing machine"
noise takes rejected by ear, 5 confirmed-good/mid songs) -- see the
constants block at the top of cut_qc.py for the measured numbers this is
based on. Loudness dynamic range (p95/p5 of frame RMS, in dB) and the
coefficient of variation of frame RMS separated the two groups cleanly:

    broken (3 samples): dynamic range 2.7-4.7 dB    rms_cv 0.104-0.196
    good   (5 samples): dynamic range 13.2-45.3 dB  rms_cv 0.416-0.936

while spectral flatness and chroma top-3 concentration (the old gate) did
not reliably separate them.

Everything here is synthesized with numpy/soundfile -- no network, no GPU,
no real songs on disk.
"""
import numpy as np
import pytest

from pipeline.kidsong import cut_qc

librosa = pytest.importorskip("librosa")
soundfile = pytest.importorskip("soundfile")

SR = 22050


# --------------------------------------------------------------- signals ---
def _noisy_drone(duration=6.0, seed=0):
    """Constant-amplitude broadband noise: the 'washing machine' ACE-Step
    failure mode -- low dynamic range, low rms_cv, AND high spectral
    flatness (it's genuinely noise-like), so both the old and new gates
    should catch it."""
    rng = np.random.default_rng(seed)
    n = int(SR * duration)
    y = rng.normal(0, 0.25, n)
    return np.clip(y, -1.0, 1.0).astype(np.float32)


def _constant_tone_drone(duration=6.0, freq=220.0):
    """A single sustained tone at constant amplitude: no loud/quiet
    structure at all (fails the primary gate hard) but a highly peaky,
    tonal spectrum (very low flatness) and total pitch-class concentration
    on one note (very high chroma top-3). This mirrors the calibration
    sample 'BROKEN bath old' (flatness 0.0296, dynamic range 2.7 dB,
    rms_cv 0.107): under the OLD flatness/chroma-only gate this shape would
    have nearly slipped through as 'musical'. Only the dynamic-range/rms_cv
    gate catches it."""
    n = int(SR * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * freq * t)
    return y.astype(np.float32)


def _bursty_tonal(duration=6.0, note_len=0.4, quiet=0.03):
    """A tonal chord gated into distinct loud notes with quiet gaps in
    between -- real music has loud/quiet structure and note onsets: high
    dynamic range, high rms_cv, low flatness, high chroma concentration."""
    n = int(SR * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    freqs = [261.63, 329.63, 392.00]  # C4 E4 G4
    sig = np.zeros_like(t)
    for f in freqs:
        sig += np.sin(2 * np.pi * f * t)
    sig = sig / len(freqs) * 0.6
    phase = (t % (2 * note_len)) < note_len
    envelope = np.where(phase, 1.0, quiet)
    return (sig * envelope).astype(np.float32)


def _write_wav(path, y, sr=SR):
    soundfile.write(str(path), y, sr)
    return str(path)


# --------------------------------------------- metric helper (no file IO) ---
def test_metrics_separate_noisy_drone_from_bursty_tonal():
    """Direct unit test of _spectral_metrics: a constant-amplitude noisy
    drone must land below both primary-gate floors, and a bursty tonal
    signal with loud/quiet sections must land above both, with a wide
    margin -- matching the calibration gap (broken maxes at 4.7 dB / 0.196
    cv, good starts at 13.2 dB / 0.416 cv)."""
    drone = cut_qc._spectral_metrics(_noisy_drone(), SR)
    tonal = cut_qc._spectral_metrics(_bursty_tonal(), SR)

    assert drone["dynamic_range_db"] < cut_qc._DYNAMIC_RANGE_MIN_DB
    assert drone["rms_cv"] < cut_qc._RMS_CV_MIN

    assert tonal["dynamic_range_db"] >= cut_qc._DYNAMIC_RANGE_MIN_DB
    assert tonal["rms_cv"] >= cut_qc._RMS_CV_MIN

    # not a hair's-width margin -- should reflect the observed calibration gap
    assert tonal["dynamic_range_db"] - drone["dynamic_range_db"] > 10.0
    assert tonal["rms_cv"] - drone["rms_cv"] > 0.2


def test_metrics_constant_tone_fails_only_the_old_gate():
    """This is the regression the task called out: a signal that the OLD
    flatness/chroma-only gate would have nearly passed (low flatness, high
    chroma concentration -- it IS a clean tone) but that is obviously not a
    song (a single sustained drone, no dynamics). The primary dynamic-range/
    rms_cv gate must catch it where flatness/chroma alone would not."""
    m = cut_qc._spectral_metrics(_constant_tone_drone(), SR)

    # old-gate signals both say "this looks musical"
    assert m["flatness"] < cut_qc._FLATNESS_MAX
    assert m["chroma_top3"] > cut_qc._CHROMA_TOP3_MIN

    # new primary gate correctly flags it as not musical
    assert m["dynamic_range_db"] < cut_qc._DYNAMIC_RANGE_MIN_DB
    assert m["rms_cv"] < cut_qc._RMS_CV_MIN


# ------------------------------------------------- threshold boundary ---
def test_dynamic_range_cv_boundary_and_and_logic(tmp_path, monkeypatch):
    """Direct boundary test of the comparison logic in _check_audio: exactly
    AT both floors must pass (strict less-than fails), just below both must
    reject, and -- because the two metrics are corroborating signals, not
    independent votes -- a single failing metric alone must NOT reject."""
    # a tiny real (silent-free) wav so librosa.load succeeds; _spectral_metrics
    # itself is monkeypatched so its actual content doesn't matter here.
    wav_path = _write_wav(tmp_path / "boundary.wav", _bursty_tonal(duration=1.0))

    base = {"rms": 1.0, "flatness": 0.0, "chroma_top3": 1.0}

    def check(dr, cv):
        fake = dict(base, dynamic_range_db=dr, rms_cv=cv)
        monkeypatch.setattr(cut_qc, "_spectral_metrics", lambda y, sr: fake)
        return cut_qc._check_audio(wav_path)

    # exactly at both floors -> pass
    reasons = check(cut_qc._DYNAMIC_RANGE_MIN_DB, cut_qc._RMS_CV_MIN)
    assert not any("dynamic range" in r for r in reasons)

    # both just below -> reject
    reasons = check(cut_qc._DYNAMIC_RANGE_MIN_DB - 0.01, cut_qc._RMS_CV_MIN - 0.01)
    assert any("audio not musical" in r and "dynamic range" in r for r in reasons)

    # only dynamic range fails, rms_cv comfortably passes -> no reject
    # (AND logic: both must fail before this is treated as "not musical")
    reasons = check(cut_qc._DYNAMIC_RANGE_MIN_DB - 5.0, cut_qc._RMS_CV_MIN + 0.5)
    assert not any("dynamic range" in r for r in reasons)

    # only rms_cv fails, dynamic range comfortably passes -> no reject
    reasons = check(cut_qc._DYNAMIC_RANGE_MIN_DB + 5.0, cut_qc._RMS_CV_MIN - 0.2)
    assert not any("dynamic range" in r for r in reasons)


def test_chroma_and_flatness_thresholds_are_coarse_nets_not_musicality(tmp_path, monkeypatch):
    """chroma_top3 was lowered 0.40 -> 0.25 because it did not discriminate
    (calibration: broken spans 0.307-0.403, good spans 0.359-0.414); a value
    like 0.35 (a verified-good song in the calibration set) must NOT trip it
    anymore, while something far more concentrated than anything observed
    (e.g. 0.20) still should. Flatness was tightened 0.25 -> 0.08."""
    wav_path = _write_wav(tmp_path / "boundary2.wav", _bursty_tonal(duration=1.0))
    base = {"rms": 1.0, "dynamic_range_db": 100.0, "rms_cv": 1.0}

    def check(flatness, chroma_top3):
        fake = dict(base, flatness=flatness, chroma_top3=chroma_top3)
        monkeypatch.setattr(cut_qc, "_spectral_metrics", lambda y, sr: fake)
        return cut_qc._check_audio(wav_path)

    # a real calibration value (0.359, GOOD bath new) no longer trips chroma
    reasons = check(flatness=0.03, chroma_top3=0.359)
    assert not any("chroma" in r for r in reasons)

    # something far more concentrated than anything observed still trips it
    reasons = check(flatness=0.03, chroma_top3=0.20)
    assert any("chroma" in r for r in reasons)

    # flatness 0.09 (above new 0.08, below old 0.25) now correctly flags
    reasons = check(flatness=0.09, chroma_top3=0.9)
    assert any("flatness" in r for r in reasons)

    # flatness 0.05 stays under the new floor
    reasons = check(flatness=0.05, chroma_top3=0.9)
    assert not any("flatness" in r for r in reasons)


# --------------------------------------------------------- end-to-end ---
def test_check_audio_rejects_noisy_drone(tmp_path):
    path = _write_wav(tmp_path / "drone.wav", _noisy_drone())
    reasons = cut_qc._check_audio(path)
    assert any("audio not musical" in r for r in reasons)
    assert any("dynamic range" in r and "rms_cv" in r for r in reasons)


def test_check_audio_rejects_constant_tone_drone(tmp_path):
    """The exact failure mode the task flagged: under the OLD thresholds
    this constant tone would have nearly passed."""
    path = _write_wav(tmp_path / "tone_drone.wav", _constant_tone_drone())
    reasons = cut_qc._check_audio(path)
    assert any("audio not musical" in r for r in reasons)


def test_check_audio_accepts_bursty_tonal_signal(tmp_path):
    path = _write_wav(tmp_path / "tonal.wav", _bursty_tonal())
    reasons = cut_qc._check_audio(path)
    assert reasons == []


def test_check_audio_rejects_silence(tmp_path):
    y = np.zeros(int(SR * 2.0), dtype=np.float32)
    path = _write_wav(tmp_path / "silent.wav", y)
    reasons = cut_qc._check_audio(path)
    assert any("audio silent" in r for r in reasons)


# ------------------------------------------------- calibration sanity ---
@pytest.mark.parametrize(
    "label,flat,top3,dynamic_range_db,rms_cv,expect_reject",
    [
        ("BROKEN bath old", 0.0296, 0.403, 2.7, 0.107, True),
        ("BROKEN rainbow", 0.1101, 0.307, 2.9, 0.104, True),
        ("BROKEN v2", 0.0910, 0.379, 4.7, 0.196, True),
        ("mid counting1", 0.0274, 0.352, 19.9, 0.492, False),
        ("mid counting2", 0.0250, 0.358, 17.3, 0.454, False),
        ("mid counting3", 0.0153, 0.432, 13.2, 0.416, False),
        ("GOOD bath new", 0.0287, 0.359, 21.3, 0.543, False),
        ("GOOD official env", 0.0131, 0.414, 45.3, 0.936, False),
    ],
)
def test_calibration_table_reclassifies_correctly(
    tmp_path, monkeypatch, label, flat, top3, dynamic_range_db, rms_cv, expect_reject
):
    """Replays the 8 measured calibration samples (rms fixed well above the
    silence floor, since that wasn't part of this measurement) through the
    real _check_audio threshold logic and confirms every BROKEN sample is
    rejected and every other sample is accepted under the new gate."""
    wav_path = _write_wav(tmp_path / "calib.wav", _bursty_tonal(duration=1.0))
    fake = {
        "rms": 1.0,
        "flatness": flat,
        "chroma_top3": top3,
        "dynamic_range_db": dynamic_range_db,
        "rms_cv": rms_cv,
    }
    monkeypatch.setattr(cut_qc, "_spectral_metrics", lambda y, sr: fake)
    reasons = cut_qc._check_audio(wav_path)
    is_rejected = any("audio not musical" in r for r in reasons)
    assert is_rejected == expect_reject, (label, reasons)
