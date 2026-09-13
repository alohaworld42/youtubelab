"""Tests for pipeline/kidsong/audio_clean.py and the cut_qc opening-noise gate.

Everything here is synthesized with numpy/soundfile -- no network, no GPU, no
real songs on disk. The signals mirror what was measured on five real ACE-Step
renders (see the audio_clean module docstring for the numbers):

  * a loud broadband burst at t=0 followed by music  -> the failure mode,
  * music opening on a loud tonal downbeat            -> must NOT be trimmed,
  * leading silence                                   -> trimmed,
  * trailing dead air                                 -> trimmed (1.5-13.3s
    measured on the real renders).

The negative control (`_downbeat_music`) is the important one: the junk in the
real renders is LOUD, so any amplitude-based trimmer would eat a real downbeat.
"""
import numpy as np
import pytest

from pipeline.kidsong import audio_clean, cut_qc

soundfile = pytest.importorskip("soundfile")
pytest.importorskip("librosa")

SR = 22050


# --------------------------------------------------------------- signals ---
def _music(duration=6.0, sr=SR, level=0.6, note=0.5):
    """Tonal, pitched, with note onsets and loud/quiet structure -- the shape
    every "this is real music" assertion below relies on."""
    n = int(sr * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    sig = np.zeros_like(t)
    for f in (261.63, 329.63, 392.00):  # C4 E4 G4
        sig += np.sin(2 * np.pi * f * t)
    sig = sig / 3.0 * level
    env = np.where((t % (2 * note)) < note, 1.0, 0.08)
    return (sig * env).astype(np.float32)


def _noise(duration, sr=SR, level=0.3, seed=0):
    rng = np.random.default_rng(seed)
    return np.clip(rng.normal(0, level, int(sr * duration)), -1.0, 1.0).astype(np.float32)


def _downbeat_music(duration=6.0, sr=SR):
    """THE NEGATIVE CONTROL: music that legitimately opens on a loud downbeat.

    Full-level tonal hit exactly at t=0, decaying into the following beat, and
    the arrangement continues. There is no gap and no noise. A trimmer that
    keys on amplitude ("loud thing at the start") would cut this; the detector
    must leave it completely alone.
    """
    n = int(sr * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    sig = np.zeros_like(t)
    for f in (261.63, 329.63, 392.00, 523.25):
        sig += np.sin(2 * np.pi * f * t)
    sig /= 4.0
    env = 0.35 + 0.65 * np.exp(-(t % 0.5) * 9.0)  # first attack lands at t=0
    return (sig * env * 0.7).astype(np.float32)


def _write(path, y, sr=SR):
    soundfile.write(str(path), y, sr)
    return str(path)


def _flatness(y, sr=SR, seconds=None):
    import librosa

    if seconds is not None:
        y = y[: int(seconds * sr)]
    return float(np.mean(librosa.feature.spectral_flatness(y=y)))


CFG = {"kidsong": {"audio_clean": {"enabled": True}}}


# ------------------------------------------------------------- detection ---
def test_noise_burst_is_removed_and_music_start_found():
    """A 0.5s broadband burst in front of music: the burst must be trimmed and
    the detected start must land on the music within a frame or two."""
    burst = 0.5
    y = np.concatenate([_noise(burst), _music(8.0)])
    trim, reason = audio_clean._detect_start(y, SR, max_trim_seconds=2.0)

    assert trim == pytest.approx(burst, abs=0.12), reason
    assert "junk" in reason and "noise-like" in reason


def test_loud_downbeat_is_never_trimmed():
    """The negative control. Music opening on a loud tonal downbeat must be
    left completely untouched -- this is what separates a musical criterion
    from an amplitude gate."""
    y = _downbeat_music()
    trim, reason = audio_clean._detect_start(y, SR, max_trim_seconds=2.0)

    assert trim == 0.0, reason


def test_loud_downbeat_survives_full_clean_pipeline(tmp_path):
    """End-to-end negative control: nothing trimmed off either end."""
    path = _write(tmp_path / "downbeat.wav", _downbeat_music(8.0))
    meta = audio_clean.clean_song(path, cfg=CFG)

    assert meta["head_trimmed"] == 0.0, meta["reason"]
    assert meta["tail_trimmed"] == 0.0, meta["reason"]
    assert meta["duration"] == pytest.approx(8.0, abs=0.05)


def test_leading_silence_is_trimmed():
    y = np.concatenate([np.zeros(int(SR * 0.8), dtype=np.float32), _music(8.0)])
    trim, reason = audio_clean._detect_start(y, SR, max_trim_seconds=2.0)

    assert trim == pytest.approx(0.8, abs=0.12), reason
    # reported as silence, not as noise -- digital silence has a degenerate
    # flatness near 1.0 and must not be described as "noise-like"
    assert "silent" in reason


def test_trailing_junk_is_trimmed(tmp_path):
    """Dead air at the end -- the dominant real-world tail case (1.5-13.3s
    measured). The tail is found by running the same analysis reversed."""
    y = np.concatenate([_music(8.0), np.zeros(int(SR * 3.0), dtype=np.float32)])
    path = _write(tmp_path / "trailing.wav", y)
    meta = audio_clean.clean_song(path, cfg=CFG)

    assert meta["tail_trimmed"] == pytest.approx(3.0, abs=0.15), meta["reason"]
    assert meta["head_trimmed"] == 0.0
    assert meta["duration"] == pytest.approx(8.0, abs=0.2)


def test_trailing_noise_burst_is_trimmed(tmp_path):
    y = np.concatenate([_music(8.0), _noise(1.0, seed=3)])
    path = _write(tmp_path / "tailnoise.wav", y)
    meta = audio_clean.clean_song(path, cfg=CFG)

    assert meta["tail_trimmed"] == pytest.approx(1.0, abs=0.15), meta["reason"]


# ------------------------------------------------------------ safety cap ---
def test_max_head_trim_cap_limits_the_trim():
    """A 3s burst against a 1s cap: the trim is clamped to the cap, never more,
    and the reason says so. The cap is the guarantee that a bad detection can
    never eat the song."""
    y = np.concatenate([_noise(3.0, seed=1), _music(8.0)])
    trim, reason = audio_clean._detect_start(y, SR, max_trim_seconds=1.0)

    assert trim == pytest.approx(1.0, abs=0.03), reason
    assert "safety cap" in reason


def test_cap_region_must_still_be_junk_to_be_trimmed():
    """Clamping to the cap must not hand out a free pass: if the region the cap
    leaves is NOT junk, nothing is trimmed. Here the first second is real music
    (only the later part would qualify), so a 1s cap must trim nothing."""
    y = np.concatenate([_music(4.0), _noise(0.4, seed=5), _music(6.0)])
    trim, reason = audio_clean._detect_start(y, SR, max_trim_seconds=1.0)

    assert trim == 0.0, reason


def test_cleaner_refuses_to_leave_less_than_a_second(tmp_path, monkeypatch):
    """Head + tail must never meet in the middle. Forced by stubbing the
    detector, since making a real signal produce two overlapping detections
    would be testing the fixture rather than the guard."""
    path = _write(tmp_path / "short.wav", _music(4.0))
    monkeypatch.setattr(
        audio_clean, "_detect_start", lambda y, sr, max_trim_seconds: (2.0, "stub")
    )

    meta = audio_clean.clean_song(path, cfg=CFG)

    assert meta["applied"] is False
    assert "would leave" in meta["reason"]
    assert meta["duration"] == pytest.approx(4.0, abs=0.05)
    assert soundfile.info(path).frames / SR == pytest.approx(4.0, abs=0.05)


def test_all_noise_input_is_left_alone(tmp_path):
    """Audio with no musical content anywhere: the detector must decline rather
    than invent a start (the trimmer is not a quality gate -- cut_qc is)."""
    path = _write(tmp_path / "allnoise.wav", _noise(4.0, seed=7))
    meta = audio_clean.clean_song(path, cfg=CFG)

    assert meta["head_trimmed"] == 0.0
    assert meta["tail_trimmed"] == 0.0
    assert "no sustained musical window" in meta["reason"]


# ----------------------------------------------------------------- fades ---
def test_fade_removes_the_click_at_both_edges(tmp_path):
    """After trimming, the cut lands mid-waveform. Without a fade the first
    sample is a step from zero -- a click. Check the edges are near silent and
    that the ramp is monotonic rather than an abrupt jump."""
    y = np.concatenate([_noise(0.5, seed=2), _music(8.0)])
    path = _write(tmp_path / "faded.wav", y)
    meta = audio_clean.clean_song(path, cfg=CFG)
    assert meta["applied"] is True

    out, sr = soundfile.read(path, dtype="float32", always_2d=False)
    out = out if out.ndim == 1 else out.mean(axis=1)

    assert abs(float(out[0])) < 1e-3
    assert abs(float(out[-1])) < 1e-3
    # the first few ms must ramp up, not jump
    fade_n = int(0.030 * sr)
    head_peak = float(np.max(np.abs(out[:fade_n])))
    later_peak = float(np.max(np.abs(out[fade_n:3 * fade_n])))
    assert head_peak <= later_peak


def test_apply_fades_is_shape_preserving_and_handles_stereo():
    mono = np.ones(SR, dtype=np.float32)
    faded = audio_clean._apply_fades(mono, SR, 30.0)
    assert faded.shape == mono.shape
    assert faded[0] == pytest.approx(0.0, abs=1e-6)
    assert faded[SR // 2] == pytest.approx(1.0)

    stereo = np.ones((SR, 2), dtype=np.float32)
    faded_s = audio_clean._apply_fades(stereo, SR, 30.0)
    assert faded_s.shape == stereo.shape
    assert faded_s[0, 0] == pytest.approx(0.0, abs=1e-6)
    assert faded_s[0, 1] == pytest.approx(0.0, abs=1e-6)


def test_zero_fade_is_a_noop():
    y = np.ones(1000, dtype=np.float32)
    assert np.array_equal(audio_clean._apply_fades(y, SR, 0.0), y)


# -------------------------------------------------------------- metadata ---
def test_metadata_is_consistent(tmp_path):
    head, tail, body = 0.5, 2.0, 8.0
    y = np.concatenate([
        _noise(head, seed=4),
        _music(body),
        np.zeros(int(SR * tail), dtype=np.float32),
    ])
    path = _write(tmp_path / "meta.wav", y)
    total = len(y) / SR

    meta = audio_clean.clean_song(path, cfg=CFG)

    assert meta["enabled"] is True
    assert meta["applied"] is True
    assert meta["duration_before"] == pytest.approx(total, abs=0.02)
    assert meta["detected_start"] == pytest.approx(meta["head_trimmed"])
    assert meta["detected_end"] == pytest.approx(total - meta["tail_trimmed"], abs=0.02)
    # duration must describe the file that actually shipped
    assert meta["duration"] == pytest.approx(
        meta["duration_before"] - meta["head_trimmed"] - meta["tail_trimmed"], abs=0.05
    )
    assert meta["audio_path"] == path
    assert meta["reason"]
    info = soundfile.info(path)
    assert meta["duration"] == pytest.approx(info.frames / info.samplerate, abs=0.02)


def test_disabled_leaves_the_file_untouched(tmp_path):
    y = np.concatenate([_noise(0.5, seed=6), _music(8.0)])
    path = _write(tmp_path / "off.wav", y)
    before = soundfile.read(path, dtype="float32")[0]

    meta = audio_clean.clean_song(path, cfg={"kidsong": {"audio_clean": {"enabled": False}}})

    assert meta["enabled"] is False
    assert meta["applied"] is False
    assert meta["head_trimmed"] == 0.0
    assert np.array_equal(soundfile.read(path, dtype="float32")[0], before)


def test_missing_file_reports_rather_than_raises(tmp_path):
    """A cleanup failure must never kill a 40-minute render."""
    meta = audio_clean.clean_song(str(tmp_path / "nope.wav"), cfg=CFG)
    assert meta["applied"] is False
    assert "skipped" in meta["reason"]


def test_settings_defaults_and_overrides():
    assert audio_clean.settings_from_cfg(None)["enabled"] is True
    assert audio_clean.settings_from_cfg({})["max_head_trim_seconds"] == 2.0
    # tail cap is deliberately far more generous than the head cap
    d = audio_clean.settings_from_cfg({})
    assert d["max_tail_trim_seconds"] > d["max_head_trim_seconds"]

    over = audio_clean.settings_from_cfg(
        {"kidsong": {"audio_clean": {"fade_ms": 12, "loudness_lufs": -16}}}
    )
    assert over["fade_ms"] == 12.0
    assert over["loudness_lufs"] == -16.0
    assert over["max_head_trim_seconds"] == 2.0  # untouched keys keep defaults


def test_runlog_receives_the_decision(tmp_path):
    """The trim decision must be logged through a run logger when one exists,
    and callers without one must still work."""
    messages = []

    class FakeRunLog:
        def info(self, msg, *args):
            messages.append(msg % args if args else msg)

    y = np.concatenate([_noise(0.5, seed=8), _music(8.0)])
    path = _write(tmp_path / "logged.wav", y)
    audio_clean.clean_song(path, cfg=CFG, runlog=FakeRunLog())

    assert any("audio_clean" in m for m in messages)
    assert any("head -" in m for m in messages)


def test_out_path_preserves_the_original(tmp_path):
    """Generated media is channel inventory: cleaning to a separate path must
    not modify the source file."""
    y = np.concatenate([_noise(0.5, seed=9), _music(8.0)])
    src = _write(tmp_path / "orig.wav", y)
    dst = str(tmp_path / "cleaned.wav")
    before = soundfile.read(src, dtype="float32")[0]

    meta = audio_clean.clean_song(src, cfg=CFG, out_path=dst)

    assert meta["applied"] is True
    assert meta["audio_path"] == dst
    assert np.array_equal(soundfile.read(src, dtype="float32")[0], before)
    # Compare durations, not frame counts: loudnorm re-emits at 44.1 kHz stereo.
    info = soundfile.info(dst)
    assert info.frames / info.samplerate < len(y) / SR


# ------------------------------------------------ opening flatness gate ---
def test_opening_metrics_flag_a_noisy_opening():
    """The cut_qc backstop, at the metric level."""
    noisy = np.concatenate([_noise(1.0, seed=10), _music(20.0)])
    clean = _music(20.0)

    m_noisy = cut_qc._opening_metrics(noisy, SR)
    m_clean = cut_qc._opening_metrics(clean, SR)

    assert m_noisy["opening_flatness"] >= cut_qc._OPENING_FLATNESS_MAX
    assert m_noisy["ratio"] >= cut_qc._OPENING_FLATNESS_RATIO
    assert m_clean["opening_flatness"] < cut_qc._OPENING_FLATNESS_MAX


def test_opening_metrics_returns_none_for_a_short_clip():
    assert cut_qc._opening_metrics(_music(1.0), SR) is None


def test_check_audio_rejects_a_noisy_opening(tmp_path):
    path = _write(tmp_path / "noisy_open.wav", np.concatenate([_noise(1.0, seed=11), _music(20.0)]))
    reasons = cut_qc._check_audio(path)

    assert any(r.startswith("audio noisy opening:") for r in reasons), reasons


def test_check_audio_accepts_a_clean_opening(tmp_path):
    path = _write(tmp_path / "clean_open.wav", _music(20.0))
    reasons = cut_qc._check_audio(path)

    assert not any("noisy opening" in r for r in reasons), reasons


def test_check_audio_accepts_a_loud_downbeat_opening(tmp_path):
    """The QC gate must not punish a song for opening loudly either."""
    path = _write(tmp_path / "downbeat_open.wav", _downbeat_music(20.0))
    reasons = cut_qc._check_audio(path)

    assert not any("noisy opening" in r for r in reasons), reasons


def test_cleaning_a_noisy_opening_makes_it_pass_qc(tmp_path):
    """The two halves joined up: audio that FAILS the QC gate must PASS it
    after audio_clean has run. This is the whole contract."""
    y = np.concatenate([_noise(1.0, seed=12), _music(20.0)])
    path = _write(tmp_path / "roundtrip.wav", y)

    before = cut_qc._check_audio(path)
    assert any("noisy opening" in r for r in before), before

    meta = audio_clean.clean_song(path, cfg=CFG)
    assert meta["applied"] is True
    assert meta["head_trimmed"] == pytest.approx(1.0, abs=0.15)

    after = cut_qc._check_audio(path)
    assert not any("noisy opening" in r for r in after), after


def test_noisy_opening_has_a_weight_and_lowers_the_score():
    """Reason-string shape and _CUT_WEIGHTS wiring, matching the existing
    categories: '<category>: <detail>' and a registered weight."""
    assert "audio noisy opening" in cut_qc._CUT_WEIGHTS

    verdict = cut_qc._verdict(
        ["audio noisy opening: first 1.5s spectral flatness 0.38 >= 0.1"],
        cut_qc._CUT_WEIGHTS,
        "recut",
    )
    assert verdict["accept"] is False
    assert verdict["score"] == pytest.approx(1.0 - cut_qc._CUT_WEIGHTS["audio noisy opening"])
    assert verdict["retry_hints"]["recut"] is True
