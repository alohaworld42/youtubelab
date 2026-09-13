"""
song_audio.py — Vocals + music bed for the kids-song pipeline.

synthesize_vocals() flattens a list of verses into plain lyric lines and reuses
pipeline.tts.synthesize (edge-tts) with a kid-friendly narrator voice, then regroups
the resulting per-line segments back into per-verse (start, end) windows.

make_music_bed() procedurally renders a cheerful, music-box-style nursery backing
track with numpy and writes it straight out as a WAV via the stdlib `wave` module
(no ffmpeg / pydub needed for this part — it's pure synthesis).
"""
import copy
import math
import os
import wave

import numpy as np

from pipeline import tts

SAMPLE_RATE = 44100


# ------------------------------------------------------------------- vocals ---
def synthesize_vocals(verses, cfg, out_path):
    """Synthesize the sung/spoken lyric lines for every verse.

    `verses` is a list of {"lines": [str, ...], "scene": str}. All lines are
    flattened (in order) into narrator lines, synthesized with tts.synthesize,
    then the returned segments are regrouped back into per-verse time windows.
    """
    kidsong_cfg = cfg.get("kidsong", {})
    voice = kidsong_cfg.get("voice", "en-US-AnaNeural")
    rate = kidsong_cfg.get("voice_rate", "+4%")

    # edge-tts fallback singer only (the ACE-Step singer ignores this voice).
    # For a German channel whose voice is still an English default, sing in a
    # German voice instead of forcing German lyrics through an en-US voice —
    # otherwise the "vocals" mispronounce every word. An explicitly-set de-*
    # (or any non-en) voice is respected as-is.
    if str(kidsong_cfg.get("language", "en")).strip().lower() == "de" and voice.startswith("en-"):
        voice = "de-DE-KatjaNeural"

    # Deep-copy so we don't clobber the caller's voice config for other stages.
    cfg_copy = copy.deepcopy(cfg)
    cfg_copy["voices"] = {"narrator": voice, "rate": rate}

    flat_lines = []
    line_counts = []
    for verse in verses:
        lines = verse["lines"]
        line_counts.append(len(lines))
        for line in lines:
            flat_lines.append({"speaker": "narrator", "text": line})

    result = tts.synthesize(flat_lines, cfg_copy, out_path)

    # Regroup the flat per-line segments back into one (start, end) per verse.
    segments = result["segments"]
    verse_times = []
    idx = 0
    for i, count in enumerate(line_counts):
        verse_segs = segments[idx : idx + count]
        start = verse_segs[0]["start"]
        end = verse_segs[-1]["end"]
        if i == len(line_counts) - 1:
            end = result["duration"]  # last verse always runs to the very end
        verse_times.append((start, end))
        idx += count

    result["verse_times"] = verse_times
    return result


# --------------------------------------------------------------- music bed ---
NOTE_FREQ = {
    "C3": 130.81, "F3": 174.61, "G3": 196.00, "A3": 220.00,
    "C4": 261.63, "D4": 293.66, "E4": 329.63, "F4": 349.23,
    "G4": 392.00, "A4": 440.00, "B4": 493.88,
    "C5": 523.25, "D5": 587.33, "E5": 659.25,
}

# I - V - vi - IV in C major (the "Axis" progression — cheerful and familiar).
# Each chord: a low root note held for the chord, plus an arpeggio played on
# eighth notes over the chord tones.
PROGRESSION = [
    {"root": "C3", "arp": ["C4", "E4", "G4", "C5"]},  # I   (C major)
    {"root": "G3", "arp": ["G4", "B4", "D5", "G4"]},  # V   (G major)
    {"root": "A3", "arp": ["A4", "C5", "E5", "A4"]},  # vi  (A minor)
    {"root": "F3", "arp": ["F4", "A4", "C5", "F4"]},  # IV  (F major)
]


def _fade(env, sample_rate, fade_s=0.006):
    """Apply a short linear fade-in/out to an envelope so notes never click."""
    n = len(env)
    fade_n = min(int(sample_rate * fade_s), n // 2)
    if fade_n > 0:
        env[:fade_n] *= np.linspace(0.0, 1.0, fade_n)
        env[-fade_n:] *= np.linspace(1.0, 0.0, fade_n)
    return env


def _pluck(freq, num_samples, sample_rate, amp=0.28, decay_tau=None):
    """Music-box/celesta pluck: fundamental + a quiet octave harmonic, fast decay."""
    t = np.arange(num_samples) / sample_rate
    tau = decay_tau if decay_tau is not None else (num_samples / sample_rate) / 4.0
    decay = np.exp(-t / tau)
    tone = 0.82 * np.sin(2 * math.pi * freq * t) + 0.18 * np.sin(2 * math.pi * 2 * freq * t)
    env = _fade(decay.copy(), sample_rate)
    return amp * tone * env


def _root_tone(freq, num_samples, sample_rate, amp=0.14):
    """Soft, low sustained sine so each chord has a gentle harmonic foundation."""
    t = np.arange(num_samples) / sample_rate
    tone = np.sin(2 * math.pi * freq * t)
    env = _fade(np.ones(num_samples), sample_rate, fade_s=0.03)
    return amp * tone * env


def _add_stereo(buf, start, mono, pan_left, pan_right):
    """Mix a mono note into the stereo buffer at `start`, clipping to buffer length."""
    n = min(len(mono), len(buf) - start)
    if n <= 0:
        return
    buf[start : start + n, 0] += mono[:n] * pan_left
    buf[start : start + n, 1] += mono[:n] * pan_right


def _build_loop(tempo_bpm, sample_rate):
    """Render one pass of the chord progression as a seamless-looping stereo buffer."""
    beat_dur = 60.0 / tempo_bpm
    chord_dur = 2 * beat_dur
    eighth_dur = beat_dur / 2.0

    loop_samples = int(round(len(PROGRESSION) * chord_dur * sample_rate))
    buf = np.zeros((loop_samples, 2), dtype=np.float64)

    for chord_i, chord in enumerate(PROGRESSION):
        chord_start = int(round(chord_i * chord_dur * sample_rate))
        chord_n = int(round(chord_dur * sample_rate))

        # Soft sustained root note under the whole chord, centered.
        root = _root_tone(NOTE_FREQ[chord["root"]], chord_n, sample_rate)
        _add_stereo(buf, chord_start, root, 0.5, 0.5)

        # Arpeggiate the chord tones on eighth notes, alternating pan for width.
        num_eighths = int(round(chord_dur / eighth_dur))
        note_n = int(round(eighth_dur * 0.9 * sample_rate))
        for j in range(num_eighths):
            note_start = chord_start + int(round(j * eighth_dur * sample_rate))
            freq = NOTE_FREQ[chord["arp"][j % len(chord["arp"])]]
            note = _pluck(freq, note_n, sample_rate)
            if j % 2 == 0:
                pan_left, pan_right = 0.75, 0.35
            else:
                pan_left, pan_right = 0.35, 0.75
            _add_stereo(buf, note_start, note, pan_left, pan_right)

    return buf


def make_music_bed(duration, out_path, tempo_bpm=96):
    """Procedurally generate a gentle nursery-style music bed and write it as a WAV.

    Loops the chord progression to fill `duration` seconds exactly, keeps the mix
    well below clipping (peak < -6 dBFS) to leave headroom for the vocal track,
    and fades the very end so trimming never produces a click.
    """
    loop = _build_loop(tempo_bpm, SAMPLE_RATE)

    # Normalize so the loudest sample sits comfortably below -6 dBFS (~0.501).
    peak = np.max(np.abs(loop))
    target_peak = 0.45  # ~ -6.9 dBFS
    if peak > 0:
        loop *= target_peak / peak

    total_samples = int(round(duration * SAMPLE_RATE))
    reps = max(1, math.ceil(total_samples / len(loop)))
    full = np.tile(loop, (reps, 1))[:total_samples]

    # Fade the last ~30ms so the hard trim doesn't produce a click.
    fade_n = min(int(SAMPLE_RATE * 0.03), len(full))
    if fade_n > 0:
        full[-fade_n:] *= np.linspace(1.0, 0.0, fade_n)[:, None]

    pcm = np.clip(full * 32767.0, -32768, 32767).astype(np.int16)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with wave.open(out_path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())

    return out_path


if __name__ == "__main__":
    import wave as _wave

    from pydub import AudioSegment

    from pipeline.config import load_config

    cfg = load_config()
    out_dir = os.path.join(cfg["_root"], "output", "kidsong_audio_test")
    os.makedirs(out_dir, exist_ok=True)

    verses = [
        {
            "lines": [
                "Little duck goes for a swim,",
                "Paddling paws and feathers trim,",
                "Splashing in the morning sun.",
            ],
            "scene": "duck swimming in a pond",
        },
        {
            "lines": [
                "Bunny hops across the field,",
                "Carrots found beneath the shield,",
                "Home before the day is done.",
            ],
            "scene": "bunny hopping in a meadow",
        },
    ]

    vocals_path = os.path.join(out_dir, "vocals.wav")
    music_path = os.path.join(out_dir, "music.wav")

    print("Synthesizing vocals...")
    vocal_result = synthesize_vocals(verses, cfg, vocals_path)
    print("duration:", vocal_result["duration"])
    print("verse_times:", vocal_result["verse_times"])

    print("Rendering music bed...")
    make_music_bed(vocal_result["duration"], music_path)

    # ---- verification ----
    assert os.path.exists(vocals_path), "vocals.wav missing"
    assert os.path.exists(music_path), "music.wav missing"
    assert os.path.getsize(vocals_path) > 1000, "vocals.wav suspiciously small"
    assert os.path.getsize(music_path) > 1000, "music.wav suspiciously small"

    with _wave.open(music_path, "rb") as wf:
        music_duration = wf.getnframes() / wf.getframerate()
    print("music duration:", music_duration)
    assert abs(music_duration - vocal_result["duration"]) < 0.1, "music duration mismatch"

    vocal_audio = AudioSegment.from_file(vocals_path)
    print("vocals dBFS:", vocal_audio.dBFS)
    assert vocal_audio.dBFS > -50, "vocals track is unexpectedly silent"

    music_audio = AudioSegment.from_file(music_path)
    print("music dBFS:", music_audio.dBFS, "peak dBFS:", music_audio.max_dBFS)

    print("All checks passed.")
