"""
tts.py — Turn the script lines into one audio track using edge-tts (free, no API key).

Each line is synthesized to its own file with the voice mapped to its speaker, then
all lines are concatenated (with a small gap) into one track. We record the start/end
time of every line so the video stage knows which character is "talking" when.

Returns:
  {
    "audio_path": "/abs/path/voice.wav",
    "duration": 31.4,
    "segments": [
        {"speaker": "speaker_a", "text": "...", "start": 0.0, "end": 2.3},
        ...
    ]
  }
"""
import asyncio
import os
import tempfile

import edge_tts
from pydub import AudioSegment

GAP_MS = 180  # small pause between lines so it doesn't sound rushed


def _rmtree_quiet(path):
    import shutil

    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass


def _voice_for(speaker, voices):
    return voices.get(speaker, voices.get("narrator"))


async def _synth_line(text, voice, rate, out_path):
    communicate = edge_tts.Communicate(text, voice, rate=rate)
    await communicate.save(out_path)


def _synth_line_with_retry(text, voice, rate, out_path, attempts=3):
    """edge-tts occasionally throttles (403) — retry with backoff before giving up."""
    import time as _time

    for i in range(attempts):
        try:
            asyncio.run(_synth_line(text, voice, rate, out_path))
            return
        except Exception:
            if i == attempts - 1:
                raise
            _time.sleep(2 * (i + 1))


def synthesize(lines, cfg, out_path):
    voices = cfg["voices"]
    rate = voices.get("rate", "+0%")

    tmpdir = tempfile.mkdtemp(prefix="brainrot_tts_")
    combined = AudioSegment.silent(duration=0)
    gap = AudioSegment.silent(duration=GAP_MS)
    segments = []

    try:
        for i, ln in enumerate(lines):
            voice = _voice_for(ln["speaker"], voices)
            part_path = os.path.join(tmpdir, f"line_{i:03d}.mp3")
            _synth_line_with_retry(ln["text"], voice, rate, part_path)

            seg_audio = AudioSegment.from_file(part_path)
            start_ms = len(combined)
            combined += seg_audio
            end_ms = len(combined)
            combined += gap

            segments.append(
                {
                    "speaker": ln["speaker"],
                    "text": ln["text"],
                    "start": round(start_ms / 1000.0, 3),
                    "end": round(end_ms / 1000.0, 3),
                }
            )

        # Export a wav so Whisper + MoviePy are happy.
        combined.export(out_path, format="wav")
        duration = round(len(combined) / 1000.0, 3)
    finally:
        # try/FINALLY: the cleanup used to sit after the loop, so any failure in
        # it — edge-tts throttling past its retries is the common one — left the
        # scratch dir and its mp3s behind. A retried job leaks one per attempt.
        _rmtree_quiet(tmpdir)

    return {"audio_path": out_path, "duration": duration, "segments": segments}


if __name__ == "__main__":
    from pipeline.config import load_config

    cfg = load_config()
    demo = [
        {"speaker": "speaker_a", "text": "Bro, did you know octopuses have three hearts?"},
        {"speaker": "speaker_b", "text": "No way. That's actually insane."},
    ]
    out = synthesize(demo, cfg, os.path.join(cfg["_root"], "output", "_tts_test.wav"))
    print(out)
