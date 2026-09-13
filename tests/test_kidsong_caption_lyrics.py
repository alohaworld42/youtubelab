"""Regression test for the "captions never showed sung lyrics" bug.

Root cause (fixed in pipeline/captions.py, cc093a0): Silero VAD discarded sung
vocals over an instrumental bed, so Whisper returned 0 words and verse timing
fell back to a proportional split with nothing to caption. The fix added
pipeline/alignment.py, which force-aligns the KNOWN lyrics onto the ASR
transcript's clock -- but only when get_word_timestamps() is actually called
with ``lyrics=`` (and ``duration=``). If either of the two call sites in
pipeline/kidsong/generate.py silently drops the kwarg, captions degrade back
to raw-transcript (or empty) with no visible error -- exactly the kind of
regression that shipped unnoticed before. This test asserts the kwarg is
forwarded, not just that the module imports cleanly.

Two call sites:
  1. pipeline.kidsong.generate._run           -- the "classic" (non-director)
     mode, edge-tts/procedural-bed path.
  2. pipeline.kidsong.generate._generate_director -- the default "director"
     mode, ACE-Step-sung path.

CPU-only, no network, no GPU, no ffmpeg, no Whisper model load: every stage
around the captions step is monkeypatched, following the same shape as
tests/test_kidsong_intro_promotion.py's `promotable_run` fixture.
"""
import json
import os
import struct
import wave

import pytest

from pipeline.kidsong import runstate
from pipeline.kidsong.comfy import ComfyClient as _RealComfyClient

_REAL_LOAD_WORKFLOW = _RealComfyClient.load_workflow


def _write_wav(path, seconds=4.0, rate=8000):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(struct.pack("<h", 0) * int(rate * seconds))


def _take(shots_dir, shot_id, attempt, size=1024):
    os.makedirs(shots_dir, exist_ok=True)
    p = os.path.join(shots_dir, f"{shot_id}_a{attempt}.mp4")
    with open(p, "wb") as f:
        f.write(b"\0" * size)
    return p


def _make_song():
    return {
        "title": "Caption Lyrics Test Song",
        "description": "d",
        "tags": ["t"],
        "characters": "Three toddlers: Zuri; Kofi; Nala",
        "verses": [
            {"scene": "backyard", "lines": ["brush brush brush", "your teeth today"]},
            {"scene": "bathroom", "lines": ["clean clean clean", "in every way"]},
        ],
    }


# --------------------------------------------------------- site 1: _run() ---
@pytest.fixture
def classic_run(tmp_path, monkeypatch):
    """Drives pipeline.kidsong.generate._run via kidsong.mode="classic" (the
    non-director/legacy path), with singer != "ace" so it goes straight to
    the edge-tts fallback and never touches ComfyUI/GPU. Captures every call
    to captions.get_word_timestamps.
    """
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    base = "20260720-091500-kidsong-classic-caption-test"

    song = _make_song()
    (out_dir / f"{base}-song.json").write_text(json.dumps(song), encoding="utf-8")

    calls = []  # list of (args, kwargs) for each get_word_timestamps call

    import pipeline.captions as captions_mod
    import pipeline.kidsong.assemble_song as assemble_song_mod
    import pipeline.kidsong.scenes as scenes_mod
    import pipeline.kidsong.song_audio as song_audio_mod

    def fake_synthesize_vocals(verses, cfg, out_path):
        _write_wav(out_path, seconds=4.0)
        return {"duration": 4.0, "verse_times": [(0.0, 2.0), (2.0, 4.0)]}

    def fake_make_music_bed(duration, out_path, tempo_bpm=96):
        with open(out_path, "wb") as f:
            f.write(b"\0" * 64)

    def fake_render_scenes(scene_descriptions, characters, out_dir, cfg, on_progress=None):
        os.makedirs(out_dir, exist_ok=True)
        paths = []
        for i in range(len(scene_descriptions)):
            p = os.path.join(out_dir, f"scene_{i:02d}.png")
            with open(p, "wb") as f:
                f.write(b"\0" * 16)
            paths.append(p)
        return paths

    def fake_get_word_timestamps(*args, **kwargs):
        calls.append((args, kwargs))
        return []

    def fake_build_song_video(cfg, voice_path, music_path, scene_media, verse_times,
                               words, duration, out_path, on_progress=None):
        with open(out_path, "wb") as f:
            f.write(b"\0" * 4096)

    monkeypatch.setattr(song_audio_mod, "synthesize_vocals", fake_synthesize_vocals)
    monkeypatch.setattr(song_audio_mod, "make_music_bed", fake_make_music_bed)
    monkeypatch.setattr(scenes_mod, "render_scenes", fake_render_scenes)
    monkeypatch.setattr(captions_mod, "get_word_timestamps", fake_get_word_timestamps)
    monkeypatch.setattr(assemble_song_mod, "build_song_video", fake_build_song_video)

    cfg = {
        "_root": str(tmp_path),
        "paths": {"output_dir": str(out_dir)},
        "video": {"width": 1080, "height": 1920, "fps": 24, "max_seconds": 60},
        "captions": {"uppercase": True},
        "kidsong": {
            "mode": "classic",
            "singer": "edge-tts",  # anything != "ace" -> skip ACE/ComfyUI entirely
            "animate": False,      # skip the i2v pass -> no GPU/ComfyUI touch
        },
    }

    return {"cfg": cfg, "base": base, "out_dir": str(out_dir), "song": song, "calls": calls}


def test_run_forwards_lyrics_and_duration(classic_run):
    """Call site 1 (pipeline.kidsong.generate._run, ~line 172-177)."""
    from pipeline.kidsong.generate import generate_kidsong

    run = classic_run
    result = generate_kidsong(cfg=run["cfg"], resume_base=run["base"])

    assert result["video_path"]
    assert len(run["calls"]) == 1
    args, kwargs = run["calls"][0]
    assert kwargs.get("lyrics") == run["song"]["verses"]
    assert kwargs.get("duration") == result["duration"]


# ----------------------------------------- site 2: _generate_director() ---
@pytest.fixture
def director_run(tmp_path, monkeypatch):
    """Drives pipeline.kidsong.generate._generate_director with 2 shots
    already fully rendered (nothing pending -> no ComfyUI network calls),
    following the same shape as test_kidsong_intro_promotion.py's
    `promotable_run` fixture. Captures every call to get_word_timestamps.
    """
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    base = "20260720-091500-kidsong-director-caption-test"
    shots_dir = runstate.shots_dir_path(str(out_dir), base)

    _write_wav(out_dir / f"{base}.wav")
    song = _make_song()
    (out_dir / f"{base}-song.json").write_text(json.dumps(song), encoding="utf-8")

    shots = []
    for i in range(2):
        shot = {
            "id": f"s{i:02d}", "verse": 0, "start": i * 1.0, "end": (i + 1) * 1.0,
            "shot_type": "medium", "characters": ["all"], "action": "clapping",
            "setting": "a backyard", "camera": "static", "reuse_of": None,
            "seed": 1000 + i, "status": "planned",
        }
        runstate.mark_rendered(shot, _take(shots_dir, shot["id"], 0), score=0.9, attempts=1)
        shots.append(shot)
    runstate.save_shotlist(str(out_dir), base, {"shots": shots})

    calls = []

    import pipeline.captions as captions_mod
    import pipeline.kidsong.comfy as comfy_mod
    import pipeline.kidsong.cut_qc as cut_qc_mod
    import pipeline.kidsong.edit as edit_mod
    import pipeline.kidsong.review as review_mod
    import pipeline.kidsong.sing as sing_mod

    class FakeComfyClient:
        def __init__(self, cfg):
            self.url = "http://fake:8188"
            self._proc = None

        def load_workflow(self, workflow_name):
            return _REAL_LOAD_WORKFLOW(self, workflow_name)

        def ensure_up(self):
            raise AssertionError("no shot is pending -- ensure_up should never be called")

        def render(self, workflow, patches, out_path):
            raise AssertionError("no shot is pending -- render should never be called")

        def free(self):
            pass

    class FakeReviewer:
        def review(self, shot, video_path, out_dir=None):
            return {"accept": True, "score": 1.0, "reasons": [], "retry_hints": {}}

    def fake_get_word_timestamps(*args, **kwargs):
        calls.append((args, kwargs))
        return []

    def fake_assemble(cfg, cut_list, renders, voice, words, duration, out_path, **kw):
        with open(out_path, "wb") as f:
            f.write(b"\0" * 4096)

    def fake_review_cut(staging_path, cut_list, beats, words, duration, cfg, review_dir):
        return {"accept": True, "score": 1.0, "reasons": [], "retry_hints": {}}

    def fake_prepend_intro(video_path, cfg, on_progress=None):
        return {"applied": True, "intro_seconds": 4.0, "reason": None}

    def _sing_must_not_run(*a, **k):
        raise AssertionError("resume must not re-sing")

    monkeypatch.setattr(comfy_mod, "ComfyClient", FakeComfyClient)
    monkeypatch.setattr(review_mod, "get_reviewer", lambda cfg: FakeReviewer())
    monkeypatch.setattr(review_mod, "contact_sheet", lambda *a, **k: None)
    monkeypatch.setattr(review_mod, "review_shots_log", lambda path, entries: path)
    monkeypatch.setattr(captions_mod, "get_word_timestamps", fake_get_word_timestamps)
    monkeypatch.setattr(sing_mod, "verse_times_from_words", lambda *a, **k: [(0.0, 4.0)])
    monkeypatch.setattr(edit_mod, "beat_grid", lambda path: [0.0, 1.0, 2.0, 3.0])
    monkeypatch.setattr(edit_mod, "build_cut_list", lambda *a, **k: [])
    monkeypatch.setattr(edit_mod, "assemble", fake_assemble)
    monkeypatch.setattr(cut_qc_mod, "review_cut", fake_review_cut)
    monkeypatch.setattr(edit_mod, "prepend_intro", fake_prepend_intro)
    monkeypatch.setattr(sing_mod, "sing_song", _sing_must_not_run)

    cfg = {
        "_root": str(tmp_path),
        "paths": {"output_dir": str(out_dir)},
        "video": {"max_seconds": 60},
        "captions": {"uppercase": True},
        "kidsong": {
            "mode": "director",
            "singer": "ace",
            "seed": 20260717,
            "shot": {"fps": 24, "max_frames": 241, "hires_pass": False},
            "review": {"script": False, "cut": True, "max_retries_per_shot": 0},
            "intro": {"enabled": True},
        },
    }

    return {"cfg": cfg, "base": base, "out_dir": str(out_dir), "song": song, "calls": calls}


def test_generate_director_forwards_lyrics_and_duration(director_run):
    """Call site 2 (pipeline.kidsong.generate._generate_director, ~line 558-563)."""
    from pipeline.kidsong.generate import generate_kidsong

    run = director_run
    result = generate_kidsong(cfg=run["cfg"], resume_base=run["base"])

    assert result["status"] == "approved"
    assert len(run["calls"]) == 1
    args, kwargs = run["calls"][0]
    assert kwargs.get("lyrics") == run["song"]["verses"]
    assert kwargs.get("duration") is not None
