"""Tests for the sing-quality fixes in pipeline.kidsong.sing/generate.py.

Motivating diagnosis (2026-07-24): ACE-Step's 5Hz "thinking" LM
(ace_runner.py: thinking=True, use_cot_metas=True) replans bpm/arrangement on
EVERY render -- the seed only pins the DiT noise, so a scrambled render is a
dice roll, not a deterministic bug. `edit.beat_grid`'s phase-fit resultant is
an objective, cheap proxy for "did this come out coherent" (measured: 0.54 at
99.38bpm sang clean; 0.235 at 129bpm and 0.068 at 86bpm were scrambled). Three
fixes are covered here:

  1. sing.py's caption resolution now prefers the per-song `song["song_style"]`
     lyrics.py stamps over the global config default (`_resolve_caption`).
  2. generate.py gates a fresh sing on that resultant, re-singing with a
     bumped seed up to `kidsong.sing_coherence.max_attempts` times and keeping
     the best-scoring render (`_sing_with_coherence_gate`,
     `_measure_sing_coherence`, `_sing_coherence_settings`).
  3. sing.py mirrors the ACE-Step subprocess's bpm/key/seed stdout lines into
     the run log instead of discarding them (`_mirror_ace_stdout`).

No GPU, no ACE-Step subprocess, no librosa: `sing_song` and `edit.beat_grid`
are monkeypatched at their call sites.
"""
import os

import pytest

from pipeline.kidsong.runlog import RunLog


# ============================================================ caption fallback
from pipeline.kidsong.sing import _DEFAULTS, _resolve_caption


def test_song_caption_wins_over_global_default():
    song = {"title": "Sleepy Time", "song_style": "gentle sleepy lullaby, soft hums"}
    settings = {"song_style": "cheerful upbeat kids song"}
    assert _resolve_caption(song, settings) == "gentle sleepy lullaby, soft hums"


def test_global_song_style_is_the_fallback_when_the_song_has_none():
    song = {"title": "No Caption Here"}
    settings = {"song_style": "cheerful upbeat kids song"}
    assert _resolve_caption(song, settings) == "cheerful upbeat kids song"


def test_module_default_is_the_fallback_when_neither_is_set():
    assert _resolve_caption({}, {}) == _DEFAULTS["song_style"]


def test_an_empty_song_caption_string_does_not_win():
    """A blank/falsy song_style must not shadow the real global setting."""
    song = {"song_style": ""}
    settings = {"song_style": "cheerful upbeat kids song"}
    assert _resolve_caption(song, settings) == "cheerful upbeat kids song"


def test_resolve_caption_tolerates_a_none_song():
    assert _resolve_caption(None, {"song_style": "x"}) == "x"


# ======================================================== ace stdout mirroring
from pipeline.kidsong.sing import _mirror_ace_stdout


def test_lines_with_bpm_key_seed_are_mirrored_into_the_run_log(tmp_path):
    messages = []

    class FakeRunLog:
        def info(self, msg, *a):
            messages.append(msg % a if a else msg)

    stdout = (
        "============================================================\n"
        "GPU 12.00GB tier=consumer recommended_backend=pt -> backend=pt\n"
        "auto_offload(<20GB)=False lm_model=acestep-5Hz-lm-0.6B DiT=acestep-v15-turbo "
        "steps=8 shift=3.0 seed=20260717\n"
        "Loading DiT/VAE/text-encoder ...\n"
        "GEN_TIME=42.1s success=True\n"
        "METAS: {'bpm': 99.38, 'key': 'C major'}\n"
        "DONE_OK\n"
    )
    _mirror_ace_stdout(stdout, FakeRunLog())

    joined = "\n".join(messages)
    assert "seed=20260717" in joined
    assert "METAS: {'bpm': 99.38" in joined
    assert "tier=consumer" in joined
    # A line with none of bpm/key/seed/lm_model/gen_time/tier is not mirrored.
    assert not any("Loading DiT/VAE" in m for m in messages)


def test_mirror_is_a_no_op_without_a_runlog():
    # Must not raise when runlog is None (positional 3-arg legacy callers).
    _mirror_ace_stdout("seed=1 bpm=100", None)


def test_mirror_tolerates_a_runlog_with_no_info_method():
    class NoInfo:
        pass

    _mirror_ace_stdout("seed=1", NoInfo())  # must not raise


# ============================================ edit.py: resultant stays visible
# beat_grid's own degenerate branches used to report ONLY a human-readable
# note ("resultant 0.235 < 0.25 ...") with no numeric field, which is exactly
# the case a sing-coherence gate needs a NUMBER for (a scrambled render, by
# definition, fails the grid's own quality bar). This is the small extraction
# _sing_with_coherence_gate relies on instead of re-deriving the metric.
import pipeline.kidsong.edit as edit_mod


def test_degenerate_phase_mismatch_still_reports_a_numeric_resultant():
    """The exact measured-bad case from the diagnosis: beats scattered with no
    consistent phase must still expose a numeric resultant, not just prose."""
    rng = [((i * 37) % 101) / 101.0 * 60.0 for i in range(60)]
    detected = sorted(rng)
    _, _, info = edit_mod.complete_beat_grid(120.0, detected, 60.0)

    assert info["source"] == "detected_only"
    assert "resultant" in info, "the numeric metric must survive the degenerate path"
    assert info["resultant"] < edit_mod._GRID_MIN_RESULTANT
    assert "rayleigh" in info


def test_too_few_beats_has_no_resultant_to_report():
    """Genuinely un-computable cases (nothing to fit a phase to) must not
    fabricate a number — absence, not a fake 0.0."""
    _, _, info = edit_mod.complete_beat_grid(120.0, [1.0], 60.0)
    assert "resultant" not in info


# ================================================================ config keys
from pipeline.kidsong.generate import _sing_coherence_settings


def test_sing_coherence_settings_default_to_the_measured_thresholds():
    assert _sing_coherence_settings({}) == (0.25, 3)
    assert _sing_coherence_settings({"kidsong": {}}) == (0.25, 3)


def test_sing_coherence_settings_read_overrides():
    cfg = {"kidsong": {"sing_coherence": {"min_resultant": 0.4, "max_attempts": 5}}}
    assert _sing_coherence_settings(cfg) == (0.4, 5)


def test_sing_coherence_settings_clamps_max_attempts_to_at_least_one():
    cfg = {"kidsong": {"sing_coherence": {"max_attempts": 0}}}
    assert _sing_coherence_settings(cfg) == (0.25, 1)


# ============================================================ measure helper
from pipeline.kidsong.generate import _measure_sing_coherence


def test_measure_sing_coherence_reads_the_grid_resultant_with_cache_disabled(monkeypatch):
    seen = {}

    def fake_beat_grid(path, cache=True, duration=None):
        seen["path"], seen["cache"], seen["duration"] = path, cache, duration
        return {"bpm": 100.0, "beat_times": [], "grid": {"resultant": 0.42}}

    monkeypatch.setattr(edit_mod, "beat_grid", fake_beat_grid)

    assert _measure_sing_coherence("song.wav", 12.0, runlog=None) == 0.42
    # Every re-sing attempt reuses the SAME path, so caching by path alone
    # would read a stale sidecar describing a different rendition.
    assert seen == {"path": "song.wav", "cache": False, "duration": 12.0}


def test_measure_sing_coherence_returns_none_when_beat_grid_raises(monkeypatch):
    def boom(path, cache=True, duration=None):
        raise RuntimeError("librosa exploded")

    monkeypatch.setattr(edit_mod, "beat_grid", boom)
    assert _measure_sing_coherence("song.wav", 10.0, runlog=None) is None


def test_measure_sing_coherence_returns_none_when_grid_has_no_resultant(monkeypatch):
    """Too-few-beats / no-period cases (see the edit.py tests above)."""
    monkeypatch.setattr(
        edit_mod, "beat_grid",
        lambda path, cache=True, duration=None: {"grid": {"source": "detected_only"}},
    )
    assert _measure_sing_coherence("song.wav", 10.0, runlog=None) is None


# ======================================================= the gate + re-sing
import pipeline.kidsong.sing as sing_mod
from pipeline.kidsong.generate import _sing_with_coherence_gate


def _fake_sing_song(seeds_used, duration=12.0):
    """Writes a marker (the seed it was asked to use) to out_path so tests can
    tell which attempt's audio ended up where, without any real ACE-Step/
    ffmpeg/audio_clean work."""

    def fn(song, cfg, out_path, runlog=None):
        seed = cfg["kidsong"]["seed"]
        seeds_used.append(seed)
        with open(out_path, "wb") as f:
            f.write(f"seed={seed}".encode())
        return {"audio_path": out_path, "duration": duration, "clean": {}}

    return fn


def _fake_beat_grid(resultants):
    """Returns resultants[i] on the i-th call; records (cache, duration)."""
    calls = []

    def fn(path, cache=True, duration=None):
        calls.append({"path": path, "cache": cache, "duration": duration})
        return {"grid": {"resultant": resultants[len(calls) - 1]}}

    return fn, calls


def _marker(path):
    return open(path, "rb").read()


def test_a_good_first_attempt_never_retries(tmp_path, monkeypatch):
    voice_path = str(tmp_path / "song.wav")
    seeds_used = []
    monkeypatch.setattr(sing_mod, "sing_song", _fake_sing_song(seeds_used))
    fake_bg, bg_calls = _fake_beat_grid([0.54])
    monkeypatch.setattr(edit_mod, "beat_grid", fake_bg)

    cfg = {"kidsong": {"seed": 100, "sing_coherence": {"min_resultant": 0.25, "max_attempts": 3}}}
    runlog = RunLog(str(tmp_path / "logs"), base="rl", stream=False)
    try:
        vocals = _sing_with_coherence_gate({"title": "t"}, cfg, voice_path, runlog, step=lambda m: None)
    finally:
        runlog.close()

    assert seeds_used == [100]
    assert len(bg_calls) == 1
    assert vocals["duration"] == 12.0
    assert _marker(voice_path) == b"seed=100"
    assert not os.path.exists(voice_path + ".rejected-a1.wav")


def test_a_low_resultant_re_sings_with_seed_plus_one_then_stops(tmp_path, monkeypatch):
    """Attempt 1 is scrambled (0.10 < 0.25); attempt 2 clears the bar (0.30)
    -> the gate stops there, never spending a 3rd attempt, and attempt 2's
    audio is what ends up at voice_path."""
    voice_path = str(tmp_path / "song.wav")
    seeds_used = []
    monkeypatch.setattr(sing_mod, "sing_song", _fake_sing_song(seeds_used))
    fake_bg, bg_calls = _fake_beat_grid([0.10, 0.30, 0.99])
    monkeypatch.setattr(edit_mod, "beat_grid", fake_bg)

    cfg = {"kidsong": {"seed": 500, "sing_coherence": {"min_resultant": 0.25, "max_attempts": 3}}}
    runlog = RunLog(str(tmp_path / "logs"), base="rl", stream=False)
    try:
        vocals = _sing_with_coherence_gate({"title": "t"}, cfg, voice_path, runlog, step=lambda m: None)
    finally:
        runlog.close()

    assert seeds_used == [500, 501], "must stop once an attempt clears the bar"
    assert len(bg_calls) == 2
    assert all(c["cache"] is False for c in bg_calls)
    assert vocals["duration"] == 12.0

    # The winner (attempt 2, seed 501) is what ships at voice_path...
    assert _marker(voice_path) == b"seed=501"
    # ...and the rejected first attempt is kept, never deleted, under the
    # documented name.
    rejected = voice_path + ".rejected-a1.wav"
    assert os.path.exists(rejected)
    assert _marker(rejected) == b"seed=500"
    # The winner itself carries no "rejected" duplicate.
    assert not os.path.exists(voice_path + ".rejected-a2.wav")

    log_text = open(runlog.path, encoding="utf-8").read()
    assert "STAGE sing_coherence attempt=1" in log_text
    assert "STAGE sing_coherence attempt=2" in log_text
    assert "attempt=3" not in log_text


def test_max_attempts_is_respected_and_the_best_of_all_ships_with_a_warning(tmp_path, monkeypatch):
    """None of the three attempts clear the bar -- the highest-scoring one
    (attempt 2, seed+1) still ships, every other attempt stays parked, and the
    shortfall is logged loudly."""
    voice_path = str(tmp_path / "song.wav")
    seeds_used = []
    monkeypatch.setattr(sing_mod, "sing_song", _fake_sing_song(seeds_used))
    fake_bg, bg_calls = _fake_beat_grid([0.10, 0.22, 0.05])
    monkeypatch.setattr(edit_mod, "beat_grid", fake_bg)

    cfg = {"kidsong": {"seed": 700, "sing_coherence": {"min_resultant": 0.25, "max_attempts": 3}}}
    runlog = RunLog(str(tmp_path / "logs"), base="rl", stream=False)
    try:
        vocals = _sing_with_coherence_gate({"title": "t"}, cfg, voice_path, runlog, step=lambda m: None)
    finally:
        runlog.close()

    assert seeds_used == [700, 701, 702], "all three attempts must be spent"
    assert len(bg_calls) == 3
    assert vocals["duration"] == 12.0

    # Attempt 2 (seed 701, resultant 0.22) is the best of a bad lot.
    assert _marker(voice_path) == b"seed=701"
    assert _marker(voice_path + ".rejected-a1.wav") == b"seed=700"
    assert _marker(voice_path + ".rejected-a3.wav") == b"seed=702"
    assert not os.path.exists(voice_path + ".rejected-a2.wav")

    log_text = open(runlog.path, encoding="utf-8").read()
    assert "WARNING" in log_text
    assert "no attempt reached resultant" in log_text
    assert "seed 701" in log_text or "seed=701" in log_text


def test_max_attempts_one_config_never_retries_even_when_scrambled(tmp_path, monkeypatch):
    voice_path = str(tmp_path / "song.wav")
    seeds_used = []
    monkeypatch.setattr(sing_mod, "sing_song", _fake_sing_song(seeds_used))
    fake_bg, bg_calls = _fake_beat_grid([0.01])
    monkeypatch.setattr(edit_mod, "beat_grid", fake_bg)

    cfg = {"kidsong": {"seed": 900, "sing_coherence": {"min_resultant": 0.25, "max_attempts": 1}}}
    runlog = RunLog(str(tmp_path / "logs"), base="rl", stream=False)
    try:
        _sing_with_coherence_gate({"title": "t"}, cfg, voice_path, runlog, step=lambda m: None)
    finally:
        runlog.close()

    assert seeds_used == [900]
    assert len(bg_calls) == 1
    assert not os.path.exists(voice_path + ".rejected-a1.wav")


def test_an_unmeasurable_attempt_is_treated_as_worse_than_any_measured_one(tmp_path, monkeypatch):
    """beat_grid failing outright (None resultant) on attempt 1, a measurable
    but sub-threshold 0.20 on attempt 2 -- attempt 2 must win (a real number
    beats no number at all), and the run must still finish (max_attempts=2)."""
    voice_path = str(tmp_path / "song.wav")
    seeds_used = []
    monkeypatch.setattr(sing_mod, "sing_song", _fake_sing_song(seeds_used))

    calls = []

    def flaky_beat_grid(path, cache=True, duration=None):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("librosa exploded")
        return {"grid": {"resultant": 0.20}}

    monkeypatch.setattr(edit_mod, "beat_grid", flaky_beat_grid)

    cfg = {"kidsong": {"seed": 300, "sing_coherence": {"min_resultant": 0.25, "max_attempts": 2}}}
    runlog = RunLog(str(tmp_path / "logs"), base="rl", stream=False)
    try:
        _sing_with_coherence_gate({"title": "t"}, cfg, voice_path, runlog, step=lambda m: None)
    finally:
        runlog.close()

    assert seeds_used == [300, 301]
    assert _marker(voice_path) == b"seed=301"
    assert _marker(voice_path + ".rejected-a1.wav") == b"seed=300"


# ================================================== resume: never re-sing ===
# The gate only ever runs from the FRESH-sing branch of _generate_director;
# the resume/reuse_audio branch (generate.py, just above the sing stage) never
# reaches it at all -- re-singing on resume would desync the beat grid every
# already-rendered take was cut against. Regression-guarded end to end by the
# existing resume fixtures (test_kidsong_runlog_resume.py and siblings, which
# monkeypatch sing_mod.sing_song to explode on resume); this test pins the
# same invariant directly against the coherence gate itself.
def test_gate_is_never_invoked_when_resuming_with_existing_audio(tmp_path, monkeypatch):
    import json
    import struct
    import wave

    from pipeline.kidsong import runstate
    from pipeline.kidsong.generate import generate_kidsong

    out_dir = tmp_path / "output"
    out_dir.mkdir()
    base = "20260725-000000-kidsong-resume-no-resing"
    shots_dir = runstate.shots_dir_path(str(out_dir), base)

    with wave.open(str(out_dir / f"{base}.wav"), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(8000)
        wf.writeframes(struct.pack("<h", 0) * (8000 * 2))

    song = {
        "title": "Resume No Resing", "description": "d", "tags": ["t"],
        "characters": "Two toddlers: Zuri; Kofi",
        "verses": [{"scene": "a park", "lines": ["a", "b"]}],
    }
    (out_dir / f"{base}-song.json").write_text(json.dumps(song), encoding="utf-8")

    os.makedirs(shots_dir, exist_ok=True)
    take_path = os.path.join(shots_dir, "s00_a0.mp4")
    with open(take_path, "wb") as f:
        f.write(b"\0" * 1024)
    shot = {
        "id": "s00", "verse": 0, "start": 0.0, "end": 2.0,
        "shot_type": "medium", "characters": ["all"], "action": "playing",
        "setting": "a park", "camera": "static", "reuse_of": None,
        "seed": 1, "status": "planned",
    }
    runstate.mark_rendered(shot, take_path, score=0.9, attempts=1)
    runstate.save_shotlist(str(out_dir), base, {"shots": [shot]})

    cfg = {
        "_root": str(tmp_path),
        "paths": {"output_dir": str(out_dir)},
        "video": {"max_seconds": 60},
        "captions": {"uppercase": True},
        "kidsong": {
            "mode": "director", "singer": "ace", "seed": 20260717,
            "shot": {"fps": 24, "max_frames": 241, "hires_pass": False},
            "review": {"script": False, "cut": False, "max_retries_per_shot": 0},
            "sing_coherence": {"min_resultant": 0.25, "max_attempts": 3},
        },
    }

    import pipeline.captions as captions_mod
    import pipeline.kidsong.comfy as comfy_mod
    import pipeline.kidsong.generate as generate_mod
    import pipeline.kidsong.review as review_mod

    class FakeComfyClient:
        def __init__(self, cfg):
            self.url = "http://fake:8188"

        def ensure_up(self):
            pass

        def free(self):
            pass

    class FakeReviewer:
        def review(self, shot, video_path, out_dir=None):
            return {"accept": True, "score": 1.0, "reasons": [], "retry_hints": {}}

    def fake_assemble(cfg, cut_list, renders, voice, words, duration, out_path, **kw):
        with open(out_path, "wb") as f:
            f.write(b"\0" * 4096)

    def _gate_must_not_run(*a, **k):
        raise AssertionError("resume must not re-sing: it would desync every take")

    monkeypatch.setattr(comfy_mod, "ComfyClient", FakeComfyClient)
    monkeypatch.setattr(review_mod, "get_reviewer", lambda cfg: FakeReviewer())
    monkeypatch.setattr(review_mod, "contact_sheet", lambda *a, **k: None)
    monkeypatch.setattr(review_mod, "review_shots_log", lambda path, entries: path)
    monkeypatch.setattr(captions_mod, "get_word_timestamps", lambda *a, **k: [])
    monkeypatch.setattr(sing_mod, "verse_times_from_words", lambda *a, **k: [(0.0, 2.0)])
    monkeypatch.setattr(edit_mod, "beat_grid", lambda path: [0.0, 1.0, 2.0])
    monkeypatch.setattr(edit_mod, "build_cut_list", lambda *a, **k: [])
    monkeypatch.setattr(edit_mod, "assemble", fake_assemble)
    monkeypatch.setattr(generate_mod, "_sing_with_coherence_gate", _gate_must_not_run)
    monkeypatch.setattr(sing_mod, "sing_song", _gate_must_not_run)

    result = generate_kidsong(cfg=cfg, resume_base=base)

    assert result["status"] == "approved"
    assert os.path.exists(result["video_path"])
