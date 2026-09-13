"""Tests for pipeline.kidsong.lipsync — the gated Wav2Lip post-process.

These prove the contract WITHOUT any GPU, network, ffmpeg, or the lipsync model:
the real subprocess worker is replaced by an injected `runner`, and the audio
slice is exercised on a stdlib-written PCM wav. Covered:

  * DISABLED (default) is byte-identical: apply() returns the SAME cut_list and
    renders objects, untouched.
  * The audio-slice window math (slice_wav extracts exactly [start, end)).
  * Shot-type GATING: only closeup/medium cuts are planned; wides/inserts skip,
    and each planned cut carries its own song-time window.
  * ENABLED happy path: qualifying cuts get a per-cut synced src + `lipsynced`
    flag; renders gains a key per synced cut; non-qualifying cuts are untouched;
    the runner is called with each cut's correct window.
  * Graceful PASS-THROUGH when the worker fails (returns False) or raises: the
    cut is left pointing at its original render and no exception escapes.
"""
import os
import wave

import pytest

from pipeline.kidsong import lipsync


# --------------------------------------------------------------- fixtures ---
def _write_pcm_wav(path, seconds=6.0, rate=8000, freq=220.0):
    """A mono PCM16 sine so slice_wav has real, checkable sample content."""
    import math
    import struct

    n = int(seconds * rate)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        frames = bytearray()
        for i in range(n):
            v = int(30000 * math.sin(2 * math.pi * freq * (i / rate)))
            frames += struct.pack("<h", v)
        wf.writeframes(bytes(frames))
    return str(path)


def _shotlist():
    # s00 wide (skip), s01 medium, s02 closeup, s03 insert (skip).
    return {"shots": [
        {"id": "s00", "shot_type": "wide"},
        {"id": "s01", "shot_type": "medium"},
        {"id": "s02", "shot_type": "closeup"},
        {"id": "s03", "shot_type": "insert"},
    ]}


def _cut_list():
    return [
        {"shot_id": "s00", "src": "s00", "start": 0.0, "end": 2.0},
        {"shot_id": "s01", "src": "s01", "start": 2.0, "end": 4.0},
        {"shot_id": "s02", "src": "s02", "start": 4.0, "end": 5.5},
        {"shot_id": "s03", "src": "s03", "start": 5.5, "end": 6.0},
    ]


def _renders(tmp_path):
    r = {}
    for sid in ("s00", "s01", "s02", "s03"):
        p = tmp_path / f"{sid}.mp4"
        p.write_bytes(b"\x00\x01\x02\x03")  # non-empty stand-in take
        r[sid] = str(p)
    return r


def _enabled_cfg(tmp_path):
    return {"kidsong": {"lipsync": {
        "enabled": True,
        "shot_types": ["closeup", "medium"],
    }}}


# --------------------------------------------------------- disabled = noop ---
def test_disabled_is_byte_identical(tmp_path):
    cfg = {"kidsong": {"lipsync": {"enabled": False}}}
    cut_list = _cut_list()
    renders = _renders(tmp_path)
    voice = _write_pcm_wav(tmp_path / "song.wav")

    def boom(*a, **k):  # must never be called when disabled
        raise AssertionError("runner called while lipsync disabled")

    out_cuts, out_renders = lipsync.apply(
        cfg, cut_list, renders, _shotlist(), voice,
        str(tmp_path / "build"), runner=boom,
    )
    # SAME objects handed straight back — the byte-identical guarantee.
    assert out_cuts is cut_list
    assert out_renders is renders
    assert not any("lipsynced" in c for c in out_cuts)
    assert not any(k.startswith("__lipsync__") for k in out_renders)
    # No build dir created when off.
    assert not os.path.exists(str(tmp_path / "build"))


def test_absent_lipsync_config_is_disabled(tmp_path):
    cfg = {"kidsong": {}}
    cut_list = _cut_list()
    renders = _renders(tmp_path)
    out_cuts, out_renders = lipsync.apply(
        cfg, cut_list, renders, _shotlist(), _write_pcm_wav(tmp_path / "s.wav"),
        str(tmp_path / "b"), runner=lambda *a, **k: True,
    )
    assert out_cuts is cut_list and out_renders is renders


# ----------------------------------------------------------- slice window ---
def test_slice_wav_extracts_exact_window(tmp_path):
    rate = 8000
    src = _write_pcm_wav(tmp_path / "full.wav", seconds=6.0, rate=rate)
    out = str(tmp_path / "slice.wav")
    lipsync.slice_wav(src, 2.0, 4.0, out)
    with wave.open(out, "rb") as wf:
        assert wf.getframerate() == rate
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        # 2.0 -> 4.0 seconds == 2.0s of frames.
        assert abs(wf.getnframes() - int(2.0 * rate)) <= 1


def test_slice_wav_clamps_past_end(tmp_path):
    rate = 8000
    src = _write_pcm_wav(tmp_path / "full.wav", seconds=3.0, rate=rate)
    out = str(tmp_path / "slice.wav")
    # Window that runs past the file end is clamped, never empty.
    lipsync.slice_wav(src, 2.5, 9.0, out)
    with wave.open(out, "rb") as wf:
        assert 0 < wf.getnframes() <= int(0.5 * rate) + 2


# --------------------------------------------------------------- gating ---
def test_plan_only_closeup_and_medium():
    ls = lipsync.resolve_config(_enabled_cfg(None))
    plans = lipsync.plan_lipsync(_cut_list(), _shotlist(), ls)
    ids = [(p["index"], p["shot_id"]) for p in plans]
    # s00 wide and s03 insert are skipped; s01 medium (idx1) + s02 closeup (idx2).
    assert ids == [(1, "s01"), (2, "s02")]
    # Windows are the cuts' own song-time boundaries.
    by_id = {p["shot_id"]: p for p in plans}
    assert (by_id["s01"]["start"], by_id["s01"]["end"]) == (2.0, 4.0)
    assert (by_id["s02"]["start"], by_id["s02"]["end"]) == (4.0, 5.5)


def test_plan_skips_zero_window():
    ls = lipsync.resolve_config(_enabled_cfg(None))
    cuts = [{"shot_id": "s01", "src": "s01", "shot_type": None, "start": 3.0, "end": 3.0}]
    sl = {"shots": [{"id": "s01", "shot_type": "closeup"}]}
    assert lipsync.plan_lipsync(cuts, sl, ls) == []


def test_gating_respects_configured_shot_types():
    cfg = {"kidsong": {"lipsync": {"enabled": True, "shot_types": ["closeup"]}}}
    ls = lipsync.resolve_config(cfg)
    plans = lipsync.plan_lipsync(_cut_list(), _shotlist(), ls)
    assert [p["shot_id"] for p in plans] == ["s02"]  # medium now excluded


# ------------------------------------------------------- enabled happy path ---
def test_enabled_syncs_only_eligible_cuts(tmp_path):
    cfg = _enabled_cfg(tmp_path)
    cut_list = _cut_list()
    renders = _renders(tmp_path)
    voice = _write_pcm_wav(tmp_path / "song.wav")
    calls = []

    def fake_runner(ls_cfg, face, audio, out_path, log=None):
        # The audio slice really was written for this cut's window.
        assert os.path.exists(audio) and os.path.getsize(audio) > 0
        calls.append((face, out_path))
        with open(out_path, "wb") as f:
            f.write(b"SYNCED-MP4-BYTES")
        return True

    out_cuts, out_renders = lipsync.apply(
        cfg, cut_list, renders, _shotlist(), voice,
        str(tmp_path / "build"), runner=fake_runner,
    )
    # Two eligible cuts synced (s01 medium, s02 closeup).
    assert len(calls) == 2
    assert out_cuts[1]["lipsynced"] is True
    assert out_cuts[2]["lipsynced"] is True
    assert out_cuts[1]["src"] == "__lipsync__01_s01"
    assert out_cuts[2]["src"] == "__lipsync__02_s02"
    # Wide + insert cuts untouched (still point at their original src).
    assert out_cuts[0]["src"] == "s00" and "lipsynced" not in out_cuts[0]
    assert out_cuts[3]["src"] == "s03" and "lipsynced" not in out_cuts[3]
    # renders gained one key per synced cut, pointing at real files.
    assert set(out_renders) - set(renders) == {"__lipsync__01_s01", "__lipsync__02_s02"}
    for k in ("__lipsync__01_s01", "__lipsync__02_s02"):
        assert os.path.getsize(out_renders[k]) > 0
    # The originals were NOT mutated (a fresh cut_list/renders was returned).
    assert cut_list[1]["src"] == "s01" and "lipsynced" not in cut_list[1]
    assert "__lipsync__01_s01" not in renders


def test_worker_failure_passes_through(tmp_path):
    cfg = _enabled_cfg(tmp_path)
    cut_list = _cut_list()
    renders = _renders(tmp_path)
    voice = _write_pcm_wav(tmp_path / "song.wav")

    def failing_runner(ls_cfg, face, audio, out_path, log=None):
        return False  # worker "ran" but produced nothing usable

    out_cuts, out_renders = lipsync.apply(
        cfg, cut_list, renders, _shotlist(), voice,
        str(tmp_path / "build"), runner=failing_runner,
    )
    # Nothing synced -> the ORIGINAL objects come straight back, unchanged.
    assert out_cuts is cut_list
    assert out_renders is renders
    assert not any("lipsynced" in c for c in out_cuts)


def test_worker_raise_passes_through(tmp_path):
    cfg = _enabled_cfg(tmp_path)
    cut_list = _cut_list()
    renders = _renders(tmp_path)
    voice = _write_pcm_wav(tmp_path / "song.wav")

    def raising_runner(ls_cfg, face, audio, out_path, log=None):
        raise RuntimeError("model blew up")

    # Must not propagate — a lip-sync failure never blocks the episode.
    out_cuts, out_renders = lipsync.apply(
        cfg, cut_list, renders, _shotlist(), voice,
        str(tmp_path / "build"), runner=raising_runner,
    )
    assert out_cuts is cut_list and out_renders is renders


def test_partial_success_keeps_failed_cut_original(tmp_path):
    cfg = _enabled_cfg(tmp_path)
    cut_list = _cut_list()
    renders = _renders(tmp_path)
    voice = _write_pcm_wav(tmp_path / "song.wav")

    def picky_runner(ls_cfg, face, audio, out_path, log=None):
        # Sync s01 only; fail s02.
        if face.endswith("s01.mp4"):
            with open(out_path, "wb") as f:
                f.write(b"OK")
            return True
        return False

    out_cuts, out_renders = lipsync.apply(
        cfg, cut_list, renders, _shotlist(), voice,
        str(tmp_path / "build"), runner=picky_runner,
    )
    assert out_cuts[1]["src"] == "__lipsync__01_s01"
    assert out_cuts[1]["lipsynced"] is True
    # s02 failed -> untouched, still its own render.
    assert out_cuts[2]["src"] == "s02" and "lipsynced" not in out_cuts[2]
    assert "__lipsync__02_s02" not in out_renders


def test_missing_source_render_passes_through(tmp_path):
    cfg = _enabled_cfg(tmp_path)
    cut_list = _cut_list()
    renders = _renders(tmp_path)
    del renders["s01"]  # no take on disk for a planned cut
    voice = _write_pcm_wav(tmp_path / "song.wav")

    def only_s02(ls_cfg, face, audio, out_path, log=None):
        with open(out_path, "wb") as f:
            f.write(b"OK")
        return True

    out_cuts, out_renders = lipsync.apply(
        cfg, cut_list, renders, _shotlist(), voice,
        str(tmp_path / "build"), runner=only_s02,
    )
    # s01 skipped (no source), s02 still synced.
    assert "lipsynced" not in out_cuts[1]
    assert out_cuts[2]["src"] == "__lipsync__02_s02"


def test_default_config_values():
    ls = lipsync.resolve_config({})
    assert ls["enabled"] is False
    assert ls["model"] == "96px"
    assert ls["enhancer"] == "none"
    assert ls["shot_types_set"] == {"closeup", "medium"}
    assert "wav2lip_gan.pth" in ls["checkpoint"]
