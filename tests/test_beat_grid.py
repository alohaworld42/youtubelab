"""Tests for beat-grid completion in pipeline/kidsong/edit.py.

Background (measured on real files, no audio needed here): librosa's beat
tracker only emits beats where it can hold a confident pulse. On
output/brushing_resing/song_official_clean.wav — the same song as
song_official.wav minus a 70ms head click and a dead tail — the tracker
emits NOTHING before 13.72s, because the click it had been using as an
anchor through the song's quiet ~14s intro is gone. The tempo is unchanged
(129.2 bpm, period 0.4644s).

The consequence was that cut_qc._check_cuts had no beat to measure the
first three cut boundaries against and reported them as 10.08s / 6.06s /
2.55s off-beat, rejecting a correctly cut episode. The boundaries were fine;
the grid was incomplete.

`complete_beat_grid` fits a period and phase to the detected beats and fills
only the regions the tracker left empty. Everything below is synthetic beat
arrays — no real audio, no librosa.
"""
import json
import math

import pytest

from pipeline.kidsong import cut_qc, edit

BPM = 129.19921875
PERIOD = 60.0 / BPM  # ~0.4644s, the real brushing-song tempo
SONG = 58.5


def _grid(start, end, period=PERIOD, phase=0.0):
    """Beats at `phase + k*period` for every k landing in [start, end]."""
    k0 = math.ceil((start - phase) / period)
    out = []
    k = k0
    while phase + k * period <= end + 1e-9:
        out.append(phase + k * period)
        k += 1
    return out


# ------------------------------------------------------- the reported bug ---
def test_gap_at_start_is_filled_at_the_right_phase():
    """The real failure: no beats at all before 13.72s."""
    detected = _grid(13.72, 57.4, phase=0.406674)
    times, flags, info = edit.complete_beat_grid(BPM, detected, SONG)

    assert info["source"] == "extrapolated"
    assert info["extrapolated"] > 25  # ~14s of empty intro at 0.4644s
    assert times[0] < 0.6, "grid should now reach the song start"

    # Every filled beat sits on the same grid the detected beats define.
    for t, is_ext in zip(times, flags):
        if is_ext:
            off = (t - info["phase"]) / info["period"]
            # period/phase are rounded to 6dp on the way into the sidecar.
            assert abs(off - round(off)) < 1e-4

    assert info["period"] == pytest.approx(PERIOD, abs=1e-4)
    assert info["bpm_from_period"] == pytest.approx(BPM, rel=1e-3)


def test_detected_beats_are_preserved_exactly():
    """Filling gaps must never move a beat the tracker actually found."""
    detected = _grid(13.72, 57.4, phase=0.406674)
    times, flags, _ = edit.complete_beat_grid(BPM, detected, SONG)

    kept = [t for t, is_ext in zip(times, flags) if not is_ext]
    assert kept == pytest.approx(detected)
    assert len(times) == len(flags)
    assert times == sorted(times)


def test_extrapolated_beats_are_flagged():
    detected = _grid(13.72, 57.4, phase=0.406674)
    times, flags, info = edit.complete_beat_grid(BPM, detected, SONG)

    assert sum(flags) == info["extrapolated"]
    assert len(times) - sum(flags) == info["detected"] == len(detected)
    # The head is synthetic, the middle is real.
    assert flags[0] is True
    assert flags[len(flags) // 2] is False


def test_tail_gap_is_filled_too():
    detected = _grid(0.0, 40.0)
    times, flags, info = edit.complete_beat_grid(BPM, detected, SONG)

    assert times[-1] > 58.0
    assert info["extrapolated"] > 30
    assert all(flags[i] for i, t in enumerate(times) if t > 41.0)


# ------------------------------------------------------------- the no-ops ---
def test_grid_already_covering_the_song_is_unchanged():
    detected = _grid(0.0, SONG)
    times, flags, info = edit.complete_beat_grid(BPM, detected, SONG)

    assert times == pytest.approx(detected)
    assert not any(flags)
    assert info["extrapolated"] == 0
    assert info["source"] == "detected_only"
    assert "no-op" in info["note"]


def test_small_interior_gap_is_not_double_filled():
    """One skipped beat mid-song: the fill lands in the hole and nowhere
    else, so no beat gets a near-duplicate neighbour."""
    detected = _grid(0.0, SONG)
    dropped = detected.pop(40)
    times, flags, info = edit.complete_beat_grid(BPM, detected, SONG)

    assert info["extrapolated"] == 1
    filled = [t for t, e in zip(times, flags) if e]
    assert filled[0] == pytest.approx(dropped, abs=0.02)
    gaps = [b - a for a, b in zip(times, times[1:])]
    assert min(gaps) > 0.5 * PERIOD


# -------------------------------------------------------------- degenerate ---
@pytest.mark.parametrize("detected", [[], [10.0], [10.0, 10.4644]])
def test_too_few_beats_degenerates_gracefully(detected):
    """Under _GRID_MIN_BEATS there is no redundancy to estimate a period
    from — return exactly what the tracker gave and say why."""
    times, flags, info = edit.complete_beat_grid(BPM, detected, SONG)

    assert times == pytest.approx(detected)
    assert flags == [False] * len(detected)
    assert info["source"] == "detected_only"
    assert info["extrapolated"] == 0
    assert "detected beat" in info["note"]


def test_no_grid_is_fabricated_from_nothing():
    times, flags, info = edit.complete_beat_grid(BPM, [], SONG)
    assert times == []
    assert flags == []


def test_implausible_tempo_is_refused():
    """Beats 8s apart imply 7.5 bpm — not a musical pulse, so no grid."""
    detected = [0.0, 8.0, 16.0, 24.0, 32.0]
    times, _, info = edit.complete_beat_grid(BPM, detected, SONG)

    assert times == pytest.approx(detected)
    assert info["source"] == "detected_only"
    assert "bpm" in info["note"]


def test_phase_scattered_beats_are_refused():
    """Randomly placed 'beats' share no phase: the Rayleigh test must reject
    them rather than extrapolating a grid off noise."""
    rng = [((i * 37) % 101) / 101.0 * SONG for i in range(60)]
    detected = sorted(rng)
    times, flags, info = edit.complete_beat_grid(BPM, detected, SONG)

    assert info["source"] == "detected_only"
    assert not any(flags)
    assert "consistent phase" in info["note"]


def test_half_time_detection_is_reported_not_silently_corrected():
    """Tracker says 129.2 bpm but the beats it emitted are 64.6 bpm apart."""
    detected = _grid(20.0, 55.0, period=PERIOD * 2)
    times, flags, info = edit.complete_beat_grid(BPM, detected, SONG)

    assert info["octave_mismatch"] == "half-time"
    assert info["bpm_from_period"] == pytest.approx(BPM / 2, rel=1e-3)
    assert sum(flags) > 0  # still extended, at the beats' own spacing
    assert "half-time" in info["note"]


def test_tempo_unrelated_to_reported_bpm_is_refused():
    """Neither equal to, nor an octave of, the tracker's own estimate."""
    detected = _grid(0.0, 40.0, period=PERIOD * 1.37)
    times, _, info = edit.complete_beat_grid(BPM, detected, SONG)

    assert info["source"] == "detected_only"
    assert "half/double-time" in info["note"]


# ----------------------------------------------------------- period fitting ---
def test_period_estimate_is_robust_to_one_jittered_beat():
    detected = _grid(10.0, 50.0)
    detected[15] += 0.18  # one badly-placed beat
    detected.sort()

    period = edit._estimate_period(detected)
    assert period == pytest.approx(PERIOD, abs=0.01)


def test_period_estimate_is_robust_to_skipped_beats():
    """A skipped beat shows up as a 2x interval; the median must ignore it."""
    detected = _grid(10.0, 50.0)
    del detected[10]
    del detected[20]
    del detected[30]

    period = edit._estimate_period(detected)
    assert period == pytest.approx(PERIOD, abs=1e-6)


def test_phase_fit_uses_all_beats_not_the_first():
    """Anchoring on beat 0 inherits beat 0's error; the circular mean over
    all beats does not."""
    detected = _grid(10.0, 50.0, phase=0.25)
    detected[0] -= 0.2  # a bad first beat

    phase, resultant = edit._fit_phase(detected, PERIOD)
    assert phase == pytest.approx(0.25, abs=0.02)
    assert resultant > 0.9


def test_resultant_is_high_for_a_rigid_grid_and_low_for_noise():
    rigid = _grid(0.0, SONG)
    _, r_rigid = edit._fit_phase(rigid, PERIOD)
    assert r_rigid > 0.99

    scattered = sorted(((i * 37) % 101) / 101.0 * SONG for i in range(60))
    _, r_noise = edit._fit_phase(scattered, PERIOD)
    assert r_noise < edit._GRID_MIN_RESULTANT


# -------------------------------------------------------- sidecar / caching ---
def _write_sidecar(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def test_legacy_sidecar_still_loads_and_is_upgraded(tmp_path, monkeypatch):
    """Existing `.beats.json` files carry only bpm/beat_times. They must keep
    working, and get completed without another librosa pass."""
    wav = tmp_path / "song.wav"
    wav.write_bytes(b"")  # never decoded: the sidecar short-circuits it
    detected = _grid(13.72, 57.4, phase=0.406674)
    _write_sidecar(str(wav) + ".beats.json", {"bpm": BPM, "beat_times": detected})

    monkeypatch.setattr(edit, "_audio_duration", lambda p: SONG)
    result = edit.beat_grid(str(wav))

    assert result["bpm"] == BPM
    assert len(result["beat_times"]) > len(detected)
    assert result["detected_beat_times"] == pytest.approx(detected)
    assert result["grid"]["source"] == "extrapolated"

    # Rewritten in place, and a second read is a straight cache hit.
    on_disk = json.loads((tmp_path / "song.wav.beats.json").read_text())
    assert "grid" in on_disk
    assert edit.beat_grid(str(wav))["beat_times"] == pytest.approx(result["beat_times"])


def test_sidecar_shape_stays_backward_compatible(tmp_path, monkeypatch):
    """Old readers only touch bpm/beat_times (cut_qc._check_cuts,
    build_cut_list, director, status) — those keys must keep their meaning."""
    wav = tmp_path / "song.wav"
    wav.write_bytes(b"")
    _write_sidecar(
        str(wav) + ".beats.json", {"bpm": BPM, "beat_times": _grid(13.72, 57.4)}
    )
    monkeypatch.setattr(edit, "_audio_duration", lambda p: SONG)

    result = edit.beat_grid(str(wav))
    assert isinstance(result["bpm"], float)
    assert all(isinstance(b, float) for b in result["beat_times"])
    assert result["beat_times"] == sorted(result["beat_times"])


def test_unknown_duration_still_fills_the_head(tmp_path, monkeypatch):
    """If we cannot probe the song length we still fix the head gap; we just
    do not invent beats past the last one the tracker found."""
    wav = tmp_path / "song.wav"
    wav.write_bytes(b"")
    detected = _grid(13.72, 57.4, phase=0.406674)
    _write_sidecar(str(wav) + ".beats.json", {"bpm": BPM, "beat_times": detected})
    monkeypatch.setattr(edit, "_audio_duration", lambda p: None)

    result = edit.beat_grid(str(wav))
    assert result["beat_times"][0] < 0.6
    assert result["beat_times"][-1] <= detected[-1] + 1e-9


# --------------------------------------------------- build_cut_list benefit ---
def test_boundaries_in_the_quiet_intro_now_snap():
    """Item 3 of the fix: with beats present in the intro, build_cut_list
    snaps those boundaries instead of leaving them unsnapped."""
    detected = _grid(13.72, 57.4, phase=0.406674)
    shotlist = {
        "shots": [
            {"id": f"s{i}", "start": round(i * 3.7, 3), "verse": 0} for i in range(15)
        ]
    }

    raw_cuts = edit.build_cut_list(shotlist, {"beat_times": detected}, SONG)
    full = edit.complete_beat_grid(BPM, detected, SONG)[0]
    full_cuts = edit.build_cut_list(shotlist, {"beat_times": full}, SONG)

    def worst(cuts, beats):
        return max(min(abs(float(c["end"]) - b) for b in beats) for c in cuts[:-1])

    # Measured against the COMPLETE grid, which is what cut_qc now sees.
    assert worst(raw_cuts, detected) > 2.0  # the old, broken measurement
    # < 0.05 rather than ~0: _split_long_segments carves an over-long segment
    # into EVEN pieces after snapping, so those sub-split points are not
    # themselves beat-aligned. Pre-existing behaviour, unrelated to the grid.
    assert worst(full_cuts, full) < 0.05
    # Every boundary that came from a shot edge (i.e. everything the snapper
    # actually touched) lands exactly on a beat.
    snapped = [c for c in full_cuts[:-1] if (c["end"] - c["start"]) > 3.5]
    assert len(snapped) >= 13
    assert max(min(abs(float(c["end"]) - b) for b in full) for c in snapped) < 0.001


# ------------------------------------------------------------ cut_qc gate ---
def test_cut_qc_accepts_boundaries_in_a_previously_unbeat_tracked_region():
    """The end-to-end point of the change, at unit scale: identical cut list,
    rejected against the tracker's raw beats, accepted against the completed
    grid."""
    detected = _grid(13.72, 57.4, phase=0.406674)
    full, _, _ = edit.complete_beat_grid(BPM, detected, SONG)

    cuts = [
        {"shot_id": "a", "start": 0.0, "end": 3.657, "src": "a"},
        {"shot_id": "b", "start": 3.657, "end": 7.837, "src": "b"},
        {"shot_id": "c", "start": 7.837, "end": 11.088, "src": "c"},
        {"shot_id": "d", "start": 11.088, "end": 15.139, "src": "d"},
    ]

    before = cut_qc._check_cuts(cuts, {"beat_times": detected}, SONG)
    assert [r for r in before if r.startswith("cut boundary off-beat")]

    after = cut_qc._check_cuts(cuts, {"beat_times": full}, SONG)
    assert not [r for r in after if r.startswith("cut boundary off-beat")]


def test_a_complete_grid_makes_the_0_5s_off_beat_gate_unfireable():
    """DOCUMENTS A CONSEQUENCE, not a desired behaviour.

    Against a grid that spans the whole song at a fixed period, the greatest
    distance any timestamp can be from the nearest beat is period/2 — 0.232s
    at this song's 129.2 bpm. cut_qc._BEAT_TOLERANCE is 0.5s, so once the
    grid is complete the off-beat check can no longer fire at any tempo
    above 60 bpm, however badly placed the boundary is.

    That is not a reason to leave the grid incomplete (the old behaviour
    rejected CORRECT cuts, which is far worse), but the tolerance wants to be
    a fraction of the beat period rather than a fixed 0.5s. Changing it is
    out of scope here; this test exists so the next person sees it.
    """
    full, _, info = edit.complete_beat_grid(
        BPM, _grid(13.72, 57.4, phase=0.406674), SONG
    )
    period = info["period"]

    # The worst-possible boundary: exactly half a period off the grid.
    worst_boundary = full[40] + period / 2.0
    dist = min(abs(worst_boundary - b) for b in full)
    assert dist == pytest.approx(period / 2.0, abs=1e-6)
    assert dist < cut_qc._BEAT_TOLERANCE, "0.5s cannot fire against a complete grid"

    cuts = [
        {"shot_id": "a", "start": 0.0, "end": worst_boundary, "src": "a"},
        {"shot_id": "b", "start": worst_boundary, "end": SONG, "src": "b"},
    ]
    reasons = cut_qc._check_cuts(cuts, {"beat_times": full}, SONG)
    assert not [r for r in reasons if r.startswith("cut boundary off-beat")]

    # A period-relative tolerance WOULD catch it — recorded here as the shape
    # the fix should take, without changing the shipped threshold.
    assert dist > 0.25 * period


def test_cut_qc_still_rejects_off_beat_boundaries_on_a_degenerate_grid():
    """The gate is extended, not loosened. When completion legitimately
    refuses (too few beats), the sparse grid is still what gets measured and
    a badly placed boundary still fails."""
    detected = [10.0, 20.0]  # under _GRID_MIN_BEATS -> returned unchanged
    times, _, info = edit.complete_beat_grid(BPM, detected, SONG)
    assert info["source"] == "detected_only"

    cuts = [
        {"shot_id": "a", "start": 0.0, "end": 14.0, "src": "a"},
        {"shot_id": "b", "start": 14.0, "end": 20.0, "src": "b"},
    ]
    reasons = cut_qc._check_cuts(cuts, {"beat_times": times}, SONG)
    assert [r for r in reasons if r.startswith("cut boundary off-beat")]


# ------------------------------------------- the cut window follows the config --
def _cuts(durations, start=0.0):
    out, t = [], start
    for i, d in enumerate(durations):
        out.append({"shot_id": f"s{i:02d}", "start": t, "end": t + d})
        t += d
    return out


def test_the_cut_floor_follows_the_config_like_the_ceiling_does():
    """The ceiling has been config-driven since scene-mode Phase 1, with a
    docstring saying the director's bound and the gate's bound must move
    together. The floor was left hardcoded at 1.4, so lowering the documented
    `kidsong.director.min_shot_seconds` knob made the director plan shots this
    gate then rejected — one reason per cut — and the episode failed its own
    gate and never reached Final/."""
    cuts = _cuts([1.0, 1.0, 1.0, 1.0])
    cfg = {"kidsong": {"review": {"min_cut_seconds": 0.9}}}

    assert cut_qc._check_cuts(cuts, [], 20.0, cfg) == []


def test_an_unset_floor_is_byte_identical_to_the_old_constant():
    cuts = _cuts([1.0, 1.0])
    for cfg in (None, {}, {"kidsong": {"review": {}}}):
        reasons = cut_qc._check_cuts(cuts, [], 20.0, cfg)
        assert reasons and all("cut duration out of range" in r for r in reasons)
        assert str(cut_qc._MIN_CUT) in reasons[0]


def test_the_floor_still_rejects_what_it_is_set_to_reject():
    cuts = _cuts([0.5, 0.5])
    cfg = {"kidsong": {"review": {"min_cut_seconds": 0.9}}}
    reasons = cut_qc._check_cuts(cuts, [], 20.0, cfg)
    assert len(reasons) == 2


def test_the_director_and_the_gate_agree_on_the_window():
    """The invariant the two knobs exist to preserve: a shot list the director
    plans within its bounds must pass the cut gate configured to match."""
    from pipeline.kidsong import director

    cfg = {
        "kidsong": {
            "director": {"min_shot_seconds": 1.0, "max_shot_seconds": 3.0},
            "shot": {"max_frames": 241, "fps": 24},
            "review": {"min_cut_seconds": 1.0, "max_cut_seconds": 3.0},
        }
    }
    min_s, max_s = director._shot_bounds(cfg)
    assert cut_qc._effective_min_cut(cfg) <= min_s
    assert cut_qc._effective_max_cut(cfg) >= max_s

    cuts = _cuts([min_s, max_s, (min_s + max_s) / 2])
    assert cut_qc._check_cuts(cuts, [], 20.0, cfg) == []


# ------------------------------------- the cut BUILDER follows the same window --
def test_raising_the_cut_window_actually_produces_longer_cuts():
    """The documented "fewer, longer shots" workflow is: raise
    kidsong.director.max_shot_seconds AND kidsong.review.max_cut_seconds. It
    could not work. `build_cut_list` — the stage that actually decides cut
    lengths — used fixed constants, so whatever the director planned it split
    anything over 4.5s back down. Measured before the fix: four 8.0s shot slots
    came out as eight 4.0s cuts. The knob moved the plan and never the picture."""
    from pipeline.kidsong import edit

    shots = [{"id": f"s{i:02d}", "start": i * 8.0, "end": (i + 1) * 8.0, "verse": 0}
             for i in range(4)]
    cfg = {"kidsong": {"review": {"max_cut_seconds": 9.0}}}

    cuts = edit.build_cut_list({"shots": shots}, [], 32.0, cfg)

    assert len(cuts) == 4, [round(c["end"] - c["start"], 2) for c in cuts]
    assert max(c["end"] - c["start"] for c in cuts) == pytest.approx(8.0)


def test_an_absent_cfg_builds_exactly_the_same_cut_list_as_before():
    from pipeline.kidsong import edit

    shots = [{"id": f"s{i:02d}", "start": i * 8.0, "end": (i + 1) * 8.0, "verse": 0}
             for i in range(4)]
    assert (edit.build_cut_list({"shots": shots}, [], 32.0)
            == edit.build_cut_list({"shots": shots}, [], 32.0, None)
            == edit.build_cut_list({"shots": shots}, [], 32.0, {}))
    durations = [c["end"] - c["start"] for c in edit.build_cut_list({"shots": shots}, [], 32.0)]
    assert max(durations) <= edit.MAX_CUT + 1e-6


def test_the_builder_stays_inside_the_gate_it_will_be_judged_by():
    """The end-to-end invariant across all three windows: what the builder emits
    must pass the gate, for the default config and for a moved one."""
    from pipeline.kidsong import edit

    shots = [{"id": f"s{i:02d}", "start": i * 8.0, "end": (i + 1) * 8.0, "verse": 0}
             for i in range(4)]
    for cfg in (None, {"kidsong": {"review": {"max_cut_seconds": 9.0,
                                              "min_cut_seconds": 1.0}}}):
        cuts = edit.build_cut_list({"shots": shots}, [], 32.0, cfg)
        assert cut_qc._check_cuts(cuts, [], 32.0, cfg) == [], cfg


def test_a_raised_window_cannot_stretch_a_cut_past_its_shot_slot():
    """Guard against reintroducing the forward-then-backward playback that
    commit 4d30322 fixed. `_extend_to_window` ping-pongs a take whose cut window
    is more than _MAX_SLOW_STRETCH longer than the take, and takes are capped by
    kidsong.shot.max_frames. Raising only the REVIEW window must therefore never
    lengthen a cut: the builder only merges and splits what the director's slots
    already gave it, so the cut length stays bounded by the plan."""
    from pipeline.kidsong import edit

    # 3.0s slots — what the director plans under the shipped max_frames=81.
    shots = [{"id": f"s{i:02d}", "start": i * 3.0, "end": (i + 1) * 3.0, "verse": 0}
             for i in range(6)]
    wide = {"kidsong": {"review": {"max_cut_seconds": 20.0}}}

    cuts = edit.build_cut_list({"shots": shots}, [], 18.0, wide)

    longest = max(c["end"] - c["start"] for c in cuts)
    assert longest <= 3.0 + 1e-6, (
        f"a cut grew to {longest:.2f}s from 3.0s slots — takes cannot cover that "
        "and _extend_to_window would ping-pong them"
    )
