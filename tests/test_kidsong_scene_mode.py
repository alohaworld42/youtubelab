"""Tests for scene-mode Phase 1: config-driven shot-duration bounds, the
opt-in camera-variety post-pass, and the matching cut_qc ceiling.

GOAL of the overhaul this is Phase 1 of: fewer, LONGER, more VARIED shots so
episodes stop feeling monotonous. Phase 1 only makes the existing hard
constants (director._MIN_SHOT_SECONDS/_MAX_SHOT_SECONDS, cut_qc._MAX_CUT)
config-driven and adds a deterministic camera-variety pass -- every new
behaviour is gated behind a new config key that defaults to today's fixed
values, so a cfg that never sets these keys must produce byte-identical
output to before this feature existed.

CPU-only, no GPU. `plan_shots` normally tries several LLM backends before
falling back to the deterministic planner; `_plan_shots_offline` below
(same pattern as tests/test_kidsong_shot_prompt.py's helper of the same
name) forces every one of those calls to fail so tests never touch a real
network host and always exercise the deterministic fallback path.
"""
import copy

import pytest

from pipeline.config import load_config
from pipeline.kidsong import cut_qc, director
from pipeline.kidsong.lyrics import FALLBACK_SONG

VERSE_TIMES = [(0, 15), (15, 30), (30, 45), (45, 60)]
BEATS = {"bpm": 96, "beat_times": [i * 0.625 for i in range(97)]}


def _plan_shots_offline(monkeypatch, song, verse_times, beats, cfg):
    """Call director.plan_shots with every network-touching LLM call forced
    to fail, so it always falls straight through to the deterministic
    fallback planner -- never hits localhost:11434 or any other host."""
    def _boom(*a, **k):
        raise RuntimeError("network disabled in tests")

    monkeypatch.setattr(director, "_call_gemini", _boom)
    monkeypatch.setattr(director, "_call_groq", _boom)
    monkeypatch.setattr(director, "_call_ollama", _boom)
    monkeypatch.setattr(director, "_post_ollama_chat", _boom)
    monkeypatch.setattr(director, "_post_openai_compatible_chat", _boom)
    return director.plan_shots(song, verse_times, beats, cfg)


def _song():
    # A cheap deep copy so no test mutates the module-level fallback song.
    import json

    return json.loads(json.dumps(FALLBACK_SONG))


# ---------------------------------------------------------- fewer/longer ---
def test_raised_max_shot_seconds_produces_fewer_longer_shots(monkeypatch):
    """Raising the ceiling only reaches the planner when the RENDER cap is
    raised with it: `_shot_bounds` clamps max_shot_seconds to the longest take
    kidsong.shot.max_frames can actually produce, and the shipped config caps
    that at 81 frames (3.13s at 24fps), well under the 4.5s default ceiling.
    So a scene-mode config has to lift both knobs -- lifting only the director
    one is a no-op, which is exactly what
    test_raising_only_the_director_ceiling_is_a_no_op below pins down."""
    cfg_default = load_config()
    cfg_scene = copy.deepcopy(cfg_default)
    cfg_scene.setdefault("kidsong", {}).setdefault("director", {}).update({
        "max_shot_seconds": 10.0,
        "target_shot_seconds": 8.0,
    })
    # 241f @ 24fps = 10.0s of renderable take, so the 10.0s ceiling binds.
    cfg_scene.setdefault("kidsong", {}).setdefault("shot", {})["max_frames"] = 241

    default_result = _plan_shots_offline(monkeypatch, _song(), VERSE_TIMES, BEATS, cfg_default)
    scene_result = _plan_shots_offline(monkeypatch, _song(), VERSE_TIMES, BEATS, cfg_scene)

    default_shots = default_result["shots"]
    scene_shots = scene_result["shots"]

    assert len(scene_shots) < len(default_shots), (
        f"expected fewer shots with a raised ceiling: "
        f"{len(scene_shots)} vs {len(default_shots)}"
    )

    max_default_dur = max(s["end"] - s["start"] for s in default_shots)
    max_scene_dur = max(s["end"] - s["start"] for s in scene_shots)
    assert max_default_dur <= 4.5 + 1e-6
    assert max_scene_dur > 4.5, f"expected a shot longer than 4.5s, got max {max_scene_dur:.2f}s"


def test_raising_only_the_director_ceiling_is_a_no_op_and_is_logged(monkeypatch, caplog):
    """The trap this test exists for: under the shipped config the render cap
    (shot.max_frames) is the binding constraint, so raising
    kidsong.director.max_shot_seconds alone changes nothing. That silence cost
    a debugging session once -- `_shot_bounds` now says so in the log."""
    cfg_default = load_config()
    cfg_raised = copy.deepcopy(cfg_default)
    cfg_raised.setdefault("kidsong", {}).setdefault("director", {})["max_shot_seconds"] = 10.0

    default_result = _plan_shots_offline(monkeypatch, _song(), VERSE_TIMES, BEATS, cfg_default)
    raised_result = _plan_shots_offline(monkeypatch, _song(), VERSE_TIMES, BEATS, cfg_raised)
    assert default_result == raised_result

    director._CLAMP_WARNED.clear()
    with caplog.at_level("WARNING", logger="kidsong.director"):
        director._shot_bounds(cfg_raised)
    assert any("max_shot_seconds" in r.message and "max_frames" in r.message
               for r in caplog.records), "the clamp must be visible in the log"


# ------------------------------------------------------- byte-identical ---
def test_absent_bounds_keys_match_the_old_hardcoded_defaults(monkeypatch):
    """A cfg that never sets min_shot_seconds/max_shot_seconds must plan
    exactly like a cfg that explicitly pins them to the old hard-coded
    constants (1.4/4.5) -- proving `_shot_bounds`'s default resolution is a
    no-op when the new keys are absent."""
    cfg_absent = load_config()
    cfg_absent.setdefault("kidsong", {}).setdefault("director", {}).pop("min_shot_seconds", None)
    cfg_absent["kidsong"]["director"].pop("max_shot_seconds", None)

    cfg_explicit_defaults = copy.deepcopy(cfg_absent)
    cfg_explicit_defaults["kidsong"]["director"].update({
        "min_shot_seconds": director._MIN_SHOT_SECONDS,
        "max_shot_seconds": director._MAX_SHOT_SECONDS,
    })

    song = _song()
    result_absent = _plan_shots_offline(monkeypatch, song, VERSE_TIMES, BEATS, cfg_absent)
    result_explicit = _plan_shots_offline(monkeypatch, song, VERSE_TIMES, BEATS, cfg_explicit_defaults)

    assert result_absent == result_explicit


def test_num_shots_for_and_tile_are_unchanged_without_new_kwargs():
    """Direct proof that `_num_shots_for`/`_tile` are byte-identical for any
    caller that omits the new min_s/max_s kwargs -- the exact contract for
    "existing caller/test that calls them without the new args"."""
    assert director._num_shots_for(10.0, 2.8) == director._num_shots_for(
        10.0, 2.8, director._MIN_SHOT_SECONDS, director._MAX_SHOT_SECONDS
    )
    assert director._tile(0.0, 10.0, 4, []) == director._tile(
        0.0, 10.0, 4, [], director._MIN_SHOT_SECONDS, director._MAX_SHOT_SECONDS
    )


# --------------------------------------------------------- camera variety ---
def _shot(id_, camera, reuse_of=None, verse=0):
    return {
        "id": id_,
        "verse": verse,
        "lyric_span": [0, 1],
        "start": 0.0,
        "end": 3.0,
        "shot_type": "medium",
        "characters": ["all"],
        "action": "The kids sing along with happy faces",
        "camera": camera,
        "setting": "a sunny backyard",
        "reuse_of": reuse_of,
        "seed": 1,
        "story_subject": None,
        "status": "planned",
    }


def test_enforce_camera_variety_no_adjacent_repeats_among_non_reuse_shots():
    shots = [
        _shot("s00", "static"),
        _shot("s01", "static"),
        _shot("s02", "static"),
        _shot("s03", "static", reuse_of="s00"),
        _shot("s04", "static"),
        _shot("s05", "static"),
    ]
    out = director._enforce_camera_variety(shots)

    # First shot (the wide establishing opener) stays static.
    assert out[0]["camera"] == "static"

    non_reuse = [s for s in out if not s.get("reuse_of")]
    for a, b in zip(non_reuse, non_reuse[1:]):
        # Only meaningful for shots that are literally adjacent in the list.
        idx_a, idx_b = out.index(a), out.index(b)
        if idx_b - idx_a == 1:
            assert a["camera"] != b["camera"], (a["id"], b["id"])

    # A reused shot inherits its source shot's camera rather than diverging.
    reused = next(s for s in out if s["id"] == "s03")
    source = next(s for s in out if s["id"] == "s00")
    assert reused["camera"] == source["camera"]

    # Only `camera` was touched -- every other field is untouched.
    by_id_in = {s["id"]: s for s in shots}
    for s in out:
        original = by_id_in[s["id"]]
        for key in original:
            if key == "camera":
                continue
            assert s[key] == original[key], (s["id"], key)


def test_enforce_camera_variety_keeps_first_shot_static_even_if_reprocessed():
    shots = [_shot("s00", "static"), _shot("s01", "static")]
    out = director._enforce_camera_variety(shots)
    assert out[0]["camera"] == "static"
    assert out[1]["camera"] != "static"


def test_enforce_camera_variety_empty_list_is_a_no_op():
    assert director._enforce_camera_variety([]) == []


def test_plan_shots_camera_variety_flag_gates_the_post_pass(monkeypatch):
    """Wires a fixed, deliberately-repetitive shot list straight into
    `plan_shots` (bypassing the LLM chain and the real fallback planner
    entirely) so the flag's effect is isolated from any planner behaviour."""
    canned = [
        _shot("s00", "static"),
        _shot("s01", "static"),
        _shot("s02", "static"),
    ]

    def _fake_normalize(raw, song, verse_times, beats, cfg):
        return [dict(s) for s in canned]

    monkeypatch.setattr(director, "_normalize_shots", _fake_normalize)

    cfg_default = load_config()
    cfg_default.setdefault("kidsong", {}).setdefault("director", {}).pop(
        "enforce_camera_variety", None
    )
    result_default = _plan_shots_offline(monkeypatch, _song(), VERSE_TIMES, BEATS, cfg_default)
    assert result_default == {"shots": canned}

    cfg_variety = copy.deepcopy(cfg_default)
    cfg_variety["kidsong"]["director"]["enforce_camera_variety"] = True
    result_variety = _plan_shots_offline(monkeypatch, _song(), VERSE_TIMES, BEATS, cfg_variety)
    variety_shots = result_variety["shots"]

    assert variety_shots != canned
    for a, b in zip(variety_shots, variety_shots[1:]):
        assert a["camera"] != b["camera"]
    assert variety_shots[0]["camera"] == "static"


# --------------------------------------------------------------- cut_qc ---
def test_cut_qc_max_cut_seconds_is_config_driven():
    cuts = [{"shot_id": "s00", "start": 0.0, "end": 9.0}]
    beats = {"beat_times": []}
    duration = 9.0

    cfg_default = {"kidsong": {}}
    reasons_default = cut_qc._check_cuts(cuts, beats, duration, cfg_default)
    assert any("cut duration out of range" in r for r in reasons_default), reasons_default

    cfg_raised = {"kidsong": {"review": {"max_cut_seconds": 10.0}}}
    reasons_raised = cut_qc._check_cuts(cuts, beats, duration, cfg_raised)
    assert not any("cut duration out of range" in r for r in reasons_raised), reasons_raised


def test_cut_qc_absent_cfg_matches_the_old_hardcoded_five_second_ceiling():
    """Existing callers (tests/test_beat_grid.py) call `_check_cuts` with no
    4th argument at all -- must stay byte-identical to the old fixed 5.0s."""
    cuts = [{"shot_id": "s00", "start": 0.0, "end": 4.9}]
    beats = {"beat_times": []}
    duration = 4.9

    assert cut_qc._check_cuts(cuts, beats, duration) == []

    cuts_over = [{"shot_id": "s00", "start": 0.0, "end": 5.1}]
    reasons = cut_qc._check_cuts(cuts_over, beats, 5.1)
    assert any("cut duration out of range" in r for r in reasons), reasons

    reasons_with_none_cfg = cut_qc._check_cuts(cuts_over, beats, 5.1, None)
    assert reasons == reasons_with_none_cfg
