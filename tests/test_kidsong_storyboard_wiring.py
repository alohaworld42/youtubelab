"""Storyboard wiring: `_fallback_planner` consuming `song["storyboard"]` beats
in place of the canned per-shot actions, and `_normalize_shots` shielding those
gated beats from the ungated LLM action overlay.

CPU-only, no LLM: boards are hand-built dicts in the exact schema
pipeline/kidsong/storyboard.py produces ({name} placeholders, per-beat gaze /
performer / camera_address). The wiring's contract (see director.py):

  - beat j drives shot j, cycling when a verse has more shots than beats;
  - "child" beats substitute the planner's own rotated subject; a "child" beat
    on the subject-less establishing wide leaves that shot untouched;
  - "all" beats phrase the ensemble; "story_subject" beats pass through
    verbatim and flow into the existing subject-only derivation;
  - gazes ride along, but placeholders and camera/lens gazes never do — the
    decreep designation pass stays the single owner of the camera budget;
  - a malformed verse entry (or absent board) leaves that verse byte-identical
    to the planner's own output;
  - `_normalize_shots` keeps a storyboard beat over the LLM's action overlay,
    while every other overlay (shot_type/camera/gaze) behaves as before.
"""
import copy
import json

from pipeline.kidsong import director


def _song(with_board=True, board=None):
    song = {
        "title": "Sunflower Family Tree",
        "characters": "Zuri, Kofi and Nala are best friends",
        "verses": [
            {"scene": "the kids plant a sunflower in a sunny garden",
             "lines": ["We plant a seed today", "We watch it grow and sway"]},
            {"scene": "the kids water the sunflower in the garden",
             "lines": ["Water it every day", "It grows up tall this way"]},
        ],
    }
    if with_board:
        song["storyboard"] = board if board is not None else _board()
    return song


def _board():
    return {
        "verses": [
            {"verse": 0, "beat_goal": "discover the sunflower", "beats": [
                {"action": "{name} spots the tiny seedling and points, eyes wide",
                 "gaze": "down at the seedling", "performer": "child",
                 "camera_address": False},
                {"action": "The kids gather around the seedling, leaning in together",
                 "gaze": "at the seedling", "performer": "all",
                 "camera_address": False},
            ]},
            {"verse": 1, "beat_goal": "water the sunflower", "beats": [
                {"action": "{name} tips a small green watering can carefully, watching what happens",
                 "gaze": "on the watering can", "performer": "child",
                 "camera_address": False},
                {"action": "the sunflower sways a little taller in the sunlight",
                 "gaze": "at the sunflower", "performer": "story_subject",
                 "camera_address": False},
            ]},
        ],
    }


_CFG = {"kidsong": {"seed": 20260717, "storyboard": {"enabled": True},
                    "decreep": {"enabled": True, "hook_camera_budget": 1}}}
_VT = [(0.0, 8.0), (8.0, 16.0)]


def _plan(song, cfg=None):
    return director._fallback_planner(song, _VT, {}, copy.deepcopy(cfg or _CFG))


# ----------------------------------------------------------- consumption ----
def test_storyboard_beats_replace_canned_actions_with_substituted_names():
    shots = _plan(_song())
    sb_shots = [s for s in shots if s.get("action_source") == "storyboard"]
    assert sb_shots, "no storyboard-derived shots at all"
    named = [s for s in sb_shots
             if "spots the tiny seedling" in s["action"]]
    assert named, "the child beat never surfaced"
    for s in named:
        assert "{name}" not in s["action"]
        assert any(n in s["action"] for n in ("Zuri", "Kofi", "Nala"))
        assert s.get("gaze") == "down at the seedling"


def test_establishing_wide_keeps_its_canonical_action():
    """Shot 0 of verse 0 has no planned subject — a '{name}' beat cannot render
    there, so the establishing wide keeps the planner's own ensemble action
    (and is NOT marked as storyboard-derived)."""
    shots = _plan(_song())
    s00 = shots[0]
    assert s00["shot_type"] == "wide"
    assert s00.get("action_source") != "storyboard" or "{name}" not in s00["action"]
    assert "spots the tiny seedling" not in s00["action"] or "The kids" in s00["action"] \
        or any(n in s00["action"] for n in ("Zuri", "Kofi", "Nala"))
    assert "{name}" not in s00["action"]


def test_all_performer_beat_reads_as_ensemble():
    shots = _plan(_song())
    group = [s for s in shots if "gather around the seedling" in s["action"]]
    assert group, "the ensemble beat never surfaced"
    for s in group:
        assert "{name}" not in s["action"]


def _mouse_song(board):
    """A song whose scenes carry a DETECTABLE story subject (the mouse — see
    director's subject vocabulary; 'sunflower' is not in it, which is exactly
    what `test_story_subject_beat_without_detected_subject_degrades` covers)."""
    return {
        "title": "The Tiny Mouse and the Clock",
        "characters": "Zuri, Kofi and Nala are best friends",
        "verses": [
            {"scene": "a tiny round cartoon mouse sits at the base of a grandfather clock in a playroom",
             "lines": ["The mouse runs up the clock", "Tick tock goes the clock"]},
            {"scene": "the tiny cartoon mouse scampers down the grandfather clock",
             "lines": ["The mouse runs down again", "Tick tock says the clock"]},
        ],
        "storyboard": board,
    }


def test_story_subject_beat_flows_into_subject_only_derivation():
    subj_action = "the tiny round cartoon mouse scampers up the tall clock case"
    board = {"verses": [
        {"verse": 0, "beats": [
            {"action": subj_action, "gaze": "up the clock",
             "performer": "story_subject", "camera_address": False},
        ]},
        {"verse": 1, "beats": [
            {"action": "{name} points at the tiny mouse and giggles",
             "gaze": "at the tiny mouse", "performer": "child",
             "camera_address": False},
        ]},
    ]}
    song = _mouse_song(board)
    assert director.song_story_subject(song), "fixture must carry a detectable subject"
    shots = _plan(song)
    subject = [s for s in shots if "scampers up the tall clock case" in s["action"]]
    assert subject, "the story_subject beat never surfaced"
    for s in subject:
        assert s["characters"] == [], "subject beat must render subject-only (no children)"


def test_story_subject_beat_without_detected_subject_degrades_to_child_shot():
    """A performer=story_subject beat in a song whose scenes yield NO detectable
    subject (sunflowers aren't in the vocabulary) degrades gracefully: the beat
    text still ships as the action, on a normal child shot — never a crash,
    never an empty-cast shot the head-count machinery can't ground."""
    shots = _plan(_song())
    subject = [s for s in shots if "sways a little taller" in s["action"]]
    assert subject, "the story_subject beat never surfaced"
    for s in subject:
        assert s["characters"], "without a detected subject the shot stays a child shot"


def test_more_shots_than_beats_cycles():
    board = _board()
    board["verses"][0]["beats"] = board["verses"][0]["beats"][:1]  # 1 beat, n shots
    shots = _plan(_song(board=board))
    v0 = [s for s in shots if s["verse"] == 0 and s.get("action_source") == "storyboard"]
    assert v0, "cycling produced no storyboard shots"
    for s in v0:
        assert "spots the tiny seedling" in s["action"]


def test_placeholder_gaze_and_camera_gaze_never_ride_along():
    board = _board()
    board["verses"][0]["beats"][0]["gaze"] = "..."
    board["verses"][1]["beats"][0]["gaze"] = "straight at the camera"
    shots = _plan(_song(board=board))
    for s in shots:
        if s.get("action_source") == "storyboard":
            assert s.get("gaze") not in ("...", "straight at the camera")


def test_camera_budget_not_exceeded_with_storyboard_present():
    """Designation pass stays the single camera owner: with a board attached,
    total camera-gaze shots still never exceed hook_camera_budget."""
    shots = _plan(_song())
    cam = [s for s in shots if s.get("gaze") == "at the camera"]
    assert len(cam) <= 1


def test_malformed_verse_falls_back_per_verse():
    board = _board()
    board["verses"][0] = {"verse": 0, "beats": [{"action": "   "}]}  # unusable
    shots_with = _plan(_song(board=board))
    v0 = [s for s in shots_with if s["verse"] == 0]
    assert all(s.get("action_source") != "storyboard" for s in v0)
    # verse 1 still consumes its beats
    v1 = [s for s in shots_with if s["verse"] == 1 and s.get("action_source") == "storyboard"]
    assert v1


def test_absent_board_is_byte_identical():
    plain = _plan(_song(with_board=False))
    baseline_song = _song(with_board=False)
    baseline = director._fallback_planner(baseline_song, _VT, {}, copy.deepcopy(_CFG))
    assert json.dumps(plain, sort_keys=True) == json.dumps(baseline, sort_keys=True)
    assert all("action_source" not in s for s in plain)


def test_flag_off_cfg_still_consumes_nothing_when_board_missing():
    cfg = {"kidsong": {"seed": 20260717}}
    shots = _plan(_song(with_board=False), cfg=cfg)
    assert all("action_source" not in s and "gaze" not in s for s in shots)


# ------------------------------------------------------- overlay shielding ---
def test_normalize_shots_llm_action_overlay_suppressed_for_storyboard_shots():
    song = _song()
    raw = {"shots": [
        {"verse": 0, "action": "Zuri does something completely different"},
        {"verse": 1, "action": "Kofi does something completely different"},
    ]}
    shots = director._normalize_shots(raw, song, _VT, {}, copy.deepcopy(_CFG))
    sb = [s for s in shots if s.get("action_source") == "storyboard"]
    assert sb, "storyboard shots vanished in normalization"
    for s in sb:
        assert "completely different" not in s["action"], \
            "LLM overlay replaced a gated storyboard beat"


def test_normalize_shots_llm_overlay_still_applies_to_non_storyboard_shots():
    song = _song(with_board=False)
    raw = {"shots": [{"verse": 0, "action": "Zuri waters the sunflower bed happily"}]}
    shots = director._normalize_shots(raw, song, _VT, {}, copy.deepcopy(_CFG))
    assert any("waters the sunflower bed" in s["action"] for s in shots), \
        "legit LLM overlay stopped working for normal shots"
