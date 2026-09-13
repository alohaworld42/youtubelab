"""Story-driven casting (kidsong.staging="story"): the episode's red thread
owns WHO is on screen. One protagonist child carries every solo body slot
through all verses; a partner child appears only in the board's role="partner"
beat (a planned TWO-child shot, never a solo shot with a phantom friend);
"all" stages only at gathering moments (opening wide, share, finale) — and the
last shot of the episode is forced back to the whole cast (closing bookend).

Ensemble mode (default / staging absent / no board) must stay byte-identical
to the pre-story planner, including its round-robin rotation.

CPU-only: no GPU, no LLM, no network.
"""
import copy

import pytest

from pipeline.kidsong import cast, director, storyboard

_STORY_CFG = {"kidsong": {"seed": 20260717, "staging": "story", "storyboard": {"enabled": True}}}


def _song(n=5, title="Rain, Rain, Go Away - A Sunny Sing-Along"):
    return {
        "title": title,
        "characters": "Three adorable Black toddlers: Zuri, a girl; Kofi, a boy; and Nala, a girl.",
        "verses": [{"lines": [f"line {i}a", f"line {i}b"], "scene": f"the kids play {i}"}
                   for i in range(n)],
    }


def _plan(song, cfg):
    n = len(song["verses"])
    vt = [(i * 10.7, (i + 1) * 10.7) for i in range(n)]
    beats = {"bpm": 100, "beat_times": [i * 0.6 for i in range(200)]}
    return director._fallback_planner(song, vt, beats, cfg)


def _story_plan(n=5, title="Rain, Rain, Go Away - A Sunny Sing-Along"):
    song = _song(n, title)
    song["storyboard"] = storyboard.fallback_storyboard(song, _STORY_CFG)
    return song, _plan(song, _STORY_CFG)


def _names(title, cfg=None):
    prot_id = cast.protagonist_for(title, cfg or _STORY_CFG)
    prot = cast.character(prot_id)["name"]
    partner = cast.character(cast.partner_for(prot_id, cfg or _STORY_CFG))["name"]
    return prot, partner


# ------------------------------------------------------------- red thread ---
def test_protagonist_carries_every_solo_body_slot():
    song, shots = _story_plan()
    prot, _partner = _names(song["title"])
    solos = [s for s in shots if len(s["characters"]) == 1 and s["characters"] != ["all"]]
    assert solos, "a story plan must have solo shots"
    for s in solos:
        assert s["characters"] == [prot], (
            f"{s['id']} stages {s['characters']} — the rotation is back; "
            f"every solo slot belongs to the protagonist {prot}"
        )


def test_no_round_robin_rotation_in_story_mode():
    song, shots = _story_plan()
    prot, partner = _names(song["title"])
    on_screen = {n for s in shots for n in s["characters"] if n != "all"}
    # Only the protagonist and (in the partner beat) the partner ever appear
    # by name — the third cast member exists only inside "all" group shots.
    assert on_screen <= {prot, partner}


def test_partner_beat_plans_exactly_protagonist_and_partner_never_closeup():
    song, shots = _story_plan()
    prot, partner = _names(song["title"])
    pairs = [s for s in shots if len(s["characters"]) == 2]
    assert len(pairs) == 1, f"exactly one planned partner moment, got {[s['id'] for s in pairs]}"
    pair = pairs[0]
    assert set(pair["characters"]) == {prot, partner}
    assert pair["shot_type"] != "closeup"
    assert prot in pair["action"] and partner in pair["action"]
    assert "{partner}" not in pair["action"] and "{name}" not in pair["action"]


def test_partner_moment_keeps_the_prop():
    song, shots = _story_plan()
    pair = next(s for s in shots if len(s["characters"]) == 2)
    assert pair.get("prop"), "the handover beat's interaction object must ride into the shot"


def test_props_are_plumbed_from_board_beats():
    song, shots = _story_plan()
    with_prop = [s for s in shots if s.get("prop")]
    assert with_prop, "try-verse shots must carry the structured prop"
    for s in with_prop:
        assert s["prop"] in s["action"], (
            f"{s['id']}: prop {s['prop']!r} not mentioned by action {s['action']!r}"
        )


# ---------------------------------------------------------------- bookends ---
def test_opening_wide_is_untouched_and_closing_shot_is_the_whole_cast():
    song, shots = _story_plan()
    first = min(shots, key=lambda s: (s["verse"], s["start"]))
    assert first["shot_type"] == "wide" and first["characters"] == ["all"]
    last_verse = max(s["verse"] for s in shots)
    finale = [s for s in shots if s["verse"] == last_verse and not s.get("reuse_of")]
    closer = finale[-1]
    assert closer["characters"] == ["all"], "the episode must close on the whole cast"
    assert closer["shot_type"] != "closeup"
    prot, _ = _names(song["title"])
    assert prot not in closer["action"], (
        "a forced group closer must not keep a protagonist-solo action "
        f"(head-count contradiction): {closer['action']!r}"
    )


@pytest.mark.parametrize("n_verses", [3, 4, 5, 6])
def test_closing_group_invariant_across_verse_counts(n_verses):
    _song_unused, shots = _story_plan(n_verses)
    last_verse = max(s["verse"] for s in shots)
    finale = [s for s in shots if s["verse"] == last_verse and not s.get("reuse_of")]
    assert finale[-1]["characters"] == ["all"]


# ------------------------------------------------------------ determinism ---
def test_same_title_same_protagonist():
    s1, shots1 = _story_plan(title="Rain, Rain, Go Away")
    s2, shots2 = _story_plan(title="Rain, Rain, Go Away")
    p1, _ = _names(s1["title"])
    p2, _ = _names(s2["title"])
    assert p1 == p2
    solos1 = [s["characters"] for s in shots1 if len(s["characters"]) == 1]
    solos2 = [s["characters"] for s in shots2 if len(s["characters"]) == 1]
    assert solos1 == solos2


def test_protagonist_override_is_honored():
    cfg = copy.deepcopy(_STORY_CFG)
    cfg["kidsong"]["protagonist"] = "nala"
    song = _song()
    song["storyboard"] = storyboard.fallback_storyboard(song, cfg)
    shots = _plan(song, cfg)
    solos = {tuple(s["characters"]) for s in shots
             if len(s["characters"]) == 1 and s["characters"] != ["all"]}
    assert solos == {("Nala",)}


# ------------------------------------------------- ensemble byte-identity ---
def test_ensemble_mode_is_byte_identical_regardless_of_staging_key():
    song_a = _song()
    song_a["storyboard"] = storyboard.fallback_storyboard(song_a)
    base_cfg = {"kidsong": {"seed": 20260717, "storyboard": {"enabled": True}}}
    ens_cfg = copy.deepcopy(base_cfg)
    ens_cfg["kidsong"]["staging"] = "ensemble"

    plan_absent = _plan(copy.deepcopy(song_a), base_cfg)
    plan_ensemble = _plan(copy.deepcopy(song_a), ens_cfg)
    assert plan_absent == plan_ensemble


def test_story_without_a_board_falls_back_to_rotation_byte_identically():
    """staging=story but no storyboard attached: nothing to cast from — the
    plan must equal the ensemble plan exactly (logged fallback)."""
    song = _song()  # deliberately NO storyboard key
    plan_story = _plan(copy.deepcopy(song), _STORY_CFG)
    plan_ens = _plan(copy.deepcopy(song),
                     {"kidsong": {"seed": 20260717, "staging": "ensemble",
                                  "storyboard": {"enabled": True}}})
    assert plan_story == plan_ens


# ------------------------------------------------------------ chorus reuse ---
def test_chorus_reuse_propagates_story_casting():
    """A mid-song reprise copies its source shots verbatim — protagonist
    casting included."""
    song = _song(5)
    # verse 3 repeats verse 1's lines exactly -> chorus reuse
    song["verses"][3] = copy.deepcopy(song["verses"][1])
    song["storyboard"] = storyboard.fallback_storyboard(song, _STORY_CFG)
    shots = _plan(song, _STORY_CFG)
    prot, partner = _names(song["title"])
    reused = [s for s in shots if s.get("reuse_of")]
    assert reused, "the repeated verse must reuse"
    for s in reused:
        for name in s["characters"]:
            assert name in ("all", prot, partner)
