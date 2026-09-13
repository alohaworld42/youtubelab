"""The channel's no-repeat rule: the same scene must never visibly play twice.

Covers the three 2026-07-22 fixes for the "gleiche Szene doppelt hintereinander"
defect the first German render shipped with:

  1. edit._extend_to_window — a take shorter than its cut window is PING-PONGED
     (forward, then reversed), never restart-looped from frame 0. At draft
     quality every take is ~40% shorter than its window, so the old restart
     loop replayed the start of the scene mid-window on essentially every shot.
  2. director storyboard-beat dedupe — a verse with more shots than board beats
     used to cycle beats verbatim (measured: "The kids join hands and circle …"
     twice AND "wave … one last time as the song ends" twice in one verse).
  3. storyboard fallback celebrate pool widened to 6 distinct beats (the final
     verse absorbs the outro and is routinely the longest) — while still
     passing the storyboard's own verdict gate.

CPU-only: moviepy ColorClip/VideoClip synthesis, no GPU, no LLM, no network.
"""
import collections

import numpy as np
from moviepy import VideoClip

from pipeline.kidsong import director, storyboard
from pipeline.kidsong.edit import _extend_to_window


# ------------------------------------------------------------ 1. ping-pong ---
def _ramp_clip(duration=2.0, fps=8):
    """A clip whose frame brightness encodes its own timestamp (0..255 over
    `duration`), so a restart (jump back to frame 0) is detectable by value."""
    def frame(t):
        v = int(255 * min(1.0, max(0.0, t / duration)))
        return np.full((8, 8, 3), v, dtype=np.uint8)

    return VideoClip(frame, duration=duration).with_fps(fps)


def _brightness(clip, t):
    return float(clip.get_frame(t)[0, 0, 0])


def test_long_take_is_trimmed_not_padded():
    out = _extend_to_window(_ramp_clip(2.0), 1.5)
    assert abs(out.duration - 1.5) < 1e-6


def test_short_take_fills_the_window_exactly():
    out = _extend_to_window(_ramp_clip(2.0), 3.0)
    assert abs(out.duration - 3.0) < 1e-6


def test_short_take_reverses_instead_of_restarting():
    """Just past the take's end the ping-pong shows the LATE frames again
    (reversed), while a restart loop would jump back to the dark frame 0.
    take=2s → at t=2.2s pingpong shows ~t=1.8s (bright ~230); restart would
    show ~t=0.2s (dark ~25)."""
    out = _extend_to_window(_ramp_clip(2.0), 3.0)
    just_after_wrap = _brightness(out, 2.2)
    assert just_after_wrap > 128, (
        f"frame at t=2.2s is dark ({just_after_wrap}) — the take restarted "
        "from frame 0 instead of ping-ponging"
    )


def test_window_longer_than_pingpong_loops_seamfree():
    """Windows past 2x the take loop the SYMMETRIZED clip. Its wrap point is
    its own first frame, so the value just after the 2nd wrap (t=4.2s → local
    t=0.2s of the pingpong) must be dark — i.e. continuous with the pingpong's
    own start, no half-cycle jump."""
    out = _extend_to_window(_ramp_clip(2.0), 5.5)
    assert abs(out.duration - 5.5) < 1e-6
    assert _brightness(out, 4.2) < 128  # forward phase of the looped pingpong


def test_small_shortfall_slows_down_never_reverses():
    """The final-quality case (max_frames=81 takes vs the old 4.5s slots):
    a window within _MAX_SLOW_STRETCH of the take is covered by slowing the
    take, so brightness must increase MONOTONICALLY to the very end — any
    reversal (ping-pong) would dim again after the take's own length. The
    2026-07-25 user report was scenes visibly playing forward-then-backward."""
    out = _extend_to_window(_ramp_clip(3.375), 3.6)  # ratio 1.067 < 1.15
    assert abs(out.duration - 3.6) < 1e-6
    samples = [_brightness(out, t) for t in (0.2, 1.0, 2.0, 3.0, 3.5)]
    assert samples == sorted(samples), (
        f"brightness not monotonic ({samples}) — the fill reversed or "
        "restarted instead of slow-stretching"
    )
    assert samples[-1] > 200  # the end of the window shows the take's END


def test_large_shortfall_still_pingpongs():
    """Draft tier (frames_scale 0.6, ~40% short) stays on the ping-pong: a
    1.5x window must NOT be slow-stretched (that much slowdown reads as slo-mo)."""
    out = _extend_to_window(_ramp_clip(2.0), 3.0)  # ratio 1.5 > 1.15
    assert _brightness(out, 2.2) > 128  # reversed tail, not a slowed ramp


# ------------------------------------- 1b. director slot cap vs render cap ---
def test_director_slots_never_exceed_the_renderable_take_length():
    """kidsong.shot.max_frames caps how long a take can BE (81f = 3.375s at
    the i2v-hires envelope); a slot longer than that minus the editor handle
    cannot be covered by real footage and forces the editor fill. The
    director's max bound must clamp to it."""
    cfg = {"kidsong": {
        "director": {"max_shot_seconds": 4.5, "min_shot_seconds": 1.4},
        "shot": {"max_frames": 81, "fps": 24},
    }}
    min_s, max_s = director._shot_bounds(cfg)
    assert max_s <= (81 - 6) / 24.0 + 1e-9
    assert min_s <= max_s


def test_director_bounds_unchanged_without_a_shot_cap():
    cfg = {"kidsong": {"director": {"max_shot_seconds": 4.5, "min_shot_seconds": 1.4}}}
    min_s, max_s = director._shot_bounds(cfg)
    assert max_s == 4.5 and min_s == 1.4  # default max_frames=241 never binds


# ------------------------------------------- 2. storyboard-beat dedupe -------
def _song_with_long_last_verse():
    """Same shape as the shipped German render: 4 verses, the last one long
    enough (absorbing the outro) to need more shots than the board has beats."""
    verses = [
        {"lines": ["Alle Vögel sind schon da", "alle Vögel, alle"],
         "scene": "small round cartoon birds hop along a branch in a blossoming garden"},
        {"lines": ["Welch ein Singen, Musizieren", "Pfeifen, Zwitschern, Tirilieren"],
         "scene": "the cartoon birds sing as little musical notes float in the sunny air"},
        {"lines": ["Sing und klatsch mit mir mit", "die Vögel sind so froh"],
         "scene": "the three toddlers clap their hands together in the blooming garden"},
        {"lines": ["Frühling will nun einmarschieren", "kommt mit Sang und Schalle"],
         "scene": "cartoon flowers open across a green meadow under falling blossom petals"},
    ]
    return {
        "title": "Alle Vögel sind schon da - Test",
        "characters": "Three adorable Black toddlers: Zuri, Kofi and Nala.",
        "verses": verses,
    }


def _plan(song, cfg=None):
    cfg = cfg or {"kidsong": {"seed": 20260717}}
    n = len(song["verses"])
    verse_times = [(i * 13.0, (i + 1) * 13.0) for i in range(n)]
    verse_times[-1] = (verse_times[-1][0], 60.0)  # long final verse
    beats = {"bpm": 100, "beat_times": [i * 0.6 for i in range(200)]}
    return director._fallback_planner(song, verse_times, beats, cfg)


def test_no_verse_repeats_a_storyboard_action_verbatim():
    song = _song_with_long_last_verse()
    song["storyboard"] = storyboard.fallback_storyboard(song)
    shots = _plan(song)

    per_verse = collections.defaultdict(list)
    for s in shots:
        per_verse[s["verse"]].append(str(s.get("action")))
    for verse, actions in per_verse.items():
        counted = collections.Counter(actions)
        dups = {a: n for a, n in counted.items() if n > 1}
        assert not dups, f"verse {verse} repeats action(s) verbatim: {dups}"


def test_no_two_adjacent_shots_share_an_action():
    song = _song_with_long_last_verse()
    song["storyboard"] = storyboard.fallback_storyboard(song)
    shots = _plan(song)
    for prev, cur in zip(shots, shots[1:]):
        assert str(prev.get("action")) != str(cur.get("action")), (
            f"adjacent shots {prev['id']}/{cur['id']} plan the identical scene"
        )


def test_one_last_time_appears_at_most_once_per_episode():
    """The absurdity the user saw: the goodbye beat ("one last time") planned
    twice. There is only one last time."""
    song = _song_with_long_last_verse()
    song["storyboard"] = storyboard.fallback_storyboard(song)
    shots = _plan(song)
    goodbyes = [s for s in shots if "one last time" in str(s.get("action", ""))]
    assert len(goodbyes) <= 1, [s["id"] for s in goodbyes]


# ------------------------------------- 3. widened pool still passes gate -----
def test_fallback_storyboard_still_passes_its_own_verdict():
    song = _song_with_long_last_verse()
    board = storyboard.fallback_storyboard(song)
    verdict = storyboard._storyboard_verdict(board, song)
    assert verdict.get("accept"), verdict.get("reasons")


def test_celebrate_pool_has_six_distinct_beats():
    song = _song_with_long_last_verse()
    board = storyboard.fallback_storyboard(song)
    last = board["verses"][-1]["beats"]
    actions = [b["action"] for b in last]
    assert len(actions) >= 6
    assert len(set(actions)) == len(actions), "celebrate pool contains duplicates"


# ------------------------------ 4. no CROSS-verse repeat (storyboard layer) ---
# The gap the "Rain, Rain, Go Away" render shipped through: the within-verse
# dedupe above never looked ACROSS verses, and the fallback's two middle stages
# ("try"/"grow") cycled, so a 5-verse song got byte-identical beats for verse 1
# and verse 3 (its whole umbrella "try" verse replayed as verse 3). Four
# distinct middle stages now cover every 3-6-verse song — the channel's real
# range — with a unique beat set per verse.
import pytest  # noqa: E402


def _all_actions_by_verse(board):
    out = {}
    for v in board["verses"]:
        out[v["verse"]] = [b["action"] for b in v["beats"]]
    return out


# The invader guard: a SOLO (performer=child) beat whose action or gaze
# mentions other children puts "The child looks toward the other kids" into a
# single-child i2v prompt — and LTX paints those kids IN as non-cast children
# (measured live: 5 of 16 takes in the pixar-full render invaded). Solo beats
# keep their focus on the child's own body, prop or surroundings.
_OTHER_KID_PHRASES = ("other kids", "other children", "a friend", "the friend",
                      "their friends", "the group", "another child")


@pytest.mark.parametrize("subject", ["a bright yellow sunflower", None])
def test_fallback_solo_beats_never_reference_other_children(monkeypatch, subject):
    monkeypatch.setattr(director, "song_story_subject", lambda song: subject)
    song = {
        "title": "A Song About Something Wonderful",
        "characters": "Three adorable Black toddlers: Zuri, Kofi and Nala.",
        "verses": [{"lines": [f"line {i}"], "scene": f"scene {i}"} for i in range(6)],
    }
    board = storyboard.fallback_storyboard(song)
    for v in board["verses"]:
        for b in v["beats"]:
            if b.get("performer") != "child":
                continue
            blob = (str(b.get("action", "")) + " " + str(b.get("gaze", ""))).lower()
            hits = [p for p in _OTHER_KID_PHRASES if p in blob]
            assert not hits, (
                f"verse {v['verse']} solo beat references other children {hits}: "
                f"action={b.get('action')!r} gaze={b.get('gaze')!r}"
            )


@pytest.mark.parametrize("n_verses", [3, 4, 5, 6])
@pytest.mark.parametrize("subject", ["a bright yellow sunflower", None])
def test_fallback_storyboard_never_repeats_a_beat_across_verses(monkeypatch, n_verses, subject):
    """No action string may appear in two different verses of the board, for
    every verse count the channel ships (3-6) and both the story-subject and
    subject-less regimes. This is the storyboard-layer guarantee behind the
    planner-layer no-repeat tests above."""
    monkeypatch.setattr(director, "song_story_subject", lambda song: subject)
    song = {
        "title": "A Song About Something Wonderful",
        "characters": "Three adorable Black toddlers: Zuri, Kofi and Nala.",
        "verses": [{"lines": [f"line {i}"], "scene": f"scene {i}"} for i in range(n_verses)],
    }
    board = storyboard.fallback_storyboard(song)

    seen = {}
    for verse, actions in _all_actions_by_verse(board).items():
        for action in actions:
            prior = seen.get(action)
            assert prior is None, (
                f"beat {action!r} appears in verse {prior} AND verse {verse} "
                f"(n={n_verses}, subject={subject!r}) — a repeated scene"
            )
            seen[action] = verse


# ---------------------------- 5. the finale is a distinct climax, not a reuse -
def _song_ABCBA():
    """A/B/C/B/A: verse 3 reprises verse 1 (mid-song chorus, SHOULD reuse) and
    verse 4 — the finale — reprises verse 0 (which must NOT collapse: the last
    verse is the narrative climax the storyboard stages as 'celebrate')."""
    A = {"lines": ["Clap your hands and sing with me", "One two three"],
         "scene": "the toddlers clap on a sunny porch"}
    B = {"lines": ["Rain rain go away", "come again another day"],
         "scene": "the toddlers watch raindrops on a big window"}
    C = {"lines": ["Sun sun come and stay", "we will dance today"],
         "scene": "bright sun as the toddlers dance in the wet grass"}
    return {
        "title": "Rain Rain Go Away - Sing-Along",
        "characters": "Three adorable Black toddlers: Zuri, Kofi and Nala.",
        "verses": [dict(A), dict(B), dict(C), dict(B), dict(A)],
    }


def test_final_verse_is_not_chorus_collapsed_when_a_storyboard_drives_the_arc():
    song = _song_ABCBA()
    song["storyboard"] = storyboard.fallback_storyboard(song)
    shots = _plan(song, cfg={"kidsong": {"seed": 20260717}})

    by_verse = collections.defaultdict(list)
    for s in shots:
        by_verse[s["verse"]].append(s)
    last = max(by_verse)

    # The finale renders fresh — none of its shots reuse an earlier shot.
    assert all(not s.get("reuse_of") for s in by_verse[last]), (
        f"final verse {last} was chorus-collapsed: "
        f"{[s.get('reuse_of') for s in by_verse[last]]}"
    )
    # …and its shots carry the storyboard's celebrate beats, not clap filler.
    finale_actions = " ".join(str(s.get("action", "")).lower() for s in by_verse[last])
    assert "celebrat" in finale_actions or "song ends" in finale_actions or "bow" in finale_actions, (
        f"final verse does not read as a celebration finale: {finale_actions!r}"
    )
    # No scene repeats WITHIN the finale (the s16==s20 class).
    fin_sigs = [s.get("reuse_of") or str(s.get("action")) for s in by_verse[last]]
    assert len(fin_sigs) == len(set(fin_sigs)), f"finale repeats a scene: {fin_sigs}"

    # Regression guard: a MID-song reprise (verse 3 == verse 1) still reuses —
    # protecting the finale must not disable chorus reuse everywhere.
    assert any(s.get("reuse_of") for s in by_verse[3]), (
        "mid-song chorus verse 3 should still reuse verse 1's shots"
    )


# ------------------------ 6. story-mode boards keep every no-repeat guarantee -
_STORY_CFG = {"kidsong": {"staging": "story"}}


@pytest.mark.parametrize("n_verses", [3, 4, 5, 6])
@pytest.mark.parametrize("subject", ["a bright yellow sunflower", None])
def test_story_mode_no_cross_verse_repeats_after_substitution(monkeypatch, n_verses, subject):
    """Uniqueness must hold on the text the RENDERER sees: after substituting
    the planner's protagonist/partner names into the placeholders."""
    monkeypatch.setattr(director, "song_story_subject", lambda song: subject)
    song = {
        "title": "A Song About Something Wonderful",
        "characters": "Three adorable Black toddlers: Zuri, Kofi and Nala.",
        "verses": [{"lines": [f"line {i}"], "scene": f"scene {i}"} for i in range(n_verses)],
    }
    board = storyboard.fallback_storyboard(song, _STORY_CFG)
    seen = {}
    for v in board["verses"]:
        for b in v["beats"]:
            action = b["action"].replace("{name}", "Zuri").replace("{partner}", "Kofi")
            prior = seen.get(action)
            assert prior is None, (
                f"substituted beat {action!r} appears in verse {prior} AND "
                f"verse {v['verse']} (n={n_verses}, subject={subject!r})"
            )
            seen[action] = v["verse"]


@pytest.mark.parametrize("subject", ["a bright yellow sunflower", None])
def test_story_mode_solo_beats_never_reference_other_children(monkeypatch, subject):
    """The invader rule holds in story mode too. The literal "{partner}"
    placeholder is exempt: it marks a PLANNED two-child beat the planner casts
    explicitly (characters=[protagonist, partner]) — the opposite of the
    phantom-friend text this guard exists to kill."""
    monkeypatch.setattr(director, "song_story_subject", lambda song: subject)
    song = {
        "title": "A Song About Something Wonderful",
        "characters": "Three adorable Black toddlers: Zuri, Kofi and Nala.",
        "verses": [{"lines": [f"line {i}"], "scene": f"scene {i}"} for i in range(6)],
    }
    board = storyboard.fallback_storyboard(song, _STORY_CFG)
    for v in board["verses"]:
        for b in v["beats"]:
            if b.get("performer") != "child" or b.get("role") == "partner":
                continue
            blob = (str(b.get("action", "")) + " " + str(b.get("gaze", ""))).lower()
            hits = [p for p in _OTHER_KID_PHRASES if p in blob]
            assert not hits, (
                f"verse {v['verse']} solo beat references other children {hits}: "
                f"action={b.get('action')!r} gaze={b.get('gaze')!r}"
            )
