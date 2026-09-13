"""`--restyle` — re-render one episode under a different look, everything else held.

The render-style registry (IMP-002) let a channel PICK a look. It never let
anyone COMPARE two, which is why every style question in `docs/quality/` —
including the prompt-length A/B of IMP-035/QM-030 — sat unmeasured. The three
existing entry points cannot do it:

  * `--finalize` re-renders under a different QUALITY TIER, so a style read
    through it is confounded by frame count and the hires pass;
  * `--resume` renders only what is missing, which for a finished episode is
    nothing;
  * a fresh run writes a different song and a different shot list, so the two
    arms share no material at all.

`restyle_episode` reuses the source's song, shot list, sung audio and — the
property that makes it an experiment rather than a second roll of the dice —
**every shot's seed**. Only the style moves.

CPU-only: `generate_kidsong` is monkeypatched out, so no GPU, no ComfyUI, no
network. What is under test is the setup the render would receive.
"""
import json
import os
import struct
import wave

import pytest

from pipeline.kidsong import runstate
from pipeline.kidsong.generate import (
    _build_parser,
    _dispatch,
    _reserve_rerender_base,
    restyle_episode,
)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _write_wav(path, seconds=2.0, rate=8000):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(struct.pack("<h", 0) * int(rate * seconds))


@pytest.fixture
def episode_on_disk(tmp_path):
    """A finished episode: song, shot list with per-shot seeds, takes, wav."""
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    base = "20260731-090000-kidsong-restyle-test"
    shots_dir = runstate.shots_dir_path(str(out_dir), base)
    os.makedirs(shots_dir, exist_ok=True)
    _write_wav(out_dir / f"{base}.wav")

    song = {"title": "Restyle Test", "description": "d", "tags": ["t"],
            "characters": "Three toddlers: Zuri; Kofi; Nala",
            "verses": [{"scene": "playroom", "lines": ["a", "b"]}]}
    (out_dir / f"{base}-song.json").write_text(json.dumps(song), encoding="utf-8")

    shots = []
    for i in range(3):
        take = os.path.join(shots_dir, f"s{i:02d}_a0.mp4")
        with open(take, "wb") as f:
            f.write(b"\0" * 1024)
        shot = {"id": f"s{i:02d}", "verse": 0, "start": i * 2.0, "end": (i + 1) * 2.0,
                "shot_type": "wide", "characters": ["all"], "action": "clapping",
                "setting": "a playroom", "camera": "static", "reuse_of": None,
                "seed": 7000 + i * 13, "status": "planned"}
        runstate.mark_rendered(shot, take, score=0.9, attempts=1)
        shots.append(shot)
    runstate.save_shotlist(str(out_dir), base, {"shots": shots})

    cfg = {
        "_root": _REPO_ROOT,
        "paths": {"output_dir": str(out_dir)},
        "video": {"max_seconds": 60},
        "captions": {"uppercase": True},
        "kidsong": {
            "mode": "director",
            "quality": "draft",
            "render_style": "pixar_toon",
            "render_styles": {
                "pixar_toon": {"prompt_skeleton": "a 3D CGI toon animation."},
                "pixar_toon_concise": {"prompt_tail": "A warm key light."},
            },
        },
    }
    return {"out_dir": str(out_dir), "base": base, "song": song, "cfg": cfg,
            "seeds": [s["seed"] for s in shots]}


@pytest.fixture
def captured(monkeypatch):
    """Swallow the render; record what it was asked to do."""
    import pipeline.kidsong.generate as gen

    box = {"n": 0}

    def fake(topic=None, do_upload=False, cfg=None, on_progress=None,
             resume_base=None, runlog=None):
        box["n"] += 1
        box.update(resume_base=resume_base, cfg=cfg, do_upload=do_upload)
        return {"video_path": "x.mp4", "status": "approved"}

    monkeypatch.setattr(gen, "generate_kidsong", fake)
    return box


# ------------------------------------------------------- the A/B guarantees --
def test_the_restyle_renders_under_the_requested_style(episode_on_disk, captured):
    restyle_episode(episode_on_disk["base"], "pixar_toon_concise",
                    cfg=episode_on_disk["cfg"])
    assert captured["n"] == 1
    assert captured["cfg"]["kidsong"]["render_style"] == "pixar_toon_concise"


def test_the_quality_tier_is_not_touched(episode_on_disk, captured):
    """The confound `--finalize` would have introduced. Both arms must render at
    whatever tier the config already says, or the comparison measures two things."""
    restyle_episode(episode_on_disk["base"], "pixar_toon_concise",
                    cfg=episode_on_disk["cfg"])
    assert captured["cfg"]["kidsong"]["quality"] == "draft"


def test_every_shot_keeps_its_seed(episode_on_disk, captured):
    """The property that makes this an experiment. No seed anywhere in the
    pipeline is derived from the render style, so identical seeds + identical
    shot plan means the only difference on screen is the look."""
    out_dir, base = episode_on_disk["out_dir"], episode_on_disk["base"]
    restyle_episode(base, "pixar_toon_concise", cfg=episode_on_disk["cfg"])

    new_shots = runstate.load_shotlist(out_dir, captured["resume_base"])["shots"]
    assert [s["seed"] for s in new_shots] == episode_on_disk["seeds"]


def test_the_song_shotplan_and_audio_are_reused_and_the_ledger_is_reset(
    episode_on_disk, captured
):
    out_dir, base = episode_on_disk["out_dir"], episode_on_disk["base"]
    restyle_episode(base, "pixar_toon_concise", cfg=episode_on_disk["cfg"])
    new_base = captured["resume_base"]

    assert runstate.load_song(out_dir, new_base)["title"] == "Restyle Test"
    new_shots = runstate.load_shotlist(out_dir, new_base)["shots"]
    assert [s["id"] for s in new_shots] == ["s00", "s01", "s02"]
    assert [(s["start"], s["end"]) for s in new_shots] == [(0.0, 2.0), (2.0, 4.0), (4.0, 6.0)]
    # reset, so the new render redraws instead of adopting the old takes
    assert all(s["status"] == "planned" for s in new_shots)
    assert all("take" not in s for s in new_shots)
    assert os.path.exists(runstate.wav_path(out_dir, new_base))


def test_the_source_episode_is_never_touched(episode_on_disk, captured):
    out_dir, base = episode_on_disk["out_dir"], episode_on_disk["base"]
    restyle_episode(base, "pixar_toon_concise", cfg=episode_on_disk["cfg"])

    assert captured["resume_base"] != base
    src = runstate.load_shotlist(out_dir, base)["shots"]
    assert all(s["status"] == "rendered" for s in src)
    assert all(s.get("take") for s in src)


def test_the_new_base_names_the_style_it_rendered(episode_on_disk, captured):
    """Two arms of the same A/B must be distinguishable on disk without opening
    a run log."""
    restyle_episode(episode_on_disk["base"], "pixar_toon_concise",
                    cfg=episode_on_disk["cfg"])
    assert captured["resume_base"].endswith("-style-pixar-toon-concise")


def test_a_second_run_of_the_same_arm_does_not_collide(episode_on_disk, captured):
    out_dir, base = episode_on_disk["out_dir"], episode_on_disk["base"]
    restyle_episode(base, "pixar_toon_concise", cfg=episode_on_disk["cfg"])
    first = captured["resume_base"]
    restyle_episode(base, "pixar_toon_concise", cfg=episode_on_disk["cfg"])
    assert captured["resume_base"] != first
    assert captured["resume_base"].endswith("-v2")


# -------------------------------------------------------------- refusals ----
def test_an_unknown_style_is_refused_not_silently_fallen_back(episode_on_disk, captured):
    """`resolve_style` warns and falls back to pixar_toon for an unknown name —
    correct for a render, fatal for an A/B: arm B would silently BE arm A and
    the comparison would report "no difference"."""
    with pytest.raises(ValueError, match="Unknown render style"):
        restyle_episode(episode_on_disk["base"], "does_not_exist",
                        cfg=episode_on_disk["cfg"])
    assert captured["n"] == 0


def test_the_error_lists_the_styles_that_do_exist(episode_on_disk):
    with pytest.raises(ValueError) as excinfo:
        restyle_episode(episode_on_disk["base"], "typo", cfg=episode_on_disk["cfg"])
    assert "pixar_toon_concise" in str(excinfo.value)


def test_a_missing_style_argument_is_refused(episode_on_disk, captured):
    with pytest.raises(ValueError, match="needs --style"):
        restyle_episode(episode_on_disk["base"], None, cfg=episode_on_disk["cfg"])
    assert captured["n"] == 0


def test_an_episode_without_its_audio_cannot_be_restyled(episode_on_disk, captured):
    """Same guard finalize has: the re-render runs against the ORIGINAL sung
    audio so the beat grid still lines up."""
    os.remove(runstate.wav_path(episode_on_disk["out_dir"], episode_on_disk["base"]))
    with pytest.raises(FileNotFoundError):
        restyle_episode(episode_on_disk["base"], "pixar_toon_concise",
                        cfg=episode_on_disk["cfg"])
    assert captured["n"] == 0


def test_an_unknown_episode_is_refused(episode_on_disk, captured):
    with pytest.raises(FileNotFoundError):
        restyle_episode("no-such-episode", "pixar_toon_concise",
                        cfg=episode_on_disk["cfg"])
    assert captured["n"] == 0


# ------------------------------------------------------------------- CLI ----
def test_the_cli_accepts_restyle_and_style():
    args = _build_parser().parse_args(["--restyle", "ep-123", "--style", "claymation"])
    assert args.restyle == "ep-123"
    assert args.style == "claymation"


def test_dispatch_routes_restyle_to_restyle_episode(monkeypatch):
    import pipeline.kidsong.generate as gen

    seen = {}
    monkeypatch.setattr(gen, "restyle_episode",
                        lambda base, style, **kw: seen.update(base=base, style=style) or {})
    args = _build_parser().parse_args(["--restyle", "ep-123", "--style", "felt"])
    _dispatch(args, {"kidsong": {}}, runlog=None)
    assert seen == {"base": "ep-123", "style": "felt"}


def test_restyle_wins_over_a_stray_finalize(monkeypatch):
    """Both name a re-render of an existing base; restyle is the more specific
    request, and running finalize instead would silently change the tier."""
    import pipeline.kidsong.generate as gen

    seen = []
    monkeypatch.setattr(gen, "restyle_episode", lambda *a, **k: seen.append("restyle") or {})
    monkeypatch.setattr(gen, "finalize_episode", lambda *a, **k: seen.append("finalize") or {})
    args = _build_parser().parse_args(
        ["--restyle", "ep", "--style", "felt", "--finalize", "ep"]
    )
    _dispatch(args, {"kidsong": {}}, runlog=None)
    assert seen == ["restyle"]


def test_a_plain_run_is_unaffected_by_the_new_flags(monkeypatch):
    import pipeline.kidsong.generate as gen

    seen = {}
    monkeypatch.setattr(gen, "generate_kidsong",
                        lambda topic=None, *a, **kw: seen.update(topic=topic) or {})
    args = _build_parser().parse_args(["--topic", "counting"])
    assert args.restyle is None and args.style is None
    _dispatch(args, {"kidsong": {}}, runlog=None)
    assert seen == {"topic": "counting"}


# ------------------------------------------------------ base reservation ----
def test_reserve_rerender_base_walks_past_existing_artifacts(tmp_path):
    out = str(tmp_path)
    assert _reserve_rerender_base(out, "ep", "style-felt") == "ep-style-felt"
    os.makedirs(runstate.shots_dir_path(out, "ep-style-felt"))
    assert _reserve_rerender_base(out, "ep", "style-felt") == "ep-style-felt-v2"
