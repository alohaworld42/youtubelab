"""Tests for kidsong QUALITY TIERS (Phase 2) and the on-request finalize path.

The kidsong pipeline gets a `kidsong.quality` tier — "draft" (fast throughput:
no 2x hires refine pass, fewer frames per shot, fewer unique shots, automatic
heuristic review) and "final" (full fidelity) — plus a `--finalize <base>` path
that re-renders a chosen draft episode at final quality on demand.

The load-bearing guarantee is REGRESSION SAFETY: the "custom"/absent/unknown
path must be byte-identical to the pre-tier pipeline — same workflow name, same
per-shot frames, same review mode. `resolve_quality` returns the SAME cfg object
untouched on that path, and the effective-settings snapshots below prove it.

CPU-only: no GPU, no ComfyUI, no network, no ffmpeg. The render loop's
ComfyClient, reviewer, Whisper, beat grid, editor and cut QC are monkeypatched;
"renders" are byte-writes to tmp_path, exactly as the resume/render-style suites
already do.
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
    _quality_overrides,
    _read_quality_settings,
    _reserve_final_base,
    _reset_shot_for_finalize,
    finalize_episode,
    generate_kidsong,
    resolve_quality,
)
from pipeline.kidsong.render_style import i2v_workflow_name, workflow_name

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------- sample cfg ---
def _base_cfg(**kidsong_over):
    """A cfg mirroring today's shipped kidsong knobs (before any tier)."""
    cfg = {
        "_root": _REPO_ROOT,
        "kidsong": {
            "render_style": "pixar_toon",
            "shot": {"hires_pass": True, "max_frames": 121, "fps": 24},
            "review": {"reviewer": "external", "script": True, "cut": True},
            "director": {"max_unique_shots": 16, "target_shot_seconds": 2.8},
        },
    }
    cfg["kidsong"].update(kidsong_over)
    return cfg


# ==================================================== resolve_quality: tiers ===
def test_draft_resolves_to_hires_off_reduced_frames_and_heuristic_review():
    cfg = _base_cfg(quality="draft")
    eff, tier, s = resolve_quality(cfg)

    assert tier == "draft"
    assert s["hires_pass"] is False
    assert s["frames_scale"] == 0.6           # fewer frames per shot
    assert s["max_unique_shots"] == 12         # fewer unique shots
    assert s["reviewer"] == "heuristic"        # no blocking vision waits
    assert s["max_frames"] == 121              # inherited from base config

    # The overrides land where the pipeline actually reads them…
    assert eff["kidsong"]["shot"]["hires_pass"] is False
    assert eff["kidsong"]["shot"]["frames_scale"] == 0.6
    assert eff["kidsong"]["director"]["max_unique_shots"] == 12
    assert eff["kidsong"]["review"]["reviewer"] == "heuristic"
    # …and the workflow selector follows the tier (no _hires graph).
    assert workflow_name(eff) == "ltx23_t2v_toon"
    # …and the i2v selector (keyframe-first/reference-anchor path) agrees.
    assert i2v_workflow_name(eff) == "ltx23_i2v_toon"

    # The caller's config is never mutated (a fresh copy carries the overrides).
    assert eff is not cfg
    assert cfg["kidsong"]["shot"]["hires_pass"] is True
    assert cfg["kidsong"]["director"]["max_unique_shots"] == 16


def test_final_resolves_to_hires_on_full_frames_and_configured_review():
    cfg = _base_cfg(quality="final")
    eff, tier, s = resolve_quality(cfg)

    assert tier == "final"
    assert s["hires_pass"] is True             # forced on
    assert s["frames_scale"] == 1.0            # current (inherited)
    assert s["max_frames"] == 121              # current (inherited)
    assert s["max_unique_shots"] == 16         # current (inherited)
    assert s["reviewer"] == "external"         # keep configured
    assert workflow_name(eff) == "ltx23_t2v_toon_hires"
    assert i2v_workflow_name(eff) == "ltx23_i2v_toon_hires"


# ============================================ REGRESSION: custom == today ======
def test_absent_quality_is_byte_identical_and_returns_the_same_object():
    cfg = _base_cfg()  # no `quality` key at all — the shipped default shape
    before = _read_quality_settings(cfg)
    eff, tier, after = resolve_quality(cfg)

    assert tier == "custom"
    assert eff is cfg                          # SAME object: no copy, no mutation
    assert after == before                     # effective settings unchanged
    assert after == {
        "hires_pass": True, "frames_scale": 1.0, "max_frames": 121,
        "max_unique_shots": 16, "reviewer": "external",
    }


def test_explicit_custom_is_byte_identical_too():
    cfg = _base_cfg(quality="custom")
    eff, tier, after = resolve_quality(cfg)
    assert tier == "custom"
    assert eff is cfg
    assert after == _read_quality_settings(cfg)


def test_unknown_tier_falls_back_to_custom_and_warns(caplog):
    cfg = _base_cfg(quality="ultra")
    with caplog.at_level("WARNING", logger="kidsong.generate"):
        eff, tier, after = resolve_quality(cfg)

    assert tier == "custom"
    assert eff is cfg                          # unchanged — never a silent downgrade
    assert after == _read_quality_settings(cfg)
    assert any("ultra" in r.message for r in caplog.records)
    assert any(r.levelname == "WARNING" for r in caplog.records)


def test_quality_overrides_are_empty_on_the_passthrough_path():
    assert _quality_overrides(_base_cfg())[1] == {}
    assert _quality_overrides(_base_cfg(quality="custom"))[1] == {}
    assert _quality_overrides({})[1] == {}


# ===================================================== config-driven tiers =====
def test_config_quality_tiers_override_the_shipped_defaults():
    cfg = _base_cfg(
        quality="draft",
        quality_tiers={"draft": {"hires_pass": False, "max_unique_shots": 8}},
    )
    _, tier, s = resolve_quality(cfg)
    assert tier == "draft"
    assert s["max_unique_shots"] == 8          # config entry wins
    # keys the config entry omits inherit the code defaults for the tier
    assert s["frames_scale"] == 0.6
    assert s["reviewer"] == "heuristic"


def test_shipped_config_defaults_to_custom_with_both_tiers_registered():
    from pipeline.config import load_config

    cfg = load_config()
    assert cfg["kidsong"]["quality"] == "custom"
    tiers = cfg["kidsong"]["quality_tiers"]
    assert set(tiers) >= {"draft", "final"}
    assert tiers["draft"]["hires_pass"] is False
    assert tiers["final"]["hires_pass"] is True
    # …and because it ships "custom", the shipped config is byte-identical.
    eff, tier, _ = resolve_quality(cfg)
    assert tier == "custom" and eff is cfg


# =============================================== render loop honours the tier ===
def _write_wav(path, seconds=6.0, rate=8000):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(struct.pack("<h", 0) * int(rate * seconds))


def _install_render_fakes(monkeypatch, render_calls, *, guard_sing=True):
    """Patch the whole GPU/IO surface the director render loop touches so a run
    is pure byte-writes into tmp_path. Mirrors the resume/render-style suites."""
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
            return {}          # no baseline NEGATIVE needed for these tests

        def ensure_up(self):
            pass

        def render(self, workflow, patches, out_path):
            render_calls.append((workflow, patches))
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "wb") as f:
                f.write(b"\0" * 1024)

        def free(self):
            pass

    class FakeReviewer:
        def review(self, shot, video_path, out_dir=None):
            return {"accept": True, "score": 1.0, "reasons": [], "retry_hints": {}}

    def fake_assemble(cfg, cut_list, renders, voice, words, duration, out_path, **kw):
        with open(out_path, "wb") as f:
            f.write(b"\0" * 4096)

    monkeypatch.setattr(comfy_mod, "ComfyClient", FakeComfyClient)
    monkeypatch.setattr(review_mod, "get_reviewer", lambda cfg: FakeReviewer())
    monkeypatch.setattr(review_mod, "contact_sheet", lambda *a, **k: None)
    monkeypatch.setattr(review_mod, "review_shots_log", lambda path, entries: path)
    monkeypatch.setattr(captions_mod, "get_word_timestamps", lambda *a, **k: [])
    monkeypatch.setattr(sing_mod, "verse_times_from_words", lambda *a, **k: [(0.0, 6.0)])
    monkeypatch.setattr(edit_mod, "beat_grid", lambda path: [0.0, 2.0, 4.0, 6.0])
    monkeypatch.setattr(edit_mod, "build_cut_list", lambda *a, **k: [])
    monkeypatch.setattr(edit_mod, "assemble", fake_assemble)
    monkeypatch.setattr(cut_qc_mod, "review_cut",
                        lambda *a, **k: {"accept": True, "score": 1.0, "reasons": [], "retry_hints": {}})
    monkeypatch.setattr(edit_mod, "prepend_intro",
                        lambda video_path, cfg, on_progress=None: {"applied": False})
    if guard_sing:
        monkeypatch.setattr(
            sing_mod, "sing_song",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not re-sing")),
        )


@pytest.fixture
def director_run(tmp_path, monkeypatch):
    """A resumable director run with ONE pending shot (a 5s slot, so the frames
    lever is visible). `make(quality)` builds the cfg for a chosen tier."""
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    base = "20260720-101500-kidsong-quality-test"
    os.makedirs(runstate.shots_dir_path(str(out_dir), base), exist_ok=True)

    _write_wav(out_dir / f"{base}.wav")
    song = {
        "title": "Quality Test Song", "description": "d", "tags": ["t"],
        "characters": "Three toddlers: Zuri; Kofi; Nala",
        "verses": [{"scene": "bathroom", "lines": ["a", "b"]}],
    }
    (out_dir / f"{base}-song.json").write_text(json.dumps(song), encoding="utf-8")
    shot = {
        "id": "s00", "verse": 0, "start": 0.0, "end": 5.0,
        "shot_type": "closeup", "characters": ["Kofi"],
        "action": "Kofi brushes his teeth with a big smile",
        "setting": "a bright cheerful bathroom", "camera": "static",
        "reuse_of": None, "seed": 1000, "status": "planned",
    }
    runstate.save_shotlist(str(out_dir), base, {"shots": [shot]})

    render_calls = []
    _install_render_fakes(monkeypatch, render_calls)

    def make(quality):
        ks = {
            "mode": "director", "singer": "ace", "seed": 20260717,
            "render_style": "pixar_toon",
            "shot": {"fps": 24, "max_frames": 121, "hires_pass": True},
            "review": {"script": False, "cut": True, "max_retries_per_shot": 0},
            "intro": {"enabled": False},
        }
        if quality is not None:
            ks["quality"] = quality
        return {
            "_root": _REPO_ROOT,
            "paths": {"output_dir": str(out_dir)},
            "video": {"max_seconds": 60},
            "captions": {"uppercase": True},
            "kidsong": ks,
        }

    return {"make": make, "base": base, "out_dir": str(out_dir),
            "render_calls": render_calls}


def test_draft_lowers_frames_and_skips_the_hires_workflow(director_run):
    run = director_run
    result = generate_kidsong(cfg=run["make"]("draft"), resume_base=run["base"])
    assert result["status"] == "approved"
    assert len(run["render_calls"]) == 1
    workflow, patches = run["render_calls"][0]
    assert workflow == "ltx23_t2v_toon"          # hires pass OFF for a draft
    # 5s slot -> 126 raw frames -> 121 at full; *0.6 -> 73 (snapped to 8n+1).
    assert patches["FRAMES"]["value"] == 73


def test_custom_keeps_full_frames_and_the_hires_workflow(director_run):
    run = director_run
    generate_kidsong(cfg=run["make"](None), resume_base=run["base"])   # no quality key
    workflow, patches = run["render_calls"][0]
    assert workflow == "ltx23_t2v_toon_hires"
    assert patches["FRAMES"]["value"] == 121


def test_the_resolved_tier_is_logged(director_run):
    run = director_run
    generate_kidsong(cfg=run["make"]("draft"), resume_base=run["base"])
    log = open(os.path.join(run["out_dir"], run["base"] + ".log"), encoding="utf-8").read()
    assert "STAGE quality tier=draft" in log
    assert "hires_pass=False" in log
    assert "reviewer=heuristic" in log
    assert "max_unique_shots=12" in log


# ============================================================ finalize path ====
def _completed_draft(out_dir, base="20260720-090000-kidsong-splashy"):
    """Write a finished DRAFT episode to disk: song, a 2-shot rendered ledger
    with takes, the retained working wav, and a promoted Final/ output."""
    shots_dir = runstate.shots_dir_path(str(out_dir), base)
    os.makedirs(shots_dir, exist_ok=True)
    _write_wav(out_dir / f"{base}.wav")                       # draft keeps its wav
    song = {
        "title": "Splashy Bath Time", "description": "d", "tags": ["t"],
        "characters": "Three toddlers: Zuri; Kofi; Nala",
        "verses": [{"scene": "bath", "lines": ["a", "b"]}],
    }
    (out_dir / f"{base}-song.json").write_text(json.dumps(song), encoding="utf-8")

    shots = []
    for i in range(2):
        take = os.path.join(shots_dir, f"s{i:02d}_a0.mp4")
        with open(take, "wb") as f:
            f.write(b"\0" * 1024)
        shot = {
            "id": f"s{i:02d}", "verse": 0, "start": i * 2.5, "end": (i + 1) * 2.5,
            "shot_type": "medium", "characters": ["all"], "action": "splashing",
            "setting": "a bathroom", "camera": "static", "reuse_of": None,
            "seed": 1000 + i, "status": "planned",
        }
        runstate.mark_rendered(shot, take, score=0.9, attempts=1)
        shots.append(shot)
    runstate.save_shotlist(str(out_dir), base, {"shots": shots})

    final_dir = out_dir / "Final"
    final_dir.mkdir(exist_ok=True)
    (final_dir / f"{base}.mp4").write_bytes(b"DRAFTFINAL")
    return base, song


@pytest.fixture
def draft_on_disk(tmp_path):
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    base, song = _completed_draft(out_dir)
    cfg = {
        "_root": _REPO_ROOT,
        "paths": {"output_dir": str(out_dir)},
        "video": {"max_seconds": 60},
        "captions": {"uppercase": True},
        "kidsong": {"mode": "director"},
    }
    return {"out_dir": str(out_dir), "base": base, "song": song, "cfg": cfg}


def test_finalize_reuses_song_shotlist_and_audio_under_a_new_base(draft_on_disk, monkeypatch):
    import pipeline.kidsong.generate as gen

    captured = {"n": 0}

    def fake_generate_kidsong(topic=None, do_upload=False, cfg=None,
                              on_progress=None, resume_base=None, runlog=None):
        captured["n"] += 1
        captured.update(resume_base=resume_base, cfg=cfg, do_upload=do_upload)
        return {"video_path": os.path.join(draft_on_disk["out_dir"], "Final",
                                           resume_base + ".mp4"), "status": "approved"}

    monkeypatch.setattr(gen, "generate_kidsong", fake_generate_kidsong)

    out_dir, base = draft_on_disk["out_dir"], draft_on_disk["base"]
    result = finalize_episode(base, cfg=draft_on_disk["cfg"])

    final_base = base + "-final"
    # Dispatched into the resume path under a NEW base, forcing the final tier.
    assert captured["n"] == 1
    assert captured["resume_base"] == final_base
    assert captured["cfg"]["kidsong"]["quality"] == "final"
    assert result["status"] == "approved"

    # The song, shot list and audio were reused under the new base…
    assert runstate.load_song(out_dir, final_base)["title"] == draft_on_disk["song"]["title"]
    new_shots = runstate.load_shotlist(out_dir, final_base)["shots"]
    assert len(new_shots) == 2
    # …with the ledger reset so the final render redraws every shot.
    assert all(s["status"] == "planned" for s in new_shots)
    assert all("take" not in s and "verdict" not in s for s in new_shots)
    assert os.path.exists(runstate.wav_path(out_dir, final_base))

    # The draft is completely untouched: ledger, takes and Final/ all intact.
    draft_shots = runstate.load_shotlist(out_dir, base)["shots"]
    assert all(s["status"] == "rendered" for s in draft_shots)
    assert all(os.path.exists(s["take"]) for s in draft_shots)
    assert open(os.path.join(out_dir, "Final", base + ".mp4"), "rb").read() == b"DRAFTFINAL"


def test_finalize_reserves_a_new_base_without_colliding(draft_on_disk, monkeypatch):
    import pipeline.kidsong.generate as gen

    out_dir, base = draft_on_disk["out_dir"], draft_on_disk["base"]
    # An earlier finalize already claimed <base>-final.
    runstate.save_shotlist(out_dir, base + "-final", {"shots": [{"id": "keep"}]})

    captured = {}
    monkeypatch.setattr(
        gen, "generate_kidsong",
        lambda **k: captured.update(k) or {"video_path": "x", "status": "approved"},
    )
    finalize_episode(base, cfg=draft_on_disk["cfg"])

    assert captured["resume_base"] == base + "-final-v2"
    # …and the earlier finalize's ledger is left exactly as it was.
    assert runstate.load_shotlist(out_dir, base + "-final")["shots"] == [{"id": "keep"}]


def test_finalize_refuses_when_the_source_audio_is_gone(draft_on_disk):
    out_dir, base = draft_on_disk["out_dir"], draft_on_disk["base"]
    os.remove(runstate.wav_path(out_dir, base))
    with pytest.raises(FileNotFoundError, match="working audio"):
        finalize_episode(base, cfg=draft_on_disk["cfg"])


def test_finalize_refuses_when_the_run_state_is_missing(tmp_path):
    cfg = {"_root": str(tmp_path), "paths": {"output_dir": str(tmp_path)},
           "video": {"max_seconds": 60}, "captions": {}, "kidsong": {"mode": "director"}}
    with pytest.raises(FileNotFoundError, match="Cannot finalize"):
        finalize_episode("no-such-run", cfg=cfg)


def test_reset_shot_clears_the_take_but_keeps_the_plan():
    shot = {
        "id": "s03", "verse": 1, "start": 1.0, "end": 3.0, "shot_type": "wide",
        "characters": ["all"], "action": "clap", "setting": "yard", "camera": "static",
        "reuse_of": None, "seed": 42, "story_subject": "a duck",
        "status": "rendered", "take": "/x/s03_a2.mp4", "verdict": "accepted",
        "score": 0.87, "attempts": 3, "failure_reason": "n/a", "last_verdict": "x",
    }
    fresh = _reset_shot_for_finalize(shot)
    assert fresh["status"] == "planned"
    for gone in ("take", "verdict", "score", "attempts", "failure_reason", "last_verdict"):
        assert gone not in fresh
    for kept in ("id", "verse", "start", "end", "shot_type", "characters",
                 "action", "setting", "camera", "reuse_of", "seed", "story_subject"):
        assert fresh[kept] == shot[kept]
    assert shot["status"] == "rendered"        # source not mutated


def test_reserve_final_base_walks_past_every_existing_artifact(tmp_path):
    out = str(tmp_path)
    assert _reserve_final_base(out, "ep") == "ep-final"
    # A ledger, a shots dir, a wav or a Final/ output each block a base.
    runstate.save_shotlist(out, "ep-final", {"shots": [{"id": "a"}]})
    os.makedirs(runstate.shots_dir_path(out, "ep-final-v2"), exist_ok=True)
    with open(runstate.wav_path(out, "ep-final-v3"), "wb") as f:
        f.write(b"\0")
    assert _reserve_final_base(out, "ep") == "ep-final-v4"


# --------------------------------------------------- finalize, end to end ------
def test_finalize_re_renders_every_shot_at_final_quality_without_re_singing(
    tmp_path, monkeypatch
):
    """The strongest evidence: a real generate_kidsong resume drives the final
    render. Every shot is redrawn on the _hires graph (final forces it, even
    though the base cfg had hires OFF), the draft's ORIGINAL audio is reused (so
    sing_song is never called), and the draft's files are never touched."""
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    base, song = _completed_draft(out_dir)

    render_calls = []
    _install_render_fakes(monkeypatch, render_calls)   # sing_song guarded to raise

    cfg = {
        "_root": _REPO_ROOT,
        "paths": {"output_dir": str(out_dir)},
        "video": {"max_seconds": 60},
        "captions": {"uppercase": True},
        "kidsong": {
            "mode": "director", "singer": "ace", "seed": 20260717,
            "render_style": "pixar_toon",
            # Base cfg has hires OFF — finalize must still force it ON.
            "shot": {"fps": 24, "max_frames": 121, "hires_pass": False},
            "review": {"script": False, "cut": True, "max_retries_per_shot": 0},
            "intro": {"enabled": False},
        },
    }

    draft_final_before = open(os.path.join(str(out_dir), "Final", base + ".mp4"), "rb").read()

    result = finalize_episode(base, cfg=cfg)

    final_base = base + "-final"
    assert result["status"] == "approved"
    # Both unique shots were re-rendered, all on the hires graph.
    assert len(render_calls) == 2
    assert {wf for wf, _ in render_calls} == {"ltx23_t2v_toon_hires"}
    # Final-quality frames at FULL scale: a 2.5s slot -> 65 frames (8n+1). A
    # draft would scale that by 0.6 down to the 49-frame floor, so 65 proves the
    # final tier, not the draft scale, is what governed this render.
    assert all(p["FRAMES"]["value"] == 65 for _, p in render_calls)

    # The finalized episode is its own file; the draft's Final/ is byte-identical.
    assert os.path.basename(result["video_path"]) == final_base + ".mp4"
    assert os.path.exists(result["video_path"])
    assert open(os.path.join(str(out_dir), "Final", base + ".mp4"), "rb").read() == draft_final_before

    # The draft ledger and its takes are untouched by the final re-render.
    draft_shots = runstate.load_shotlist(str(out_dir), base)["shots"]
    assert all(s["status"] == "rendered" for s in draft_shots)
    assert all(os.path.exists(s["take"]) for s in draft_shots)


# ================================================================ CLI wiring ===
def test_finalize_flag_parses():
    args = _build_parser().parse_args(["--finalize", "20260720-090000-kidsong-splashy"])
    assert args.finalize == "20260720-090000-kidsong-splashy"
    assert args.resume is None and args.topic is None


def test_dispatch_routes_finalize_to_finalize_episode(monkeypatch):
    import pipeline.kidsong.generate as gen

    seen = {}
    monkeypatch.setattr(
        gen, "finalize_episode",
        lambda base, **k: seen.update(finalize=base, kw=k) or {"status": "approved"},
    )
    monkeypatch.setattr(gen, "generate_kidsong",
                        lambda *a, **k: seen.update(generate=True))

    args = _build_parser().parse_args(["--finalize", "draft-x", "--upload"])
    _dispatch(args, cfg={"c": 1}, runlog="RL")

    assert seen.get("finalize") == "draft-x"
    assert "generate" not in seen                       # the new-run path is NOT taken
    assert seen["kw"]["do_upload"] is True
    assert seen["kw"]["cfg"] == {"c": 1} and seen["kw"]["runlog"] == "RL"


def test_dispatch_routes_resume_to_generate_kidsong(monkeypatch):
    import pipeline.kidsong.generate as gen

    seen = {}
    monkeypatch.setattr(gen, "finalize_episode",
                        lambda *a, **k: seen.update(finalize=True))

    def fake_generate(topic=None, do_upload=False, **k):
        seen.update(generate=True, topic=topic, resume_base=k.get("resume_base"))
        return {"status": "approved"}

    monkeypatch.setattr(gen, "generate_kidsong", fake_generate)

    args = _build_parser().parse_args(["--resume", "run-1"])
    _dispatch(args, cfg={}, runlog=None)

    assert seen.get("generate") is True and seen["resume_base"] == "run-1"
    assert "finalize" not in seen
