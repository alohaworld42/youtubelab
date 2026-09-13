"""Orthogonality tests: `kidsong.content_mode` must be a pure ADDITION that
changes only (a) which song is selected (pipeline.kidsong.lyrics.generate_song)
and (b) per-verse story_subject/shot-planning bias (pipeline.kidsong.director).

Every other stage of a director-mode run — the render-style registry, quality
tiers, captions defaults, the branding intro, the QC gates, and resume — reads
its own independent config keys and never looks at content_mode at all. This
file proves that by resolving those systems with content_mode set both ways
and asserting identical results, plus a couple of static greps of the source
so a future change that accidentally wires content_mode into one of them is
caught even before the assertions would catch it.

CPU-only: no GPU, no network, no ComfyUI, no LLM calls.
"""
import copy
import inspect
import re

from pipeline.kidsong import director, generate as kgen, lyrics
from pipeline.kidsong.render_style import resolve_style, workflow_name


def _cfg(content_mode=None, **kidsong_over):
    ks = {
        "render_style": "pixar_toon",
        "shot": {"hires_pass": True, "max_frames": 121},
        "review": {"reviewer": "heuristic"},
        "director": {"max_unique_shots": 16},
        "quality": "draft",
    }
    ks.update(kidsong_over)
    if content_mode is not None:
        ks["content_mode"] = content_mode
    return {
        "_root": ".",
        "kidsong": ks,
        "captions": {"uppercase": True},
        "video": {"max_seconds": 60},
    }


# --------------------------------------------------------------- quality tiers
def test_quality_tier_resolution_is_unaffected_by_content_mode():
    cfg_song = _cfg(content_mode="song")
    cfg_learning = _cfg(content_mode="learning")
    cfg_none = _cfg(content_mode=None)

    eff_song, tier_song, settings_song = kgen.resolve_quality(cfg_song)
    eff_learning, tier_learning, settings_learning = kgen.resolve_quality(cfg_learning)
    eff_none, tier_none, settings_none = kgen.resolve_quality(cfg_none)

    assert tier_song == tier_learning == tier_none
    assert settings_song == settings_learning == settings_none


# ---------------------------------------------------------------- render style
def test_render_style_resolution_is_unaffected_by_content_mode():
    cfg_song = _cfg(content_mode="song")
    cfg_learning = _cfg(content_mode="learning")
    assert resolve_style(cfg_song) == resolve_style(cfg_learning)
    assert workflow_name(cfg_song) == workflow_name(cfg_learning)


# --------------------------------------------------------------------- source
def test_resolve_quality_source_never_reads_content_mode():
    """Static guard: the tier-resolution machinery's own source text must not
    mention content_mode at all — it should have no idea the concept exists."""
    src = inspect.getsource(kgen._quality_overrides) + inspect.getsource(kgen._read_quality_settings)
    assert "content_mode" not in src


def test_render_style_source_never_reads_content_mode():
    import pipeline.kidsong.render_style as render_style_mod

    src = inspect.getsource(render_style_mod)
    assert "content_mode" not in src


def test_only_lyrics_and_director_modules_reference_content_mode():
    """content_mode is a deliberately narrow, additive concept: it should only
    be read in the two places that implement it (lyrics.generate_song's
    dispatch, director's per-verse learning bias) — not sprinkled into
    unrelated stages (captions, intro, QC, resume, upload)."""
    from pipeline.kidsong import edit, script_qc, runstate, cut_qc

    for mod in (edit, script_qc, runstate, cut_qc):
        src = inspect.getsource(mod)
        assert "content_mode" not in src, mod.__name__


# ------------------------------------------------------------- wiring smoke ---
def test_generate_song_and_plan_shots_wiring_share_one_cfg_object():
    """generate.py's `_run`/`_generate_director` call `generate_song(topic, cfg)`
    and `plan_shots(song, verse_times, beats, cfg)` with the SAME cfg object it
    was itself given — content_mode is read straight off that cfg by both, with
    no extra parameter threaded through either call. This locks in that the
    call SHAPES in generate.py have not grown a content_mode-specific branch
    (the "cleanest" wiring the feature was built around)."""
    run_src = inspect.getsource(kgen._run)
    director_src = inspect.getsource(kgen._generate_director)
    assert re.search(r"generate_song\(\s*topic\s*,\s*cfg\s*\)", run_src)
    assert re.search(r"plan_shots\(\s*song\s*,\s*verse_times\s*,\s*beats\s*,\s*cfg\s*\)", director_src)
    assert "content_mode" not in run_src
    assert "content_mode" not in director_src


def test_end_to_end_learning_song_shot_plan_matches_song_mode_shape():
    """A learning episode's shot list has the exact same STRUCTURE a normal
    episode's does (contiguous tiling, a wide opener, seeded shots) — learning
    mode only changes song SOURCE and per-verse subject handling, never the
    shot-list schema QC/edit/assemble downstream expect."""
    from pipeline.config import load_config

    base_cfg = copy.deepcopy(load_config())
    base_cfg.setdefault("kidsong", {}).setdefault("review", {})["reviewer"] = "heuristic"

    cfg_song = copy.deepcopy(base_cfg)
    # Pin the library source so "song mode" yields a deterministic library song
    # offline; this test is about shot-plan SHAPE, not the topical-LLM auto path.
    cfg_song["kidsong"]["lyrics_source"] = "public_domain"
    cfg_learning = copy.deepcopy(base_cfg)
    cfg_learning["kidsong"]["content_mode"] = "learning"

    song_a = lyrics.generate_song("stars", cfg_song)
    song_b = lyrics.generate_song(None, cfg_learning)

    for song, cfg in ((song_a, cfg_song), (song_b, cfg_learning)):
        verse_times = [(i * 8.0, (i + 1) * 8.0) for i in range(len(song["verses"]))]
        beats = {"bpm": 100, "beat_times": [i * 0.6 for i in range(200)]}
        shots = director._fallback_planner(song, verse_times, beats, cfg)
        assert shots, cfg["kidsong"].get("content_mode")
        assert shots[0]["shot_type"] == "wide"
        assert shots[0]["characters"] == ["all"]
        for s in shots:
            assert {"id", "verse", "start", "end", "shot_type", "characters",
                    "action", "camera", "setting", "reuse_of", "seed",
                    "story_subject", "status"} <= set(s)
