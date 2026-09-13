"""Effectiveness tests for `studio.profiles` against the REAL config
(`config.example.json` via `pipeline.config.load_config()`).

tests/test_profiles.py already covers `resolve_profile`/`apply_profile`'s
merge mechanics and hardening against typos, using a small fixture. This file
answers a different question: does applying a real profile from
config.example.json actually change the config keys the GENERAL (non-kidsong)
pipeline path reads?

The defect this guards against: `pipeline/generate.py` branches into the
kidsong pipeline ONLY when `video_type == "kidsong"` (~line 106); hyperframes
and brainrot run the exact same general path as facts/reddit/scary/
conversation, which never reads `cfg["kidsong"]["render_style"]` unless the
GPU scene tier is explicitly turned on. A `content_profiles` entry that sets
ONLY `overrides.kidsong.*` for a non-kidsong video_type is therefore a no-op
on that channel's actual render -- exactly what shipped before this fix.

Keys verified here, each backed by a call site (see individual test
docstrings): `voices` (pipeline/tts.py), `whisper.language`
(pipeline/captions.py, read via pipeline/generate.py's own
get_word_timestamps() call), a profile's `style` field (applied through
`pipeline.config.apply_style`, the same mechanism pipeline/generate.py's
--style flag and studio/scheduler.py use), and `kidsong.render_style` +
`video.gpu_scene_mode` + `studio.gpu_scenes.enabled` together (the ComfyUI
background tier's graph selection, studio/gpu_scenes.py, gated by
pipeline/generate.py ~line 194).
"""
import copy

from pipeline.config import load_config
from pipeline.kidsong.render_style import resolve_style
from studio import profiles


def _channel(profile_name, video_type="facts", language="en"):
    return {
        "content_profile": profile_name,
        "default_video_type": video_type,
        "language": language,
    }


# ------------------------------------------------------- voice / transcription --
def test_hyperframes_de_gets_german_voice_and_transcription_language():
    """voices: pipeline/tts.py::_voice_for() reads cfg["voices"][speaker].
    whisper.language: pipeline/captions.py::transcribe_words(), read by
    pipeline/generate.py's own get_word_timestamps(voice_path, cfg, ...) call
    (no explicit `language=` there, so it falls through to whisper.language,
    default "en")."""
    cfg = load_config()
    merged = profiles.apply_profile(cfg, _channel("hyperframes_de"))

    assert merged["whisper"]["language"] == "de"
    assert merged["voices"]["narrator"].startswith("de-")
    assert merged["voices"]["speaker_a"].startswith("de-")
    assert merged["voices"]["speaker_b"].startswith("de-")


def test_hyperframes_en_gets_english_voice_and_transcription_language():
    cfg = load_config()
    merged = profiles.apply_profile(cfg, _channel("hyperframes_en"))

    assert merged["whisper"]["language"] == "en"
    assert merged["voices"]["narrator"].startswith("en-")
    assert merged["voices"]["speaker_a"].startswith("en-")
    assert merged["voices"]["speaker_b"].startswith("en-")


def test_hyperframes_de_and_en_are_provably_different():
    """Not just "both set something" -- the two profiles must actually diverge
    on the keys the general path reads, or a profile that copy-pasted the
    English values under a German name would pass the two tests above and
    still do nothing."""
    cfg = load_config()
    merged_de = profiles.apply_profile(cfg, _channel("hyperframes_de"))
    merged_en = profiles.apply_profile(cfg, _channel("hyperframes_en"))

    assert merged_de["whisper"]["language"] != merged_en["whisper"]["language"]
    assert merged_de["voices"] != merged_en["voices"]


def test_brainrot_short_gets_english_transcription_language():
    cfg = load_config()
    merged = profiles.apply_profile(cfg, _channel("brainrot_short", video_type="brainrot"))
    assert merged["whisper"]["language"] == "en"


# ------------------------------------------------------------------- style --
def test_hyperframes_profile_carries_its_style_preset():
    """A profile's top-level `style` field names a `cfg["styles"]` preset,
    applied by `apply_profile` via `pipeline.config.apply_style` -- the same
    mechanism pipeline/generate.py's --style flag and studio/scheduler.py's
    per-job style use. Checked against values that differ from BOTH the
    pre-profile base config and each other, so this can't pass by accident."""
    cfg = load_config()
    clean_preset = cfg["styles"]["clean"]
    assert cfg["camera"]["intensity"] != clean_preset["camera"]["intensity"]

    merged = profiles.apply_profile(cfg, _channel("hyperframes_en"))
    assert merged["camera"]["intensity"] == clean_preset["camera"]["intensity"]
    assert merged["captions"]["font_size"] == clean_preset["captions"]["font_size"]


def test_brainrot_short_carries_its_style_preset():
    cfg = load_config()
    brainrot_preset = cfg["styles"]["brainrot"]
    assert cfg["assets"]["stock_query_bias"] != brainrot_preset["assets"]["stock_query_bias"]

    merged = profiles.apply_profile(cfg, _channel("brainrot_short", video_type="brainrot"))
    assert merged["assets"]["stock_query_bias"] == brainrot_preset["assets"]["stock_query_bias"]
    assert merged["camera"]["intensity"] == brainrot_preset["camera"]["intensity"]


def test_apply_profile_with_a_style_field_does_not_mutate_cfg():
    """tests/test_profiles.py's mutation guard only exercises profiles without
    a `style` field (its RAW_PROFILES fixture has none) -- the `apply_style`
    call inside `apply_profile` is new code, so it needs its own guard."""
    cfg = load_config()
    before = copy.deepcopy(cfg)
    profiles.apply_profile(cfg, _channel("brainrot_short", video_type="brainrot"))
    assert cfg == before


def test_unknown_style_on_a_profile_entry_never_raises():
    """`apply_style` is already a no-op on an unknown/empty style name
    (tests/test_styles.py); confirm that guarantee survives being called from
    inside `apply_profile` with a hand-broken config."""
    cfg = {
        "content_profiles": {
            "broken_style": {
                "video_type": "hyperframes", "language": "en",
                "style": "does_not_exist_either", "overrides": {},
            }
        },
    }
    merged = profiles.apply_profile(cfg, {"content_profile": "broken_style"})  # must not raise
    assert merged["content_profiles"] == cfg["content_profiles"]


# --------------------------------------------------- ComfyUI background graph --
def test_hyperframes_profile_selects_the_explainer_graph_via_gpu_scene_tier():
    """The one kidsong-namespaced key the general path CAN reach:
    studio/gpu_scenes.py's `_resolve_borrowed_style` reads
    `cfg["kidsong"]["render_style"]` -- but only once BOTH gates it and its
    caller check are open: `video.gpu_scene_mode` (pipeline/generate.py
    ~line 194) and `studio.gpu_scenes.enabled` (studio/gpu_scenes.py::_conf).
    Without both, `kidsong.render_style` is exactly as dead as it was before
    this fix. `pipeline.kidsong.render_style.resolve_style` is the same
    stdlib-only resolver studio/gpu_scenes.py calls, so calling it directly on
    the merged cfg proves the graph a real render would pick."""
    cfg = load_config()
    merged = profiles.apply_profile(cfg, _channel("hyperframes_de"))

    assert merged["video"]["gpu_scene_mode"] is True
    assert merged["studio"]["gpu_scenes"]["enabled"] is True
    assert resolve_style(merged)["workflow_base"] == "ltx23_t2v_explainer"


def test_brainrot_short_selects_the_meme_graph_via_gpu_scene_tier():
    cfg = load_config()
    merged = profiles.apply_profile(cfg, _channel("brainrot_short", video_type="brainrot"))

    assert merged["video"]["gpu_scene_mode"] is True
    assert merged["studio"]["gpu_scenes"]["enabled"] is True
    assert resolve_style(merged)["workflow_base"] == "ltx23_t2v_meme"


def test_unprofiled_channel_keeps_gpu_scene_mode_off():
    """The GPU scene gates are set per-profile, not globally -- a channel
    without a content_profile (or on some other profile) must keep the
    studio-wide default off (config.example.json's own guarantee: "Off =
    byte-identical")."""
    cfg = load_config()
    assert cfg["video"]["gpu_scene_mode"] is False
    merged = profiles.apply_profile(cfg, {"default_video_type": "facts", "language": "en"})
    assert merged["video"]["gpu_scene_mode"] is False


# ------------------------------------------------------- kidsong stays kidsong --
def test_kidsong_profiles_still_drive_the_kidsong_language_key():
    """kidsong_de/kidsong_en are untouched in substance (they run through the
    actual kidsong pipeline, which DOES read cfg["kidsong"]["language"] and
    cfg["kidsong"]["render_style"] directly -- see apply_profile's
    docstring) -- this is a non-regression check, not a new behaviour."""
    cfg = load_config()
    merged_de = profiles.apply_profile(cfg, _channel("kidsong_de", video_type="kidsong"))
    merged_en = profiles.apply_profile(cfg, _channel("kidsong_en", video_type="kidsong"))

    assert merged_de["kidsong"]["language"] == "de"
    assert merged_de["kidsong"]["render_style"] == "pixar_toon"
    assert merged_en["kidsong"]["language"] == "en"


# ------------------------------------------------- the regression itself: -----
# a non-kidsong profile that sets ONLY overrides.kidsong.* is a no-op on the
# general path (pipeline/generate.py never reads cfg["kidsong"].* for it
# outside the opt-in GPU scene tier) -- this must never come back.
def _sets_only_kidsong_overrides(entry):
    overrides = entry.get("overrides") or {}
    override_keys = set(overrides.keys())
    has_style = bool(entry.get("style"))
    return bool(override_keys) and override_keys <= {"kidsong"} and not has_style


def test_defect_pattern_is_detected_by_the_guard_helper():
    """Prove the guard below actually catches the exact shape that shipped
    broken, before trusting it to police the real config."""
    broken = {"video_type": "hyperframes", "language": "de",
              "overrides": {"kidsong": {"language": "de", "render_style": "explainer_clean"}}}
    assert _sets_only_kidsong_overrides(broken)

    fixed_via_style = {"video_type": "hyperframes", "language": "de", "style": "clean",
                        "overrides": {"kidsong": {"language": "de", "render_style": "explainer_clean"}}}
    assert not _sets_only_kidsong_overrides(fixed_via_style)

    fixed_via_other_key = {"video_type": "hyperframes", "language": "de",
                            "overrides": {"kidsong": {"language": "de"}, "whisper": {"language": "de"}}}
    assert not _sets_only_kidsong_overrides(fixed_via_other_key)

    # kidsong itself is exempt: it legitimately only ever needs kidsong.* --
    # the actual kidsong pipeline reads those keys directly.
    kidsong_profile = {"video_type": "kidsong", "language": "de",
                        "overrides": {"kidsong": {"language": "de", "render_style": "pixar_toon"}}}
    assert _sets_only_kidsong_overrides(kidsong_profile)  # true, but exempted below by video_type


def test_no_real_non_kidsong_profile_sets_only_kidsong_overrides():
    """The actual regression guard, run against the shipped config."""
    cfg = load_config()
    raw = cfg.get("content_profiles") or {}
    names = profiles.profile_names(cfg)
    assert names, "config.example.json should define at least one content profile"

    offenders = [
        name for name in names
        if raw[name].get("video_type") != "kidsong" and _sets_only_kidsong_overrides(raw[name])
    ]
    assert offenders == [], (
        f"content_profiles {offenders} set ONLY overrides.kidsong.* for a "
        "non-kidsong video_type -- pipeline/generate.py never routes that "
        "video_type through the kidsong pipeline, so those overrides are "
        "dead. Give the profile a 'style' and/or a general-path override key "
        "(voices, whisper, video, studio, ...)."
    )
