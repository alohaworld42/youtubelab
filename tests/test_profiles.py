"""`studio.profiles` resolves which video_type/language/look a channel
renders under, from `channel["content_profile"]` + `cfg["content_profiles"]`
(deep-merged over the base config exactly like `styles`).

Covers: name lookup, the two-case resolution order (named profile in the
registry vs. derived from the channel row), every hardening case an operator
typo or a hand-edited config can hit (unknown name, non-dict entry, non-dict
overrides, missing block entirely), that `apply_profile` never mutates its
input, and -- the most important test in the file -- that every profile
actually defined in the real `config.example.json` resolves to a real
video_type and a real render_style.
"""
import copy

import pytest

from pipeline.config import load_config
from studio import profiles
from studio.views import VIDEO_TYPE_IDS

RAW_PROFILES = {
    "_comment": "doc key, must be stripped defensively",
    "kidsong_de": {
        "video_type": "kidsong",
        "language": "de",
        "overrides": {"kidsong": {"language": "de", "render_style": "pixar_toon"}},
    },
    "hyperframes_en": {
        "video_type": "hyperframes",
        "language": "en",
        "overrides": {"kidsong": {"language": "en", "render_style": "explainer_clean"}},
    },
    "brainrot_short": {
        "video_type": "brainrot",
        "language": "en",
        "overrides": {"kidsong": {"render_style": "meme_pop"}},
    },
}


def _cfg(content_profiles=None, **extra):
    cfg = {"kidsong": {"language": "en", "render_style": "pixar_toon"}}
    if content_profiles is not None:
        cfg["content_profiles"] = content_profiles
    cfg.update(extra)
    return cfg


# --------------------------------------------------------------- profile_names --
def test_profile_names_lists_keys_in_order_excluding_comment_keys():
    assert profiles.profile_names(_cfg(RAW_PROFILES)) == [
        "kidsong_de",
        "hyperframes_en",
        "brainrot_short",
    ]


def test_profile_names_empty_when_block_missing():
    assert profiles.profile_names(_cfg()) == []
    assert profiles.profile_names({}) == []


@pytest.mark.parametrize("bad", ["not-a-dict", None, ["a", "list"], 5])
def test_profile_names_empty_when_block_is_not_a_dict(bad):
    assert profiles.profile_names(_cfg(bad)) == []


# ------------------------------------------------------------- resolve_profile --
def test_resolve_named_profile_returns_its_own_fields():
    channel = {"content_profile": "kidsong_de", "default_video_type": "kids", "language": "en"}
    resolved = profiles.resolve_profile(_cfg(RAW_PROFILES), channel)

    assert resolved == {
        "name": "kidsong_de",
        "video_type": "kidsong",
        "language": "de",
        "overrides": {"kidsong": {"language": "de", "render_style": "pixar_toon"}},
    }


def test_profile_language_wins_over_channel_language():
    channel = {"content_profile": "hyperframes_en", "language": "de"}
    resolved = profiles.resolve_profile(_cfg(RAW_PROFILES), channel)
    assert resolved["language"] == "en"


def test_channel_language_used_when_profile_names_none():
    profiles_block = {
        "no_lang": {"video_type": "kidsong", "overrides": {}},
    }
    channel = {"content_profile": "no_lang", "language": "de"}
    resolved = profiles.resolve_profile(_cfg(profiles_block), channel)
    assert resolved["language"] == "de"


def test_no_content_profile_set_derives_from_channel():
    channel = {"default_video_type": "kids", "language": "de-DE"}
    resolved = profiles.resolve_profile(_cfg(RAW_PROFILES), channel)

    assert resolved["name"] == ""
    assert resolved["video_type"] == "kids"
    assert resolved["language"] == "de"  # normalized via studio.models.channel_language
    assert resolved["overrides"] == {}


def test_blank_content_profile_derives_from_channel():
    channel = {"content_profile": "", "default_video_type": "kids"}
    resolved = profiles.resolve_profile(_cfg(RAW_PROFILES), channel)
    assert resolved["name"] == ""
    assert resolved["video_type"] == "kids"


def test_none_channel_does_not_raise():
    resolved = profiles.resolve_profile(_cfg(RAW_PROFILES), None)
    assert resolved["name"] == ""
    assert resolved["video_type"] is None
    assert resolved["language"] == "en"


# ---------------------------------------------------------- hardening: typos --
def test_unknown_profile_name_falls_back_and_logs_warning(caplog):
    channel = {"content_profile": "does_not_exist", "default_video_type": "kids", "language": "en"}
    with caplog.at_level("WARNING", logger="studio.profiles"):
        resolved = profiles.resolve_profile(_cfg(RAW_PROFILES), channel)

    assert resolved["name"] == ""
    assert resolved["video_type"] == "kids"
    assert any("does_not_exist" in r.message for r in caplog.records)


@pytest.mark.parametrize("bad_entry", ["a string", None, ["a", "list"], 42])
def test_non_dict_profile_entry_falls_back_and_never_raises(bad_entry, caplog):
    channel = {"content_profile": "broken", "default_video_type": "kids", "language": "en"}
    cfg = _cfg({"broken": bad_entry})
    with caplog.at_level("WARNING", logger="studio.profiles"):
        resolved = profiles.resolve_profile(cfg, channel)  # must not raise
    assert resolved["name"] == ""
    assert resolved["video_type"] == "kids"


@pytest.mark.parametrize("bad_overrides", ["not-a-dict", None, ["x"], 5])
def test_non_dict_overrides_becomes_empty_dict(bad_overrides):
    cfg = _cfg({"weird": {"video_type": "kidsong", "language": "en", "overrides": bad_overrides}})
    channel = {"content_profile": "weird"}
    resolved = profiles.resolve_profile(cfg, channel)
    assert resolved["overrides"] == {}


def test_content_profiles_block_entirely_missing_never_raises():
    channel = {"content_profile": "kidsong_de", "default_video_type": "kids", "language": "en"}
    resolved = profiles.resolve_profile({}, channel)  # no content_profiles at all
    assert resolved["name"] == ""
    assert resolved["video_type"] == "kids"


def test_content_profiles_block_wrong_type_never_raises():
    channel = {"content_profile": "kidsong_de", "default_video_type": "kids"}
    resolved = profiles.resolve_profile(_cfg("not-a-dict"), channel)
    assert resolved["name"] == ""


# --------------------------------------------------------------- apply_profile --
def test_apply_profile_merges_overrides_onto_cfg():
    cfg = _cfg(RAW_PROFILES)
    channel = {"content_profile": "hyperframes_en"}
    merged = profiles.apply_profile(cfg, channel)

    assert merged["kidsong"]["render_style"] == "explainer_clean"
    assert merged["kidsong"]["language"] == "en"


def test_apply_profile_writes_resolved_language_even_without_kidsong_override():
    profiles_block = {"no_kidsong_override": {"video_type": "kidsong", "language": "de", "overrides": {}}}
    cfg = _cfg(profiles_block)
    channel = {"content_profile": "no_kidsong_override", "language": "en"}
    merged = profiles.apply_profile(cfg, channel)
    assert merged["kidsong"]["language"] == "de"


def test_apply_profile_fallback_case_writes_channel_language():
    cfg = _cfg(RAW_PROFILES)
    channel = {"default_video_type": "kids", "language": "de"}
    merged = profiles.apply_profile(cfg, channel)
    assert merged["kidsong"]["language"] == "de"


def test_apply_profile_does_not_mutate_cfg():
    cfg = _cfg(RAW_PROFILES)
    before = copy.deepcopy(cfg)
    channel = {"content_profile": "kidsong_de"}

    profiles.apply_profile(cfg, channel)

    assert cfg == before


def test_apply_profile_none_channel_does_not_raise_or_mutate():
    cfg = _cfg(RAW_PROFILES)
    before = copy.deepcopy(cfg)
    profiles.apply_profile(cfg, None)
    assert cfg == before


# ---------------------------------------------------------- profile_video_type --
def test_profile_video_type_named():
    cfg = _cfg(RAW_PROFILES)
    assert profiles.profile_video_type(cfg, {"content_profile": "brainrot_short"}) == "brainrot"


def test_profile_video_type_derived_fallback():
    cfg = _cfg(RAW_PROFILES)
    channel = {"default_video_type": "reddit"}
    assert profiles.profile_video_type(cfg, channel) == "reddit"


# ------------------------------------------------- real config.example.json --
def test_every_real_content_profile_resolves_to_a_real_video_type_and_style():
    """The most important test in the file: catches a profile pointing at a
    video_type or render_style that doesn't exist. Loads the actual shipped
    config (pipeline.config.load_config(), i.e. config.example.json merged
    with any local config.json) and, for every profile it defines, checks
    the profile (a) resolves without error, (b) names a video_type present
    in studio.views.VIDEO_TYPE_IDS, and (c) if its overrides name a
    kidsong.render_style, that style exists in cfg["kidsong"]["render_styles"].
    """
    cfg = load_config()
    names = profiles.profile_names(cfg)
    assert names, "config.example.json should define at least one content profile"

    render_styles = cfg.get("kidsong", {}).get("render_styles", {})
    assert render_styles, "expected kidsong.render_styles to be non-empty"

    for name in names:
        channel = {"content_profile": name, "default_video_type": "kids", "language": "en"}
        resolved = profiles.resolve_profile(cfg, channel)

        assert resolved["name"] == name, f"profile {name!r} did not resolve by name (typo in fixture?)"
        assert resolved["video_type"] in VIDEO_TYPE_IDS, (
            f"content_profiles.{name}.video_type={resolved['video_type']!r} is not a known "
            f"video type ({sorted(VIDEO_TYPE_IDS)})"
        )

        render_style = resolved["overrides"].get("kidsong", {}).get("render_style")
        if render_style is not None:
            assert render_style in render_styles, (
                f"content_profiles.{name}.overrides.kidsong.render_style={render_style!r} is not a "
                f"known entry in kidsong.render_styles ({sorted(render_styles)})"
            )
