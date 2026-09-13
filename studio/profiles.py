"""Content profiles: which generator a channel runs, in which language, and
with which look.

A channel used to carry only `default_video_type` and a `language` column,
and the global config was the only source of look/style. `content_profiles`
(a top-level block in config.example.json, see its own `_comment`) names a
profile per kind of channel this studio runs -- e.g. `kidsong_de` (German
kids songs, `pixar_toon` render style) vs. `hyperframes_en` (English
explainer videos, `explainer_clean` render style) -- and a channel row picks
one by name in `channels.content_profile`. This module is the ONLY place
that interprets that name; everywhere else keeps treating `content_profile`
as an opaque string (see tests/test_channel_profile.py).

`content_profiles` entries are deep-merged over the base config exactly like
`styles` are (`pipeline.config.merge_overrides` -- no second merge mechanism
is invented here). An entry may also name a `style` (a `cfg["styles"]` preset
name, e.g. "clean" or "brainrot"), applied via `pipeline.config.apply_style`
-- see `apply_profile`'s docstring for why that is a second, separate step
from `overrides`. An unset, empty, or unknown `content_profile` -- an
operator typo, or simply a channel that predates profiles -- falls back to
deriving everything from `channel["default_video_type"]` /
`channel["language"]`, which is the pre-profile behaviour. That fallback
path must never raise: a bad channel row must not be able to kill a
scheduler tick over one broken config line.
"""
import logging

from pipeline.config import merge_overrides

_LOG = logging.getLogger("studio.profiles")


def _resolve_channel_language(channel):
    """The channel's language, preferring `studio.models.channel_language`
    (normalizes tags, defaults to "en") when it is importable, else the raw
    `channel["language"]` value -- so this module works even against a
    half-finished or not-yet-landed `models.py`, same pattern as
    `studio.branding._channel_language`."""
    channel = channel or {}
    try:
        from studio import models
    except ImportError:
        models = None
    fn = getattr(models, "channel_language", None) if models else None
    if callable(fn):
        try:
            return fn(channel)
        except Exception:
            pass
    return channel.get("language")


def _content_profiles(cfg):
    """cfg["content_profiles"] as a dict, or {} for anything else (missing
    block, wrong type, or a raw un-stripped dict handed in directly)."""
    profiles = (cfg or {}).get("content_profiles")
    return profiles if isinstance(profiles, dict) else {}


def profile_names(cfg):
    """The defined profile names in cfg["content_profiles"], in the order
    they appear, excluding "_"-prefixed inline-doc keys.

    `pipeline.config.load_config()` already strips those via
    `_strip_comments` before a caller ever sees the config, but this stays
    defensive in case someone hands in a raw dict straight off `json.load`.
    """
    return [name for name in _content_profiles(cfg) if not str(name).startswith("_")]


def resolve_profile(cfg, channel):
    """Resolve the content profile a channel renders under.

    Resolution order:
      1. `channel["content_profile"]`, if set and present in
         `cfg["content_profiles"]` as a dict -- that profile's own
         `video_type`/`language`/`overrides` are used (an empty/missing
         `overrides` on the entry becomes {}; the profile's `language`
         wins over the channel's language column only when the profile
         actually names one).
      2. Otherwise -- unset, blank, or naming something not in the
         registry, or an entry that isn't a dict -- derived from the
         channel row: name="", video_type=channel["default_video_type"],
         language=channel_language(channel), overrides={}. A name that
         fails to resolve is an operator typo, not a code bug: it is
         logged as a warning and never raises.

    Always returns a dict with the keys "name", "video_type", "language",
    "overrides".
    """
    channel = channel or {}
    name = channel.get("content_profile")

    if name:
        entry = _content_profiles(cfg).get(name)
        if isinstance(entry, dict):
            overrides = entry.get("overrides")
            if not isinstance(overrides, dict):
                overrides = {}
            language = entry.get("language") or _resolve_channel_language(channel)
            video_type = entry.get("video_type") or channel.get("default_video_type")
            return {
                "name": name,
                "video_type": video_type,
                "language": language,
                "overrides": overrides,
            }
        _LOG.warning(
            "channel content_profile %r is not a known entry in content_profiles "
            "(missing, or not an object) -- falling back to the channel's "
            "default_video_type",
            name,
        )

    return {
        "name": "",
        "video_type": channel.get("default_video_type"),
        "language": _resolve_channel_language(channel),
        "overrides": {},
    }


def apply_profile(cfg, channel):
    """`cfg` deep-merged with the channel's resolved profile overrides, with
    the resolved language additionally pinned onto `cfg["kidsong"]["language"]`,
    and (for a named profile whose entry carries a `style` field) that style
    preset from `cfg["styles"]` merged in too.

    Every pipeline site that reads the sung/spoken content language reads it
    from exactly that one key -- `(cfg.get("kidsong") or {}).get("language",
    "en")` in pipeline/kidsong/{lyrics,script_qc,song_audio,generate}.py, and
    pipeline/captions.py takes it in from the kidsong caller as
    `kidsong.language` -- confirmed by grep across pipeline/ at the time this
    was written. There is no second place to write it; if a future pipeline
    change reads language from elsewhere too, extend this function.

    The general (non-kidsong) pipeline path in pipeline/generate.py never
    branches on video_type -- it is the SAME code for hyperframes, brainrot,
    facts, reddit, scary, conversation. Setting only `overrides.kidsong.*`
    (the pre-existing behaviour this module shipped with) therefore did
    nothing for those types: nothing on that path ever reads
    `cfg["kidsong"]["render_style"]` unless the GPU scene tier is on. A
    profile's `style` field names an entry in `cfg["styles"]` -- the exact
    preset mechanism `pipeline.config.apply_style` already applies for the
    `--style` CLI flag and `studio.scheduler`'s per-job/per-channel style --
    and is merged in here so a profile can hand the general path a look
    (captions/camera/grade/assets/...) without a second style-selection path
    being invented. `cfg["content_profiles"][name]["style"]` is read directly
    off the raw entry (not through `resolve_profile`'s return value) so the
    dict `resolve_profile` returns keeps its existing four-key shape --
    tests/test_profiles.py asserts that shape verbatim.

    Never mutates `cfg`: `merge_overrides` (and `apply_style`, itself built on
    `merge_overrides`) always returns a new dict.
    """
    cfg = cfg or {}
    profile = resolve_profile(cfg, channel)
    merged = merge_overrides(cfg, profile["overrides"])
    language = profile.get("language")
    if language:
        merged = merge_overrides(merged, {"kidsong": {"language": language}})
    name = profile.get("name")
    if name:
        entry = _content_profiles(cfg).get(name)
        style_name = entry.get("style") if isinstance(entry, dict) else None
        if style_name:
            from pipeline.config import apply_style

            merged = apply_style(merged, style_name)
    return merged


def profile_video_type(cfg, channel):
    """The video_type a job for this channel should get."""
    return resolve_profile(cfg, channel)["video_type"]
