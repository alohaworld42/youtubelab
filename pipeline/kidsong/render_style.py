"""kidsong.render_style — the named render-style registry.

A "render style" is a bundle of the knobs that decide what a kidsong shot
*looks* like: which ComfyUI workflow graph renders it, which style LoRA is
loaded and at what strength, the prompt trigger token, the prose skeleton that
opens every shot prompt, the child-identity clause clamped into it, and the
lighting/set tail that closes it. The single look the pipeline used to hardcode
is now just one entry (``pixar_toon``) in a config map, so a channel can pick a
different look — or A/B a shorter prompt — without touching code.

Everything is config-driven and lives under ``cfg["kidsong"]``::

    render_style     "pixar_toon"      the active style's name
    render_styles    {name -> {...}}   the registry

``resolve_style(cfg)`` returns the active style's dict with every key filled
(missing keys — or a missing/unknown active name — fall back to the
``pixar_toon`` defaults below, so a bad config can never crash a render). It is
the ONE source of truth `generate.py` and the tests share.

This module is stdlib-only (no torch, no requests) on purpose: it is imported
at prompt-build time and by CPU-only tests, and must stay import-cheap.
"""
import logging

log = logging.getLogger("kidsong.render_style")

# The default style's name, and the fallback used whenever the active name is
# missing/unknown or a style entry omits a key. These strings ARE today's
# hardcoded behaviour — keeping them here makes the `pixar_toon` path
# byte-identical to the pre-registry pipeline (the regression guarantee).
DEFAULT_STYLE_NAME = "pixar_toon"

_PIXAR_TOON_DEFAULTS = {
    # Base workflow file (generate.py appends "_hires" when kidsong.shot.hires_pass).
    "workflow_base": "ltx23_t2v_toon",
    # Style LoRA file swapped onto the LORA_STYLE node, and its strength.
    "style_lora": "ltx23_pixar_toon.safetensors",
    "style_strength": 1.0,
    # Trigger token prefixed to every shot prompt ("" = no trigger).
    "style_trigger": "P1x4r",
    # The prose skeleton that opens every shot prompt, after the trigger.
    "prompt_skeleton": "a high-quality 3D CGI toon animation in a preschool show style.",
    # The child-identity clamp sentence kept verbatim in the prompt.
    "identity_clause": (
        "Every child on screen is one of these Black toddlers — no other children appear."
    ),
    # The lighting/set sentences that CLOSE every children-on-screen prompt,
    # mirroring `prompt_skeleton` at the other end. "" = no tail at all.
    #
    # This is the length knob. Measured across the shipped corpus (recorded in
    # docs/quality/SOURCES.md), composed t2v prompts run min 166 / median 184 /
    # p90 231 / max 270 words against the ~200-word cap the RunDiffusion guide
    # recommends; the length is dominated by the three full wardrobe
    # descriptions in the identity sentence — the most load-bearing part —
    # while this fixed 48-word tail is what gets crowded out. Shortening the
    # tail rather than the identity block is therefore the candidate
    # experiment, and it lives here so the A/B is a config flip on the same
    # seeds rather than a code edit. `config.example.json` ships
    # `pixar_toon_concise` as the ready-made B arm.
    #
    # "concrete props" used to sit in this sentence, meaning "specific,
    # tangible" — but this string is not an instruction to an LLM, it is prompt
    # text for the Gemma-3 encoder driving LTX-2, which read it as props MADE OF
    # CONCRETE. Grey stone blocks then appeared as ground-level set dressing in
    # every shot of every episode, playroom and backyard alike. Keep any
    # replacement tail free of unintended material readings.
    "prompt_tail": (
        "A warm key light from one side with a soft cool rim light along hair "
        "and shoulders, gentle soft-edged shadows. The set is lovingly dressed "
        "with two or three simple props that fit the location, kept clear "
        "around the children — bold happy colors, a joyful everyday toddler "
        "moment."
    ),
}

# The keys every resolved style dict carries (order is stable for readability).
STYLE_KEYS = tuple(_PIXAR_TOON_DEFAULTS)


def default_style():
    """A fresh copy of the built-in ``pixar_toon`` defaults (all keys filled)."""
    return dict(_PIXAR_TOON_DEFAULTS)


def resolve_style(cfg):
    """Return the active render style as a dict with every key filled.

    The active name is ``cfg["kidsong"]["render_style"]`` (default
    ``"pixar_toon"``); its entry is read from ``cfg["kidsong"]["render_styles"]``.
    Any key the entry omits — or a wholly missing entry — falls back to the
    ``pixar_toon`` defaults. An active name that names a style NOT in the
    registry logs a warning and falls back to ``pixar_toon``, so an unknown
    style can never crash a render. The returned dict also carries ``name``.
    """
    ks = (cfg or {}).get("kidsong", {}) if isinstance(cfg, dict) else {}
    ks = ks or {}
    styles = ks.get("render_styles") or {}
    name = ks.get("render_style") or DEFAULT_STYLE_NAME

    entry = styles.get(name)
    if entry is None and name != DEFAULT_STYLE_NAME:
        log.warning(
            "kidsong.render_style=%r is not defined in kidsong.render_styles "
            "(available: %s) — falling back to %r.",
            name, ", ".join(sorted(styles)) or "none", DEFAULT_STYLE_NAME,
        )
        name = DEFAULT_STYLE_NAME
        entry = styles.get(name)

    if not isinstance(entry, dict):
        entry = {}

    resolved = {}
    for key, default in _PIXAR_TOON_DEFAULTS.items():
        value = entry.get(key)
        # `is None` (not falsiness): style_strength 0.0 and style_trigger ""
        # are meaningful values a style may set on purpose, not "unset".
        resolved[key] = default if value is None else value
    resolved["name"] = name
    return resolved


def _hires_enabled(cfg):
    """The one source of truth for ``kidsong.shot.hires_pass`` (default True)."""
    ks = (cfg or {}).get("kidsong", {}) if isinstance(cfg, dict) else {}
    shot_cfg = (ks or {}).get("shot", {}) or {}
    return bool(shot_cfg.get("hires_pass", True))


def workflow_name(cfg, style=None):
    """The workflow graph name for the active style, ``_hires`` when configured.

    ``kidsong.shot.hires_pass`` (default True) decides whether the 2x refine
    graph is used; this appends ``"_hires"`` to the active style's
    ``workflow_base`` exactly as the pre-registry code did for the single
    hardcoded look. Pass an already-resolved ``style`` to avoid re-resolving.
    """
    if style is None:
        style = resolve_style(cfg)
    base = style["workflow_base"]
    return f"{base}_hires" if _hires_enabled(cfg) else base


# The i2v graph every style shares today. Not a style-registry key on purpose:
# a per-style i2v graph does not exist, and a registry entry would imply it.
I2V_WORKFLOW_BASE = "ltx23_i2v_toon"


def i2v_workflow_name(cfg):
    """The image-to-video graph name, ``_hires`` when hires_pass is on.

    Mirrors ``workflow_name`` so the keyframe-first/reference-anchored path
    gets the same 2x refine the t2v path does — before this existed, every
    anchored shot silently skipped the hires pass the config asked for.
    """
    return f"{I2V_WORKFLOW_BASE}_hires" if _hires_enabled(cfg) else I2V_WORKFLOW_BASE
