"""kidsong.refs — canonical per-character REFERENCE IMAGES for pixel-anchored identity.

The channel's worst recurring defect is character DRIFT: a fourth, off-model
child appears (lighter skin, straight hair), or the correct child is cloned.
Two GPU A/B rounds proved prompt-level control has hit its ceiling for IDENTITY
— describing a child in text only gets you so far. The next lever is PIXEL
ANCHORING: render one canonical still per cast member with the locally-installed
Z-Image Turbo, then feed THAT image into the LTX i2v graph as frame-0
conditioning so a rejected shot is re-drawn from the reference instead of from a
text description of it.

This module owns the reference lifecycle:

    reference_path(cfg, id)          where a character's canonical PNG lives
    reference_for(cfg, id)           the persisted reference (bible field or
                                     default location), or None if absent
    harvest_reference(cfg, id)       FALLBACK: pull a clean single-subject frame
                                     from an already-accepted take (free, no GPU)
    generate_reference(client, ...)  the ONLY GPU-touching function: render a
                                     fresh Z-Image still from the cast-bible
                                     description
    ensure_reference(client, ...)    harvest first, else generate; idempotent

Stdlib + existing repo imports only. Everything heavy — the ComfyUI HTTP client
(``requests``) and moviepy/ffmpeg (via ``transitions``) — is imported lazily
inside the functions that need it, so ``import pipeline.kidsong.refs`` stays
cheap, torch-free and safe to run in tests with no GPU and no network.
"""
import json
import logging
import os
import re

log = logging.getLogger("kidsong.refs")

# The Z-Image Turbo text->image graph (workflows/zimage_ref.json). Node titles
# PROMPT/SEED/WIDTH/HEIGHT/FILENAME_PREFIX are patched by ComfyClient._apply_patches.
_ZIMAGE_WORKFLOW = "zimage_ref"
# The identity-anchored variant: same graph + three LoadImage nodes wired into
# TextEncodeZImageOmni's image1/2/3 (and its vae input) so cast reference PNGs
# condition the still at pixel level. Used by `generate_still` when the caller
# passes `ref_images`.
_ZIMAGE_REF_WORKFLOW = "zimage_still_ref"
# The WORKING identity-ref graph: FLUX.2 klein 4B with a 3-deep ReferenceLatent
# chain (workflows/flux2_klein_ref.json). The zimage_still_ref graph above is
# retired for identity work — Tongyi never shipped the Omni weights its image
# inputs target (IMP-012) — but kept on disk for the day they do.
_KLEIN_REF_WORKFLOW = "flux2_klein_ref"

# output/_cast_refs/<char_id>/canonical.png
_REF_DIRNAME = "_cast_refs"
_REF_BASENAME = "canonical.png"

# output/_cast_refs/<char_id>/style.json — the style-stamp sidecar written
# beside every canonical.png this module produces (generate_reference AND
# harvest_reference), so a later render-style switch can tell a stale
# reference apart from a fresh one. See `_write_style_sidecar`/
# `_style_staleness` and `ensure_reference`, below.
_STYLE_SIDECAR_BASENAME = "style.json"

# output/_learning_cards/<kind>_<label>.png — see generate_card, below.
_LEARNING_CARD_DIRNAME = "_learning_cards"

# Z-Image Turbo renders best around its native ~1MP; a standing single-subject
# reference wants portrait framing. Both must be multiples of 16 (the
# EmptySD3LatentImage step in the graph). Overridable via kidsong.reference.
_DEFAULT_REF_WIDTH = 768
_DEFAULT_REF_HEIGHT = 1024


# ------------------------------------------------------------------ paths ---
def _output_dir(cfg):
    """Absolute ``output/`` dir for this config, tolerant of ad-hoc test cfgs."""
    try:
        from pipeline.config import abspath

        return abspath(cfg, cfg["paths"]["output_dir"])
    except Exception:
        root = cfg.get("_root", ".") if isinstance(cfg, dict) else "."
        return os.path.join(str(root), "output")


def _abs(cfg, path):
    if os.path.isabs(path):
        return path
    try:
        from pipeline.config import abspath

        return abspath(cfg, path)
    except Exception:
        root = cfg.get("_root", ".") if isinstance(cfg, dict) else "."
        return os.path.join(str(root), path)


def reference_path(cfg, char_id):
    """Canonical on-disk location of a character's reference PNG.

    ``output/_cast_refs/<char_id>/canonical.png``. This is where
    ``generate_reference`` / ``harvest_reference`` write, and the default
    location ``reference_for`` falls back to when the bible records no explicit
    ``reference_image`` path.
    """
    return os.path.join(_output_dir(cfg), _REF_DIRNAME, str(char_id).strip().lower(),
                        _REF_BASENAME)


# ------------------------------------------------------------- persisted ---
def _pinned_reference_for(cfg, char_id):
    """The cast bible's hand-chosen ``reference_image`` for ``char_id``, if
    recorded AND present on disk — ``reference_for``'s first candidate, in
    isolation, so ``ensure_reference`` can treat it specially (never
    auto-regenerated; see below). Never raises."""
    try:
        from pipeline.kidsong import cast

        recorded = cast.reference_for(char_id)
    except Exception:
        recorded = None
    if not recorded:
        return None
    path = _abs(cfg, recorded)
    try:
        if path and os.path.isfile(path) and os.path.getsize(path) > 0:
            return path
    except OSError:
        pass
    return None


def reference_for(cfg, char_id):
    """The persisted canonical reference PNG for ``char_id``, or ``None``.

    Resolution order:
      1. the path recorded in the cast bible's optional per-character
         ``reference_image`` field (resolved through cast alias handling), if
         that file exists;
      2. the default location (``reference_path``), if that file exists;
      3. otherwise ``None``.

    Only ever returns a path whose file is actually present and non-empty, so a
    caller can treat a non-None result as "a reference is ready to anchor on".
    Never raises — a lookup problem must never block a render.
    """
    pinned = _pinned_reference_for(cfg, char_id)
    if pinned:
        return pinned

    default_path = reference_path(cfg, char_id)
    try:
        if default_path and os.path.isfile(default_path) and os.path.getsize(default_path) > 0:
            return default_path
    except OSError:
        pass
    return None


# ---------------------------------------------------------- style sidecar ---
def _style_sidecar_path(ref_path):
    """Where the style-stamp sidecar for a reference PNG at ``ref_path``
    lives: beside it, as ``style.json``."""
    return os.path.join(os.path.dirname(ref_path), _STYLE_SIDECAR_BASENAME)


def _active_style_name(cfg):
    """The active render style's name, resolved the same way the rest of this
    module resolves style (``render_style.resolve_style``). Never raises —
    falls back to ``None`` so a lookup problem can never crash a caller."""
    try:
        from pipeline.kidsong.render_style import resolve_style

        return resolve_style(cfg).get("name")
    except Exception:
        return None


def _write_style_sidecar(cfg, ref_path):
    """Best-effort: stamp the reference at ``ref_path`` with the render style
    active when it was produced, so a later style switch can tell a stale
    reference apart from a fresh one (see ``_style_staleness`` /
    ``ensure_reference``). Failure only logs a warning — a sidecar write must
    never raise and block an otherwise-successful reference render."""
    try:
        with open(_style_sidecar_path(ref_path), "w", encoding="utf-8") as f:
            json.dump({"render_style": _active_style_name(cfg)}, f)
    except Exception as exc:
        log.warning("refs: could not write style sidecar for %s (%s)", ref_path, exc)


def _read_style_sidecar(ref_path):
    """Best-effort parse of the style sidecar beside ``ref_path``. Returns the
    parsed dict, or ``None`` when the sidecar is missing, unreadable, or not a
    JSON object. Never raises."""
    try:
        with open(_style_sidecar_path(ref_path), "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _style_staleness(cfg, ref_path):
    """``(is_stale, recorded_style_name)`` for the reference at ``ref_path``
    against the currently active render style. A reference is STALE when its
    sidecar is missing, unreadable/corrupt, or names a style other than the
    one active now — the three cases ``ensure_reference`` must treat as "this
    reference no longer matches the channel's current look"."""
    sidecar = _read_style_sidecar(ref_path)
    if sidecar is None:
        return True, None
    recorded = sidecar.get("render_style")
    return recorded != _active_style_name(cfg), recorded


# ------------------------------------------------------------- generate ---
def _reference_prompt(cfg, char_id):
    """The Z-Image text prompt describing ONE cast child on a neutral set.

    Built from the cast-bible appearance (``cast.describe``) plus the active
    render style's prose skeleton so the still matches the channel's look. The
    LTX LoRA trigger token (``style_trigger``, e.g. "P1x4r") is DELIBERATELY
    omitted: it activates an LTX-specific LoRA and is meaningless to Z-Image,
    which is a separate Tongyi model driven by a Qwen text encoder — feeding it
    would only inject noise.
    """
    from pipeline.kidsong import cast
    from pipeline.kidsong.render_style import resolve_style

    desc = cast.describe(char_id) or str(char_id)
    skeleton = str(resolve_style(cfg).get("prompt_skeleton") or "").rstrip(".")

    parts = [
        skeleton,
        f"a single full-body character reference of {desc}",
        "standing and facing the camera with a warm friendly smile",
        "the whole body visible from head to shoes, centred in frame",
        "on a plain neutral light-grey studio background with soft even lighting",
        "one child only, no other people anywhere in the frame, no text",
        "bold happy colours, soft rounded shapes, big expressive eyes",
    ]
    return ", ".join(p for p in parts if p) + "."


def _reference_dims(cfg):
    """``(width, height)`` for a reference render, snapped to a multiple of 16.

    Defaults to portrait 768x1024 (good for a standing toddler); override with
    ``kidsong.reference.width`` / ``.height``.
    """
    ref = ((cfg.get("kidsong", {}) or {}).get("reference", {}) or {}) if isinstance(cfg, dict) else {}
    width = int(ref.get("width", _DEFAULT_REF_WIDTH))
    height = int(ref.get("height", _DEFAULT_REF_HEIGHT))

    def snap(value):
        return max(16, (int(value) // 16) * 16)

    return snap(width), snap(height)


def _reference_seed(cfg, char_id):
    """Deterministic per-character seed so a regenerated reference is stable.

    Reuses the cast's own seed family (``cast.seed_for``) keyed on the base
    ``kidsong.seed``, so the same child always renders from the same seed.
    """
    base = int(((cfg.get("kidsong", {}) or {}) if isinstance(cfg, dict) else {}).get("seed", 20260717))
    try:
        from pipeline.kidsong import cast

        return int(cast.seed_for([char_id], base))
    except Exception:
        return base


def generate_reference(client, cfg, char_id):
    """Render a fresh canonical reference PNG for ``char_id`` with Z-Image Turbo.

    THE ONLY GPU-TOUCHING FUNCTION in this module: it POSTs the ``zimage_ref``
    graph to ComfyUI. Builds the prompt from the cast-bible description, patches
    PROMPT/SEED/WIDTH/HEIGHT/FILENAME_PREFIX, and writes the still to
    ``reference_path(cfg, char_id)``. Returns that path.
    """
    dest = reference_path(cfg, char_id)
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    width, height = _reference_dims(cfg)
    patches = {
        "PROMPT": {"prompt": _reference_prompt(cfg, char_id)},
        "SEED": {"value": _reference_seed(cfg, char_id)},
        "WIDTH": {"value": int(width)},
        "HEIGHT": {"value": int(height)},
        "FILENAME_PREFIX": {"filename_prefix": f"{_REF_DIRNAME}/{str(char_id).strip().lower()}"},
    }
    log.info("refs: generating Z-Image reference for %s (%dx%d) -> %s",
             char_id, width, height, dest)
    client.render(_ZIMAGE_WORKFLOW, patches, dest)
    _write_style_sidecar(cfg, dest)
    return dest


# -------------------------------------------------------------- harvest ---
def _iter_shot_ledgers(out_dir):
    """Yield ``(base, shotlist)`` for every run ledger in ``out_dir``, newest first."""
    from pipeline.kidsong import runstate

    try:
        names = os.listdir(out_dir)
    except OSError:
        return
    ledgers = []
    for name in names:
        if not name.endswith(runstate.SHOTS_SUFFIX):
            continue
        full = os.path.join(out_dir, name)
        try:
            mtime = os.path.getmtime(full)
        except OSError:
            mtime = 0.0
        ledgers.append((mtime, name))
    for _, name in sorted(ledgers, reverse=True):
        base = name[: -len(runstate.SHOTS_SUFFIX)]
        shotlist = runstate.load_shotlist(out_dir, base)
        if isinstance(shotlist, dict) and shotlist.get("shots"):
            yield base, shotlist


def _find_accepted_single_subject_take(cfg, char_id):
    """Path of an ACCEPTED single-subject take whose one child is ``char_id``.

    Scans run ledgers newest-first; a candidate must resolve (via the same cast
    resolver the renderer uses, so aliases/substitutions match) to exactly ONE
    cast child equal to ``char_id`` and have a QC-accepted take on disk. Returns
    the take path, or ``None`` when no clean footage exists yet.
    """
    from pipeline.kidsong import cast, runstate

    target = str(char_id).strip().lower()
    out_dir = _output_dir(cfg)
    for base, shotlist in _iter_shot_ledgers(out_dir):
        shots_dir = runstate.shots_dir_path(out_dir, base)
        for shot in shotlist["shots"]:
            if shot.get("reuse_of"):
                continue
            try:
                resolved = cast._resolve_with_fallback(
                    shot.get("characters"), shot_type=shot.get("shot_type")
                )
            except Exception:
                continue
            if len(resolved) != 1 or str(resolved[0].get("id")).strip().lower() != target:
                continue
            path, verdict, _score = runstate.best_take(shots_dir, shot.get("id", ""))
            if path and verdict == runstate.VERDICT_ACCEPTED:
                return path
    return None


def harvest_reference(cfg, char_id):
    """FALLBACK reference: a clean frame lifted from already-accepted footage.

    For a character that already has good single-subject footage, this is free
    (no GPU): find an accepted single-subject take of that child and extract a
    frame into the reference path. Returns the written path, or ``None`` when no
    suitable take exists (so ``ensure_reference`` falls back to generation).
    """
    src = _find_accepted_single_subject_take(cfg, char_id)
    if not src:
        return None

    from pipeline.kidsong import transitions

    dest = reference_path(cfg, char_id)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    try:
        # "first" frame: a t2v single-subject take is a fresh draw at frame 0,
        # cheapest to grab (no ffprobe) and free of end-of-clip motion blur.
        transitions.extract_frame(src, "first", dest)
    except Exception as exc:
        log.warning("refs: could not harvest a reference frame from %s for %s (%s)",
                    src, char_id, exc)
        return None
    if os.path.isfile(dest) and os.path.getsize(dest) > 0:
        log.info("refs: harvested reference for %s from accepted take %s -> %s",
                 char_id, src, dest)
        _write_style_sidecar(cfg, dest)
        return dest
    return None


# --------------------------------------------------------------- ensure ---
def ensure_reference(client, cfg, char_id):
    """Return a ready reference path for ``char_id``, creating one if needed.

    Idempotent and cheap-first, with a style-freshness check on the default
    location so a render-style switch (``kidsong.render_style``) can never
    leave an old-style reference silently anchoring new-style renders:
      1. a bible-PINNED ``reference_image`` (hand-chosen curation) is always
         returned untouched, no matter its style-sidecar state — only a
         WARNING is logged on a style mismatch, since this path is a
         deliberate human choice this module must never override;
      2. else an existing DEFAULT-path ``canonical.png`` is returned untouched
         IF its style sidecar matches the active style;
      3. else — no default reference yet, OR it is STALE (sidecar missing,
         unreadable/corrupt, or naming a different style) — HARVEST a frame
         from accepted footage (free, no GPU). Harvesting is SKIPPED for a
         stale reference: harvested frames come from the same old-style
         episodes as the stale PNG and would just reintroduce the stale look;
      4. else GENERATE a fresh one with Z-Image Turbo (needs ``client``),
         which also (re)writes the style sidecar (see ``generate_reference``).

    Returns the path, or ``None`` if there is nothing to harvest/generate (or
    generation failed). Never raises.
    """
    pinned = _pinned_reference_for(cfg, char_id)
    if pinned:
        stale, recorded = _style_staleness(cfg, pinned)
        if stale:
            log.warning(
                "refs: bible-pinned reference for %s was generated under render "
                "style %r; the active style is %r — using it anyway (bible-pinned "
                "references are hand-chosen and are never auto-regenerated).",
                char_id, recorded, _active_style_name(cfg),
            )
        return pinned

    default_path = reference_path(cfg, char_id)
    default_exists = False
    try:
        default_exists = os.path.isfile(default_path) and os.path.getsize(default_path) > 0
    except OSError:
        default_exists = False

    if default_exists:
        stale, recorded = _style_staleness(cfg, default_path)
        if not stale:
            return default_path
        log.info(
            "refs: style changed %s -> %s, regenerating reference for %s "
            "(skipping harvest — old-style footage would reintroduce the stale look).",
            recorded, _active_style_name(cfg), char_id,
        )
    else:
        harvested = harvest_reference(cfg, char_id)
        if harvested:
            return harvested

    if client is None:
        return None
    try:
        return generate_reference(client, cfg, char_id)
    except Exception as exc:
        log.warning("refs: could not generate a reference for %s (%s)", char_id, exc)
        return None


# ------------------------------------------------------- learning cards ---
# Video diffusion (LTX) renders on-screen LETTERS and NUMBERS poorly — legible
# text is exactly the thing t2v models are worst at. A Z-Image Turbo STILL of
# the taught item (a clean, bold, single-subject graphic) is the reliable path
# to something a toddler can actually read. This is the card-generator half of
# that: kidsong.director's learning-mode insert shots (the item "shown large
# and clear") give the flexible in-scene version; a generated card is the
# crisp fallback/asset a caller can composite in directly. Wiring a card into
# the video timeline is intentionally OUT OF SCOPE here — this only produces
# the tested, GPU-touching-only-via-`client` asset, exactly like
# `generate_reference` above.
_DEFAULT_CARD_WIDTH = 1024
_DEFAULT_CARD_HEIGHT = 1024

# Per-`kind` phrasing for the card's subject clause. Any `kind` not listed here
# still renders (falls back to a generic "a big bold colorful <label>"), so an
# unexpected kind can never raise.
_CARD_KIND_PHRASES = {
    "letter": "a big bold colorful capital letter {label}",
    "number": "a big bold colorful number {label}",
    "shape": "a big bold colorful {label} shape",
}


def _slug(text):
    """Filesystem/prompt-safe slug: 'A' -> 'a', '3' -> '3', 'red circle' ->
    'red-circle'. Never empty (falls back to 'item') so a blank label can
    never collide with another card's path."""
    slug = re.sub(r"[^a-z0-9]+", "-", str(text or "").strip().lower()).strip("-")
    return slug or "item"


def card_path(cfg, text_label, kind):
    """Default on-disk location for one learning card:
    ``output/_learning_cards/<kind>_<label>.png``."""
    return os.path.join(
        _output_dir(cfg), _LEARNING_CARD_DIRNAME,
        f"{_slug(kind)}_{_slug(text_label)}.png",
    )


def _card_subject(text_label, kind):
    """The card's subject clause, e.g. kind='letter', text_label='A' ->
    'a big bold colorful capital letter A'."""
    label = str(text_label or "").strip()
    phrase = _CARD_KIND_PHRASES.get(str(kind or "").strip().lower(), "a big bold colorful {label}")
    return phrase.format(label=label)


def _card_prompt(cfg, text_label, kind):
    """The Z-Image text prompt for one learning card: the active render
    style's prose skeleton (so the card's palette/finish matches the channel's
    look), then the item itself, centered, on a plain background, with
    nothing else in frame — no people, no clutter, nothing to compete with
    the one thing a toddler needs to read.
    """
    from pipeline.kidsong.render_style import resolve_style

    skeleton = str(resolve_style(cfg).get("prompt_skeleton") or "").rstrip(".")
    subject = _card_subject(text_label, kind)
    parts = [
        skeleton,
        f"{subject}, centered, clean flat vector, plain solid background, no other text",
        "bold happy colors, soft rounded shapes",
    ]
    return ", ".join(p for p in parts if p) + "."


def _card_dims(cfg):
    """``(width, height)`` for a card render, snapped to a multiple of 16
    (the graph's EmptySD3LatentImage step). Defaults to a square 1024x1024 —
    good for a centered single-subject graphic; override with
    ``kidsong.learning_card.width`` / ``.height``."""
    card_cfg = ((cfg.get("kidsong", {}) or {}).get("learning_card", {}) or {}) if isinstance(cfg, dict) else {}
    width = int(card_cfg.get("width", _DEFAULT_CARD_WIDTH))
    height = int(card_cfg.get("height", _DEFAULT_CARD_HEIGHT))

    def snap(value):
        return max(16, (int(value) // 16) * 16)

    return snap(width), snap(height)


def _card_seed(cfg, text_label, kind):
    """Deterministic per-(kind, label) seed, so the same card is stable across
    regenerations without depending on the cast bible (a letter/number/shape
    is not a cast member). Small stable hash — NOT Python's randomized
    `hash()` — offset from the base `kidsong.seed`."""
    base = int(((cfg.get("kidsong", {}) or {}) if isinstance(cfg, dict) else {}).get("seed", 20260717))
    key = f"{_slug(kind)}:{_slug(text_label)}"
    h = 0
    for ch in key:
        h = (h * 31 + ord(ch)) % 100000
    return base + h


def generate_card(client, cfg, text_label, kind, dest=None):
    """Render a fresh Z-Image Turbo card for one taught item.

    A thin generalization of `generate_reference`: same `zimage_ref` workflow,
    same PROMPT/SEED/WIDTH/HEIGHT/FILENAME_PREFIX patch shape, same "the ONLY
    GPU touch is `client.render`" contract. `kind` is "letter" | "number" |
    "shape" (see `_CARD_KIND_PHRASES`; anything else still renders, with a
    generic subject phrase). Writes to `dest` if given, else
    `card_path(cfg, text_label, kind)`. Returns the path written.
    """
    out_path = dest or card_path(cfg, text_label, kind)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    width, height = _card_dims(cfg)
    patches = {
        "PROMPT": {"prompt": _card_prompt(cfg, text_label, kind)},
        "SEED": {"value": _card_seed(cfg, text_label, kind)},
        "WIDTH": {"value": int(width)},
        "HEIGHT": {"value": int(height)},
        "FILENAME_PREFIX": {
            "filename_prefix": f"{_LEARNING_CARD_DIRNAME}/{_slug(kind)}_{_slug(text_label)}"
        },
    }
    log.info("refs: generating Z-Image learning card for %s %r (%dx%d) -> %s",
             kind, text_label, width, height, out_path)
    client.render(_ZIMAGE_WORKFLOW, patches, out_path)
    return out_path


# --------------------------------------------------------------- stills ---
_OMNI_REF_W, _OMNI_REF_H = 384, 512


def _prep_ref_for_omni(src_path, dest_hint):
    """Resize an identity ref to fixed patchify-safe dims (384x512) for the
    Omni encoder, writing beside `dest_hint`'s directory as
    ``_omni_<original stem>.png``. Idempotent (existing non-empty output is
    reused). 384x512 preserves the canonical refs' exact 3:4 portrait aspect
    (768x1024 halved) and divides cleanly by 8 (latent) and again by 2
    (sampler patchify) — the graph's own auto-resize does neither reliably
    for a portrait ref against a landscape render target (see caller).
    Falls back to the ORIGINAL path if PIL cannot process the file — the
    render then either succeeds anyway or fails into the caller's existing
    per-keyframe degradation, never raises from here."""
    try:
        from PIL import Image, ImageOps

        out_dir = os.path.dirname(os.path.abspath(dest_hint))
        os.makedirs(out_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(src_path))[0]
        out = os.path.join(out_dir, f"_omni_{stem}.png")
        if os.path.isfile(out) and os.path.getsize(out) > 0:
            return out
        with Image.open(src_path) as im:
            ImageOps.fit(im.convert("RGB"), (_OMNI_REF_W, _OMNI_REF_H)).save(out)
        return out
    except Exception:
        log.warning("refs: could not pre-fit %s for Omni — staging the original.",
                    src_path)
        return str(src_path)


def generate_still(client, cfg, prompt, width, height, dest, seed, ref_images=None):
    """Render one Z-Image still from a RAW caller-supplied prompt.

    `ref_images` (optional): up to 3 LOCAL image paths fed to
    `TextEncodeZImageOmni`'s image1/image2/image3 inputs as pixel-level
    IDENTITY ANCHORS — the canonical cast reference PNGs, so a keyframe's
    children keep their exact skin tone/hair/outfit instead of drifting on
    text alone (measured on job 22: "Zuri" rendered visibly lighter-skinned
    in one keyframe). Uses the `zimage_still_ref` graph (three LoadImage
    nodes wired into the Omni encoder + its vae input; validated against the
    live server's object_info). Fewer than 3 paths: the FIRST path fills the
    remaining slots (LoadImage cannot be empty; duplicate conditioning of the
    same child is harmless). None/empty: the plain text-only `zimage_ref`
    graph, byte-identical to before this parameter existed.

    The raw primitive underneath `generate_reference` / `generate_card`
    (both are thin prompt/dims-building wrappers around this same graph) and
    the future per-shot KEYFRAME renderer: unlike those two, this function
    builds nothing itself — no prompt templating, no dimension snapping. The
    caller is trusted to already know exactly what it wants.

    `prompt` must be the caller's COMPLETE positive prompt, used verbatim.
    Z-Image has no negative-text path — its `NEGATIVE` node is a
    `ConditioningZeroOut` of the positive conditioning, not an encoded
    negative prompt (see workflows/zimage_ref.json) — so anything to avoid
    must be phrased positively instead, and the LTX LoRA trigger token (e.g.
    "P1x4r") must NOT be included: it activates an LTX-specific LoRA and is
    meaningless to Z-Image's Qwen text encoder. Same reasoning as
    `_reference_prompt`, above.

    `width`/`height` are patched EXACTLY as given, with no snapping to a
    multiple of 16 — that is the caller's responsibility. For a video
    keyframe, that means matching the LTX shot's own dims (already multiples
    of 32). `cfg` is accepted for signature symmetry with `generate_reference`
    / `generate_card` (and so a future caller can be swapped in without a
    signature change); it is not consulted here for prompt, dims, or paths —
    the caller controls all of that via `prompt`/`width`/`height`/`dest`.

    Same "the ONLY GPU touch is `client.render`" contract as
    `generate_reference` / `generate_card`: patches
    PROMPT/SEED/WIDTH/HEIGHT/FILENAME_PREFIX (the latter derived from
    `dest`'s basename, under a "stills/" prefix) and writes the still to
    `dest`. Returns `dest`.
    """
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    stem = os.path.splitext(os.path.basename(dest))[0]
    patches = {
        "PROMPT": {"prompt": prompt},
        "SEED": {"value": int(seed)},
        "WIDTH": {"value": int(width)},
        "HEIGHT": {"value": int(height)},
        "FILENAME_PREFIX": {"filename_prefix": f"stills/{stem}"},
    }
    workflow = _ZIMAGE_WORKFLOW
    refs_list = [str(p) for p in (ref_images or []) if p][:3]
    if refs_list:
        # Identity-anchored variant — FLUX.2 klein 4B (workflows/
        # flux2_klein_ref.json), NOT the Z-Image Omni graph: Tongyi never
        # released the Edit/Omni weights that graph's image inputs target, and
        # the Turbo checkpoint renders references as spatial COLLAGE (three
        # live probes 2026-07-22, IMP-012). klein's ReferenceLatent chain held
        # identity on the clay cast at first try (one coherent scene, the
        # right child per reference, correct count — output/_klein_probe/).
        # Different graph, different patch dialect: CLIPTextEncode wants
        # "text" (not "prompt"), the seed rides the SEED_VALUE PrimitiveInt,
        # and identity binds best when the prompt points at the references
        # explicitly. Refs still pre-fit to 384x512 (deterministic, cheap;
        # klein's VAEEncode accepts arbitrary sizes, but a conditioning latent
        # has no use for 768x1024).
        workflow = _KLEIN_REF_WORKFLOW
        staged = [client.stage_input_image(_prep_ref_for_omni(p, dest)) for p in refs_list]
        while len(staged) < 3:
            staged.append(staged[0])
        patches = {
            # Style-NEUTRAL reference pointer: this used to say "the clay
            # children", which was correct only for the claymation style — on a
            # pixar_toon episode it would push every anchored keyframe toward a
            # clay look. The render style's own skeleton (inside `prompt`)
            # carries the look; this clause only needs to bind identity.
            "PROMPT": {"text": (
                "Using the children from the reference images with their "
                "exact faces, skin tones, hairstyles and outfits: " + prompt
            )},
            "SEED_VALUE": {"value": int(seed)},
            "WIDTH": {"value": int(width)},
            "HEIGHT": {"value": int(height)},
            "FILENAME_PREFIX": {"filename_prefix": f"stills/{stem}"},
            "IMAGE1": {"image": staged[0]},
            "IMAGE2": {"image": staged[1]},
            "IMAGE3": {"image": staged[2]},
        }
    log.info("refs: generating %s still (%dx%d, %d identity refs) -> %s",
             "klein" if refs_list else "Z-Image", width, height, len(refs_list), dest)
    client.render(workflow, patches, dest)
    return dest
