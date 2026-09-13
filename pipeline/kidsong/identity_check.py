"""kidsong.identity_check — a CLIP-based "is this the RIGHT cast child?" test
for a rendered single-subject keyframe.

The klein reference-anchored keyframe (refs.generate_still with ref_images) pins
identity from the child's canonical reference, but it is a stochastic 4-step
sample: most renders land on-model, a minority still drift a child into a
castmate's hair/wardrobe (measured 2026-07-23: solo Kofi keyframes drifting into
Zuri's afro puffs). Because the drift is stochastic — a fresh render at a bumped
seed usually lands clean — the render loop can simply RE-RENDER a flagged
keyframe until it matches. This module is the flag: it scores a keyframe against
every cast child's reference and reports whether the intended child wins.

`identity_margin(image, char_id, cfg)` returns
    sim(image, ref[char_id]) - max(sim(image, ref[other cast child]))
in CLIP image-embedding cosine space:
  * > 0  — the keyframe looks more like `char_id` than any castmate (on-model);
  * < 0  — it looks more like some OTHER cast child (drifted).
Empirically (openai/clip-vit-base-patch32) this separates clean from drifted
solo keyframes with a clean gap around 0 for the ZubiBop clay cast.

Best-effort by contract, exactly like the rest of the anchor stack: any missing
piece — transformers/torch absent, the CLIP weights not cached locally, a
reference PNG missing — makes every function return None, so the caller skips
the retry and the keyframe is byte-identical to a no-check render. Never raises.
The heavy model is lazy-loaded once and cached process-wide.

`prop_presence(image, prop_text, cfg)` is the sibling check for shots whose
story hinges on an interaction OBJECT rather than a cast identity (e.g. a shot
whose beat is "Kofi holds the umbrella" — the child can be perfectly on-model
while the umbrella never rendered). It is a zero-shot CLIP binary classifier
scoring how much the image looks like it contains `prop_text` vs. not, with no
reference image needed. Same best-effort contract as identity_margin: any
failure — including a bad path or an empty prop_text — returns None, never
raises.
"""
import logging
import math
import os

log = logging.getLogger("kidsong.identity_check")

# openai/clip-vit-base-patch32 — small (~600MB), already used to validate the
# detector; cached under HF_HOME. Overridable via kidsong.identity_check.model.
_DEFAULT_MODEL = "openai/clip-vit-base-patch32"
_DEFAULT_HF_HOME = "D:/brainrot/hf"

# Lazy singleton: None = not tried yet, False = tried and unavailable,
# else (torch, model, processor).
_LOADED = None


def _load(cfg=None):
    global _LOADED
    if _LOADED is not None:
        return _LOADED or None

    ic = ((cfg or {}).get("kidsong", {}) or {}).get("identity_check", {}) or {} if isinstance(cfg, dict) else {}
    model_name = ic.get("model", _DEFAULT_MODEL)
    # Load from the local HF cache only — a render must never block on a network
    # download. HF_HOME points at the D: model store (C: is space-constrained).
    os.environ.setdefault("HF_HOME", ic.get("hf_home", _DEFAULT_HF_HOME))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        model = CLIPModel.from_pretrained(model_name)
        model.eval()
        processor = CLIPProcessor.from_pretrained(model_name)
        _LOADED = (torch, model, processor)
        log.info("identity_check: CLIP model %s ready.", model_name)
        return _LOADED
    except Exception as exc:
        log.warning(
            "identity_check: CLIP unavailable (%s) — keyframe identity retries "
            "are disabled (renders are byte-identical to no-check).", exc,
        )
        _LOADED = False
        return None


def _embed(loaded, path):
    torch, model, processor = loaded
    from PIL import Image

    with torch.no_grad():
        pixel_values = processor(images=Image.open(path).convert("RGB"),
                                 return_tensors="pt")["pixel_values"]
        # Some transformers builds return an output object from
        # get_image_features; go through the vision model + projection directly
        # so the embedding is always a plain tensor.
        feats = model.visual_projection(model.vision_model(pixel_values=pixel_values).pooler_output)
    return feats / feats.norm(dim=-1, keepdim=True)


def _embed_text(loaded, text):
    torch, model, processor = loaded
    with torch.no_grad():
        inputs = processor(text=[text], return_tensors="pt", padding=True)
        feats = model.text_projection(
            model.text_model(input_ids=inputs["input_ids"],
                             attention_mask=inputs.get("attention_mask")).pooler_output
        )
    return feats / feats.norm(dim=-1, keepdim=True)


def available(cfg=None):
    """True if the detector can actually score (model + at least the reference
    lookups importable). Cheap after the first call."""
    return _load(cfg) is not None


def identity_margin(image_path, char_id, cfg):
    """How much more `image_path` looks like `char_id` than any other cast child.

    Returns a float (positive = on-model, negative = drifted into a castmate) or
    None when the check cannot run (no CLIP, no references, bad path). Never
    raises."""
    loaded = _load(cfg)
    if not loaded:
        return None
    try:
        from pipeline.kidsong import cast, refs

        own_ref = refs.reference_for(cfg, char_id)
        if not own_ref or not os.path.exists(own_ref):
            return None
        other_ids = [c.get("id") for c in cast.load_bible().get("characters", [])
                     if c.get("id") and c.get("id") != char_id]
        other_refs = [refs.reference_for(cfg, oid) for oid in other_ids]
        other_refs = [r for r in other_refs if r and os.path.exists(r)]
        if not other_refs:
            return None

        img = _embed(loaded, image_path)

        def sim(ref_path):
            return float((img @ _embed(loaded, ref_path).T).item())

        return sim(own_ref) - max(sim(r) for r in other_refs)
    except Exception as exc:
        log.warning("identity_check: margin failed for %s (%s) — skipping retry.", char_id, exc)
        return None


def prop_presence(image_path, prop_text, cfg=None):
    """Zero-shot CLIP check: how much does `image_path` look like it contains
    `prop_text` (a short noun phrase for the shot's interaction object, e.g.
    "an umbrella")?

    Embeds the image once and two prompts — "a picture with {prop_text}" and
    "a picture without {prop_text}" — via the same CLIP model as
    identity_margin, scores both with CLIP's own logit_scale (matching how
    CLIP is trained/calibrated for zero-shot classification), softmaxes the
    pair, and returns P(with) as a plain float in [0, 1]: closer to 1 means
    the prop is visibly present, closer to 0 means it looks absent.

    Offline calibration procedure (no labeled dataset — use existing renders):
    run this over a set of episode keyframes with KNOWN prop presence/absence,
    e.g. an episode whose story_subject is the prop in question (the umbrella
    episode's keyframes, where every shot's beat puts the umbrella on-frame)
    as positives, and output/_cast_refs/*/canonical.png (plain character
    portraits, no props) as negatives. Plot the two score distributions and
    pick the midpoint of the separation as the threshold. As of 2026-07-24
    this hasn't been run yet; kidsong.keyframe_first.prop_min_prob ships at a
    conservative 0.6 default until it is. This function does not read that
    config — the caller (the keyframe retry loop) owns the threshold decision.

    Returns None — never raises — when: prop_text is empty/None, CLIP is
    unavailable, image_path doesn't exist/can't be opened, or any other
    failure occurs; the caller then skips the prop-presence retry exactly as
    if the check were never wired in."""
    if not prop_text:
        return None
    loaded = _load(cfg)
    if not loaded:
        return None
    try:
        _torch, model, _processor = loaded
        img = _embed(loaded, image_path)
        with_emb = _embed_text(loaded, f"a picture with {prop_text}")
        without_emb = _embed_text(loaded, f"a picture without {prop_text}")

        scale = float(model.logit_scale.exp().item())
        logit_with = float((img @ with_emb.T).item()) * scale
        logit_without = float((img @ without_emb.T).item()) * scale

        # Two-way softmax, computed by hand (rather than a torch.softmax call)
        # so this stays testable with plain float stand-ins, exactly like
        # identity_margin's sim() above.
        peak = max(logit_with, logit_without)
        e_with = math.exp(logit_with - peak)
        e_without = math.exp(logit_without - peak)
        return e_with / (e_with + e_without)
    except Exception as exc:
        log.warning("identity_check: prop_presence failed for %r (%s) — skipping retry.", prop_text, exc)
        return None
