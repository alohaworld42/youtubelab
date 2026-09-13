# workflows/

ComfyUI node graphs in **API format** (`Save (API Format)` from the ComfyUI
UI — a dict of `node-id -> {"class_type", "inputs", "_meta": {"title": ...}}`).
`pipeline.kidsong.comfy.ComfyClient` loads these straight off disk, patches a
handful of values by node title, and POSTs the result to a local ComfyUI
server (`/prompt`). Nothing here is a ComfyUI UI export — there is no `nodes`/
`links` array, just the API dict ComfyUI's `/prompt` endpoint accepts as-is.

Hardware target: RTX 5070, 12GB VRAM, LTX-2.3 fp8 + the distilled LoRA. Every
graph below stays inside that budget — see "VRAM guardrail" below.

## Graphs

| Graph | Purpose | Used by (`render_style` / caller) | Pendant / derived from |
|---|---|---|---|
| `ltx23_t2v_toon.json` | Text-to-video, base pass, 896x512, 97 frames, 3-step distilled schedule | `pixar_toon`, `pixar_toon_concise`, `pixar_toon_lean`, `flat_storybook`, `claymation`, `felt` (all via `workflow_base: "ltx23_t2v_toon"`) | — (original) |
| `ltx23_t2v_toon_hires.json` | Same, + a 2x latent-upscale refine pass (~1792x1024) | same styles, when `kidsong.shot.hires_pass` is true | `ltx23_t2v_toon.json` (adds `UPSCALE_MODEL`/`UPSAMPLER`/`*_REFINE` nodes) |
| `ltx23_t2v_explainer.json` | Text-to-video, non-toon "clean motion-graphics explainer" look — no cast, prompt-driven, style LoRA neutralized | `explainer_clean` (base pass) | derived from `ltx23_t2v_toon.json` — see "Contract" |
| `ltx23_t2v_explainer_hires.json` | Same, hires pass | `explainer_clean` (`hires_pass: true`) | derived from `ltx23_t2v_toon_hires.json` |
| `ltx23_t2v_meme.json` | Text-to-video, non-toon "punchy high-contrast internet-native" look — no cast, prompt-driven, style LoRA neutralized | `meme_pop` (base pass) | derived from `ltx23_t2v_toon.json` — see "Contract" |
| `ltx23_t2v_meme_hires.json` | Same, hires pass | `meme_pop` (`hires_pass: true`) | derived from `ltx23_t2v_toon_hires.json` |
| `ltx23_i2v_toon.json` | Image-to-video: animates a `LoadImage` conditioning frame (keyframe-first / reference-anchored shots) | `render_style.i2v_workflow_name` — shared by every style, base pass | — (independent graph; shares the toon look/LoRA) |
| `ltx23_i2v_toon_hires.json` | Same, + hires refine | `i2v_workflow_name`, `hires_pass: true` | `ltx23_i2v_toon.json` |
| `ltx23_flf2v_toon.json` | First-last-frame image-to-video: bridges two conditioning frames into a shot-to-shot transition | `pipeline.kidsong.transitions` (hardcoded name, not style-registry driven) | — (independent graph) |
| `flux2_klein_ref.json` | Flux.2 Klein text-to-image: cast reference stills | `pipeline.kidsong.refs` (`_KLEIN_REF_WORKFLOW`) | — |
| `zimage_ref.json` | Z-Image Turbo text-to-image: single-subject reference stills / keyframes | `pipeline.kidsong.refs` (`_ZIMAGE_WORKFLOW`) | — |
| `zimage_still_ref.json` | Z-Image Turbo image-to-image: composites up to three `LoadImage` cast references into one still | `pipeline.kidsong.refs` (`_ZIMAGE_REF_WORKFLOW`) | — |

**Note on wiring:** `config.example.json`'s `explainer_clean` and `meme_pop`
entries currently still set `workflow_base: "ltx23_t2v_toon"` (they predate
these two graphs and reused the toon graph with `style_strength: 0.0`, which
silently renders the toon look's own prompt/negative baseline under a
different skeleton). `ltx23_t2v_explainer[_hires].json` and
`ltx23_t2v_meme[_hires].json` above satisfy the exact same patch contract the
toon graphs do (see "Contract" below) and are ready to be pointed at from
`workflow_base` — that config edit is a separate change, not made here.

## Contract

`pipeline/kidsong/comfy.py` (`ComfyClient._apply_patches`) patches graphs **by
node title** (`_meta.title`), never by node id — `generate.py` never sees or
depends on the numeric ids a graph happens to use internally. This is what
lets a style swap its `workflow_base` without any code change: as long as the
new graph carries the same titles, the same patch calls apply.

For a `ltx23_t2v_*` (text-to-video) graph, every render unconditionally
patches these titles (`generate.py`'s `base_patches`, primary input per
`comfy.py::_PRIMARY_INPUT`):

| Title | Patched input | Meaning |
|---|---|---|
| `PROMPT` | `text` | The composed shot prompt (trigger + skeleton + identity + tail) |
| `NEGATIVE` | `text` | Baseline negative (read back from the graph itself, per-shot terms appended) — see `_baseline_negative` in `generate.py`: this is **not** overwritten wholesale, so a graph's own authored negative text is load-bearing |
| `LORA_STYLE` | `lora_name` + `strength_model` | The active style's LoRA file and strength (`style_lora`/`style_strength`) |
| `SEED` | `noise_seed` | Per-shot/per-attempt seed |
| `WIDTH` / `HEIGHT` | `value` | `kidsong.shot.width`/`.height` |
| `FRAMES` | `value` | `kidsong.shot.frames` (scaled by quality tier) |
| `FILENAME_PREFIX` | `filename_prefix` | Output path prefix |

A graph missing any of these titles renders `_apply_patches` fail with
`KeyError: No node titled ... in this workflow` — a real render break, not a
cosmetic one. `tests/test_workflow_graphs.py` asserts every
`ltx23_t2v_*.json` graph carries the full set.

Image-to-video and first-last-frame graphs carry additional titles
(`INPUT_IMAGE`; `FIRST_IMAGE`/`LAST_IMAGE`/`GUIDE_FIRST`/`GUIDE_LAST`) that
`ltx23_t2v_*` graphs must **not** need — those exist only to bracket a
conditioning frame the t2v path has no equivalent of.

### Deriving a new non-toon look

`ltx23_t2v_explainer[_hires].json` and `ltx23_t2v_meme[_hires].json` are byte-
for-byte copies of their `ltx23_t2v_toon[_hires].json` pendant with exactly
three deltas (verified by `tests/test_workflow_graphs.py`):

1. `LORA_STYLE.strength_model` set to `0.0` (the LoRA loads but contributes
   nothing — these looks are prompt-driven, matching `explainer_clean`'s and
   `meme_pop`'s `style_strength: 0.0` in `config.example.json`).
2. `NEGATIVE`'s authored baseline text extended: `explainer` adds terms
   against children/toddlers and cartoon-mascot figures appearing in frame
   (the toon graph's baseline is written for a fixed toddler cast — an
   explainer frame must not get one by accident); `meme` adds terms against
   burned-in/embedded text and watermark artifacts (high-contrast internet-
   native renders are more prone to inventing caption-like text than the
   toon look is).
3. `PROMPT`'s placeholder text and `FILENAME_PREFIX`'s default value —
   cosmetic only, both are unconditionally overwritten by every real render.

Everything else — node ids, wiring, `class_type`s, resolution (896x512 /
~1792x1024 hires), frame count (97), sampler (`euler`), and the sigma
schedule — is untouched, so the same patch code and the same VRAM budget as
the toon graph apply unchanged.

### VRAM guardrail

New graphs must never exceed the toon graph's VRAM footprint for the target
hardware (RTX 5070, 12GB). Concretely: same `WIDTH`/`HEIGHT`, same `FRAMES`,
same `SIGMAS`/`SIGMAS_REFINE` step count as the toon pendant they derive
from. `tests/test_workflow_graphs.py` checks all three for the four derived
graphs.

## Validation

```
python -m pytest tests/test_workflow_graphs.py -q
```

Checks every `workflows/*.json` graph (not just the `ltx23_t2v_*` family):
valid JSON/dict shape, every node has `class_type` + `inputs`, no two nodes
share a `_meta.title` within one graph (would make title-patching
ambiguous), and every `[node_id, slot]` link input resolves to a node that
exists in the same graph (a dangling one means ComfyUI refuses to load the
graph at all). Then the contract and look-specific checks described above.

This is a static/offline validator — it never talks to a ComfyUI server. Use
`python -m pipeline.kidsong.comfy --validate-only` (needs a running or
autostartable ComfyUI) to additionally confirm every `class_type` a graph
uses is actually installed.
