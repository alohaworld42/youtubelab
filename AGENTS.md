# 🤖 Agent Guide — Brainrot Studio

This guide helps AI agents work effectively in this repo.

## Project

**Brainrot Studio** — local YouTube Shorts generator for Windows + NVIDIA GPU.

- **Web UI**: Flask at `app.py`, running on `http://127.0.0.1:5000`
- **Pipeline** (in `pipeline/` package):
  - `script_gen` → LLM generates video script
  - `tts` → edge-tts voices it
  - `captions` → Whisper extracts words, syncs timing
  - `assemble` → FFmpeg/MoviePy renders to MP4 (9:16, with gameplay background)
  - `youtube_upload` → optional one-click upload as YouTube Short
- **Kidsong type** (new, in `pipeline/kidsong/`):
  - **Director mode** (default): LLM lyrics → ACE-Step sings → Whisper word alignment → librosa beat grid → director-generated shot list (kids-TV grammar) → ComfyUI renders per shot (LTX-2.3 fp8 + distilled LoRA, Pixar toon style) → per-take QC review (heuristics + optional blocking external vision review) → beat-aligned edit + color grade → cut QC gate → Final/ promotion
  - **Classic mode**: legacy SDXL keyframes + Wan2.2/LTX-0.9.1 i2v clips (fallback escape hatch)
  - ComfyUI headless on 127.0.0.1:8188 (venv python, torch cu128); models cached in D:\brainrot\comfy-models

## Environment

- **OS**: Windows 11 Pro
- **GPU**: RTX 5070 12GB (Blackwell, sm_120)
- **Python**: 3.11+ (in project venv at `./venv`)
- **Venv location**: `D:\brainrot\venv` (junctioned to `./venv` because C: drive is space-constrained)
- **HuggingFace model cache**: `D:\brainrot\hf` (large downloads go here, not C:)

**Never install large packages or download ML models to C:**.

## ComfyUI workflows

### `workflows/` is the source of truth, and it is API format

The LTX-2.3 graphs the kidsong pipeline renders with live in `workflows/*.json`:

| File | Used for |
| --- | --- |
| `ltx23_t2v_toon.json` | text→video shots (23 nodes) |
| `ltx23_t2v_toon_hires.json` | text→video + latent upsample refine pass (33 nodes) |
| `ltx23_i2v_toon.json` | image→video shots (26 nodes) |
| `ltx23_flf2v_toon.json` | first-last-frame→video transition clips (30 nodes) |

These are stored in **API format** — `{node_id: {class_type, inputs, _meta}}` —
because that is the only shape ComfyUI's `POST /prompt` accepts.
`pipeline/kidsong/comfy.py` loads one, splices in per-shot values, and POSTs it
verbatim. **This is the authoritative copy.**

### How `comfy.py` patches a graph

Graphs are patched **by node title**, never by node id, so the graph can be
rearranged without breaking the pipeline. `ComfyClient.render(workflow, patches)`
takes `{title: value}`; a scalar goes to that title's primary input (below), and
a `{input_name: value}` dict sets inputs explicitly.

| `_meta.title` | Primary input | Node class |
| --- | --- | --- |
| `PROMPT` | `text` | `CLIPTextEncode` |
| `NEGATIVE` | `text` | `CLIPTextEncode` |
| `SEED` | `noise_seed` | `RandomNoise` |
| `WIDTH` | `value` | `PrimitiveInt` |
| `HEIGHT` | `value` | `PrimitiveInt` |
| `FRAMES` | `value` | `PrimitiveInt` |
| `FILENAME_PREFIX` | `filename_prefix` | `SaveVideo` |
| `INPUT_IMAGE` | `image` | `LoadImage` (i2v only) |
| `FIRST_IMAGE` | `image` | `LoadImage` (flf2v only) |
| `LAST_IMAGE` | `image` | `LoadImage` (flf2v only) |
| `GUIDE_FIRST` | `strength` | `LTXVAddGuide` @ `frame_idx=0` (flf2v only) |
| `GUIDE_LAST` | `strength` | `LTXVAddGuide` @ `frame_idx=-1` (flf2v only) |
| `LORA_STYLE` | `strength_model` (a **string** value swaps `lora_name` instead) | `LoraLoaderModelOnly` |

Renaming or deleting one of these titles breaks the pipeline: `_apply_patches`
raises `KeyError: No node titled ...`. Other titles (`CHECKPOINT`, `GUIDER`,
`SIGMAS`, `DECODE`, …) are documentation and safe to read but not patched.

### Opening the same graphs in the ComfyUI UI

The ComfyUI sidebar cannot open API JSON — it loads files through
`validateComfyWorkflow()`, which rejects anything without a `version` field and
expects litegraph's UI format (`{nodes: [...], links: [...]}`). So the repo
copies are converted and installed here:

```
C:\Users\Aloha\Desktop\projects\ComfyUI\user\default\workflows\kidsong\
  kidsong_ltx23_t2v_toon.json
  kidsong_ltx23_t2v_toon_hires.json
  kidsong_ltx23_i2v_toon.json
```

Open ComfyUI → **Workflows** sidebar → `kidsong/`. Node titles are preserved, so
`PROMPT`, `SEED`, `WIDTH` etc. are visible and editable exactly as the pipeline
names them.

### Re-syncing after a workflow change

```
venv\Scripts\python.exe tools/sync_comfy_workflows.py            # convert + install
venv\Scripts\python.exe tools/sync_comfy_workflows.py --dry-run  # preview only
```

The script is idempotent, prints every file it writes, and **refuses to write a
graph that fails verification** (widget count vs `/object_info`, link endpoints,
title preservation). It reads `/object_info` from the running ComfyUI to get
authoritative input ordering, so ComfyUI must be up; pass
`--object-info <cached.json>` to work from a snapshot instead.

`tools/comfy_workflow_to_ui.py` holds the conversion itself and can be used
standalone (`python tools/comfy_workflow_to_ui.py workflows/x.json -o out.json`).
Covered by `tests/test_comfy_workflow_sync.py` (no network — stubbed
`/object_info`).

### ⚠️ WARNING: the UI copy is a read-only mirror

**Editing a graph in the ComfyUI UI does NOT change what the pipeline renders.**
The synced files under `user/default/workflows/kidsong/` are generated artifacts;
the next `sync_comfy_workflows.py` run overwrites them.

To make a UI tweak stick:

1. Edit the graph in ComfyUI and confirm it renders.
2. Export it as **API format** (Workflow → Export (API), *not* plain Export).
3. Save it over the matching file in `workflows/`.
4. Confirm the `_meta.title` values above survived the round-trip — ComfyUI
   writes node titles into `_meta.title` on API export, but a retitled or
   replaced node will silently drop the patch target.
5. Re-run `tools/sync_comfy_workflows.py` so the UI copy matches again.

## Transitions and visual continuity (LTX-2.3 frame guiding)

There is **no local first-last-frame node** in ComfyUI: every `*FirstLastFrame*`
class (`Kling`, `Veo3`, `Runway`, `ByteDance`, `LumaRay32Keyframe*`,
`PixverseTransitionVideo`) is a paid cloud API node and unusable here. The local
mechanism is **`LTXVAddGuide`**, chained twice.

### How `LTXVAddGuide` actually works

Verified against `ComfyUI/comfy_extras/nodes_lt.py` (`class LTXVAddGuide`) and
the live `/object_info`:

```
inputs:  positive (COND), negative (COND), vae (VAE), latent (LATENT),
         image (IMAGE), frame_idx (INT, default 0, min -9999, max 9999),
         strength (FLOAT, default 1.0, 0.0-10.0), attention_mask (MASK, opt)
outputs: positive (COND), negative (COND), latent (LATENT)
```

- `frame_idx` is in **pixel frames**, not latent frames. It is converted with
  `latent_idx = (frame_idx + time_scale_factor - 1) // time_scale_factor`, where
  the LTX VAE's `time_scale_factor` is **8**.
- **Negative indices are supported** and count from the end:
  `frame_idx = max((latent_count - 1) * 8 + 1 + frame_idx, 0)`. So `-1` is the
  last frame. This is what makes FLF2V possible.
- For **multi-frame** guides `frame_idx` is snapped down to the form `8n + 1`.
  For a **single still image** (`guide_length == 1`) that rounding is skipped
  entirely, so any `frame_idx` is legal. We always guide with single stills.
- Guides are **appended** to the tail of the latent tensor, not written in place,
  and each guide records `keyframe_idxs` / `guide_attention_entries` into the
  conditioning. Therefore the **second `LTXVAddGuide` must consume the first's
  `positive`/`negative`/`latent` outputs** — chaining is mandatory, or the
  `num_keyframes` bookkeeping breaks and `-1` resolves to the wrong frame.
- `strength` sets the guide's noise mask to `max(0, 1 - strength)`: `1.0` pins
  the frame exactly, lower values leave the model freedom. The official template
  uses **0.7** for both guides, and so do we — a hard 1.0 pin tends to freeze
  motion at the clip edges.
- `LTXVCropGuides` strips the appended guide frames back off after sampling. It
  must sit **after** the sampler (and, in our AV graphs, after
  `LTXVSeparateAVLatent`) and **before** `VAEDecodeTiled`, and it must read the
  same `positive`/`negative` that fed the sampler.

`workflows/ltx23_flf2v_toon.json` is a faithful port of ComfyUI's own
`video_ltx2_3_flf2v.json` template onto our LoRA stack, negatives and sigmas:

```
LTXVConditioning ─► LTXVAddGuide(frame_idx=0,  first frame)
                 ─► LTXVAddGuide(frame_idx=-1, last frame)
                    ├─ positive/negative ─► CFGGuider
                    └─ latent ─► LTXVConcatAVLatent ─► SamplerCustomAdvanced
                                 ─► LTXVSeparateAVLatent ─► LTXVCropGuides
                                 ─► VAEDecodeTiled ─► CreateVideo ─► SaveVideo
```

### Where transitions belong — and where they do not

`pipeline/kidsong/transitions.py` implements two independent mechanisms. Both
default to **off** and both **fail soft to today's hard cut**.

1. **Shot chaining** (`kidsong.shot_chaining_enabled`) — shot N+1 renders through
   `ltx23_i2v_toon` using shot N's accepted last frame as its opening guide.
   This costs **zero extra renders** and is the real fix for cast drift, because
   it gives the model a pixel-level anchor rather than only a text description.
   Chains are capped (`shot_chain_max_len`, default 3) to bound generation loss
   and are **never allowed to bridge a scene or verse change**.
2. **Generated transition clips** (`kidsong.transitions_enabled`) — a short
   FLF2V clip from shot A's last frame to shot B's first frame, inserted by the
   editor at **verse/section boundaries only** (`transitions_at: "verse"`,
   capped by `transitions_max_per_episode`).

**Hard cuts stay the default everywhere else, deliberately.** Our own reference
analysis of real preschool-TV footage found cuts follow lyric phrasing, and
`edit.py` already ships with `_CROSSFADES_ENABLED = False` for the same reason:
a mid-blend frame reads as a double exposure and a channel that dissolves every
shot looks *worse*, not smoother. Transitions are punctuation for section
changes, not glue for every cut.

## Model routing playbook (for orchestrator sessions)

When dispatching work across multiple Claude instances:

1. **Frontier model** (Opus) for orchestration/synthesis
2. **Sonnet** for implementation, edits, tests
3. **Haiku** for search, classification, docs
4. **Escalate**: if a task fails, bump it one tier up instead of defaulting to frontier
5. **Parallel dispatch**: send independent worker jobs async; don't block orchestrator
6. **Fresh-context verification**: verify finished work with a fresh-context agent; don't trust prior session state

## Quality management (meta-QC)

The four blocking gates judge one artifact at a time. The level above them is the
**quality-manager agent** (`.claude/agents/quality-manager.md`), invoked via
`/quality-manager [episode|all] [quick|full]`: it audits final cuts and mines
`review_log.json` history across episodes, researches video-generation sources on the web,
and files workflow improvements. Its ledger lives in `docs/quality/`:

- `DEFECT_BACKLOG.md` — one entry per defect class (QM-NNN), with stage attribution and a metric
- `IMPROVEMENT_LOG.md` — one entry per workflow change (IMP-NNN), hypothesis + verification status
- `SOURCES.md` — vetted external sources (LTX prompting guides, AIGC-video evaluation papers, IP/compliance)

Rules of the loop: one change per defect class at a time, always with a metric the next
audit re-checks; gates may get stricter freely but loosening one requires false-positive
evidence; IP-lookalike drift (e.g. toward CoComelon characters/trade dress) is a
ship-blocking S1 finding.

**Keep the vision gate answered (QM-010).** The external review gate only works if something
answers its `*.request.json` files — unattended runs previously shipped with 0-6% real vision
coverage because nobody was watching the browser queue. Run
`python -m pipeline.kidsong.auto_review --watch` alongside the scheduler/CLI for any unattended
production stretch: it discovers pending shot/cut/script/shotlist requests
(`pipeline/kidsong/review_queue.py`), judges each one with the `scene-plausibility` rubric (or a
matching cut/script/shotlist prompt) via the local `claude` CLI in headless mode, and writes the
same `response.json` a human reviewer would. `--once` drains the current backlog and exits. No QC
criteria change — this only makes sure the gate that already exists actually gets exercised.

The human side of the same gate is `/kidsong-review` (**Take-Review** in the nav, with a live
pending counter on every page). Things to know before touching it:

- It writes exactly the `*.response.json` files `review.poll_response()` polls for — that
  contract is the whole interface, do not add fields the pipeline doesn't read.
- The page auto-refreshes, but only while nobody is using it: a dirty form, a focused input,
  playing media or a backgrounded tab all pause it. It used to be a `<meta http-equiv=refresh>`
  that threw away half-finished reviews every 8 seconds — do not put one back (QM-016).
- Ticking a reject reason means reject, enforced server-side. The score is clamped to 0–1, the
  target path must be a `*.response.json` next to an existing `*.request.json`, and an
  already-answered gate is never overwritten.

## UI pages

Every template extends `templates/base.html` (nav, theme, toast, focus styles). A page that
rolls its own `<html>` is a page that silently drops the navigation — that is how the
production dashboard and the take-review queue ended up unreachable except from body text.

## Code conventions

- **MoviePy 2.x idioms**: use `.resized()`, `.subclipped()`, `.with_start()` (not `subclip()`, `set_start()`)
- **Lazy heavy imports**: import `moviepy`, `diffusers`, `PIL`, `torch` inside functions, not at module top
- **Config**: load via `pipeline.config.load_config()` with sensible code defaults for missing keys; don't require all keys in config.json
- **Captions**: render with Pillow (`PIL.ImageDraw`), never ImageMagick
- **Kidsong content**: all original only (no copyrighted characters, lyrics, or brands)
- **Kidsong uploads**: always declare `made_for_kids=True` when uploading to YouTube. The declaration follows what the video *is*, not which column was filled in: `scheduler._made_for_kids` treats the channel checkbox, a kids `video_type` and a kids `style` as three opt-in signals, any of which turns it on and none of which can turn it off. Never infer it from channel name/description prose — a wrong COPPA declaration is a legal filing, and over-declaring costs personalised ads, comments and end screens (QM-043/IMP-051)
- **AI disclosure**: every published video is disclosed as AI-generated via `pipeline/ai_disclosure.py`, applied inside `youtube_upload.upload()` so no call site has to remember. Never add a fourth path to `videos.insert` that bypasses it, and never append the note *after* truncating a description — the disclosure must not be the part that gets cut (QM-042/IMP-050)
- **Generated media in output/ are channel inventory — never delete takes, songs, or superseded cuts; only test artifacts. QC gates**: script, shot, and cut review gates must stay blocking in external mode; never bypass them to ship faster. Final/ mp4s are the shipped, studio.db-referenced artifacts and must never be hand-deleted for disk space — reclaim space from shots dirs/staging/wav first, and remove a shipped final only via `runstate.remove_final_episode()`

