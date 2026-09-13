# ComfyUI workflows for local AI video backgrounds

The app can call a **local ComfyUI server** to generate animated backgrounds
using **LTX-Video 2B (v0.9.5)** — a text-to-video model small enough for an
8 GB GPU like an RTX 4060. This whole feature is **optional and off by
default** (`comfyui.enabled: false` in `config.example.json`); it needs an
NVIDIA GPU and a ComfyUI install that is **not** part of this repo. Without
one, the `aivideo` asset tier is simply skipped. This file documents the
workflow JSONs in this folder and what a working install looks like.

## What a working install looks like (under `comfyui/`, gitignored)

You install this yourself (or via your own automation) — the repo scripts only
*start* an existing install, they don't create one:

- ComfyUI core (own venv, PyTorch + CUDA 12.4)
- Custom nodes: `ComfyUI-LTXVideo`, `ComfyUI-VideoHelperSuite` (mp4 export),
  `ComfyUI-Manager`
- Models: `models/checkpoints/ltx-video-2b-v0.9.5.safetensors` +
  `models/text_encoders/t5xxl_fp16.safetensors`

`scripts/start-comfyui.ps1` starts the server on `http://127.0.0.1:8188` and
is called automatically by `Start-BrainrotStudio.bat`; if `comfyui/` or the
model file is missing it skips quietly and rendering falls back to the other
background tiers. Enable the tier in `config.json`:

```json
"comfyui": { "enabled": true, ... }
```

## `ltx_t2v.json` — the real workflow

This is a hand-authored **API-format** ComfyUI graph (not something you need
to export yourself): `CheckpointLoaderSimple` → `CLIPLoader` (t5xxl, type
`ltxv`) → two `CLIPTextEncode` (positive/negative) → `EmptyLTXVLatentVideo` →
`LTXVConditioning` → `LTXVScheduler` → `SamplerCustom` (sampler:
`res_multistep`, cfg 3, 30 steps — Lightricks' recommended 0.9.5 settings) →
`VAEDecode` → `VHS_VideoCombine` (h264 mp4 output).

The app overwrites the node titled **`positive`** with a per-video prompt
(`comfyui.prompt_template`, default a kids 3D-cartoon style) and injects
`width`/`height`/`frames`/`frame_rate` from `config.json → comfyui` into every
matching node, so resolution changes there apply without touching the JSON.

If ComfyUI is disabled, unreachable, or the model files are missing, the
`aivideo` asset tier is skipped and the app falls back to stock/generated
backgrounds — rendering never blocks on this.

## Swapping the model later

Want a different/better model (e.g. once you have more VRAM, or a newer LTX
release)? Replace `ltx_t2v.json` with your own **Save (API Format)** export
from the ComfyUI web UI (`http://127.0.0.1:8188`) — give the positive prompt
node the title `positive` (or update `comfyui.prompt_node_title`) so the app
can find it. `ltx_t2v.example.json` shows the minimal expected shape.

## Files in this folder

- `ltx_t2v.json` — the API-format workflow the app actually submits
  (`comfyui.workflow_path` in config points here)
- `ltx_t2v.example.json` — minimal example of the expected API-format shape
- `ltxv_ui_format.json` — the same graph as a normal ComfyUI **UI** export, for
  loading/editing in the web UI (the app does not read this one)
