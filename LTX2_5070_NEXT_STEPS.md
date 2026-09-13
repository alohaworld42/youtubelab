> # ⚠️ ARCHIVED — superseded, do not follow
>
> This is a **historical record of 2026-07-11**, kept only because
> `docs/quality/DEFECT_BACKLOG.md` cites it as the "before" state of the
> character-coherence defect class. Every "next step" below has since been done
> or overtaken; following it now would walk the pipeline backwards.
>
> What actually happened since:
>
> | This doc says | Reality now |
> | --- | --- |
> | Move rendering to the 5070 and upgrade to LTX-2 FP8 | Done, and past it — the kidsong path renders on **LTX-2.3 fp8 + distilled LoRA** |
> | Point `comfyui.workflow_path` at an LTX-2 workflow | Obsolete — the kidsong graphs live in **`workflows/*.json`** (API format) and are patched by node title; `studio/comfy_workflows/` is only the legacy `aivideo` background tier |
> | `prompts/kids.txt` → `studio/scenes.py` is the kids path | Superseded by **`pipeline/kidsong/`** (ACE-Step sings → Whisper alignment → librosa beats → director shot list → per-shot render → 4 blocking QC gates → beat-aligned edit) |
> | Swap edge-tts → Kokoro TTS | Still open, but only for the **legacy** Shorts path; kidsong does not use TTS at all (ACE-Step sings the lyrics) |
> | "Cocomelon-style" as the target look | **Reversed.** Look-alike drift toward CoComelon is now a ship-blocking S1 finding — see `docs/quality/SOURCES.md` (Moonbug v. Babybus) and QM-011 |
>
> Live documentation: **`AGENTS.md`** (architecture, workflows, conventions) and
> **`docs/quality/`** (defect backlog, improvement log, sources).

# Cocomelon on the RTX 5070 — continuation notes (2026-07-11)

State when this was pushed, and what to do next on the 12GB machine.

## Where we are
- Pipeline works end-to-end. Kids/Cocomelon path is real: `prompts/kids.txt` (sing-song
  script + per-line `visual` + recurring character) → `studio/scenes.py` (one AI clip per
  scene via ComfyUI) → `assemble.build_video`.
- On the laptop (RTX 4060 8GB) we generated a full kids video with **LTX-Video 0.9.5 (2B)**.
  Verdict: characters are **creepy / incoherent** — that is the 2B model's ceiling, not a bug.
- Decision: move rendering to the **RTX 5070 (12GB)** and upgrade the model to **LTX-2**
  (FP8), which fits 12GB and is a large jump in coherence. (Wan 2.2 14B GGUF is plan B —
  slightly better characters but stuck ~480p on 12GB and more setup.)

## config.json is NOT in the repo (gitignored)
Recreate on the 5070: `copy config.example.json config.json`, then set for the 5070:
- `llm.backend` = groq (needs `GROQ_API_KEY`) or ollama.
- `whisper.device` = `cuda`, `compute_type` = `float16` (5070 has the VRAM).
- `comfyui.enabled` = `true`, and point `comfyui.workflow_path` at the LTX-2 workflow (below).
- API keys live in `keys/` and `.env` (also gitignored) — copy them over manually / re-enter
  via the Setup page.

## LTX-2 setup on the 5070 (ComfyUI)
1. Fix the node that failed to import on the laptop: `ComfyUI-LTXVideo` threw
   `ImportError: cannot import name 'pad' from kornia.geometry.transform.pyramid`
   → update the node + pin a compatible `kornia` in the comfyui venv (`pip install -U kornia`
   or the version the node's requirements specify). Update the node itself to the LTX-2 release.
2. Get the **LTX-2 FP8** checkpoint (fits 12GB; GGUF variants exist for more headroom).
   Install via ComfyUI Manager → search "LTXVideo", or drop the model in
   `comfyui/models/checkpoints/`. Confirmed viable on 12GB per current guides
   (FP8 = 12GB, bf16 = 24GB).
3. Update `studio/comfy_workflows/ltx_t2v.json` (and `config.json comfyui.*`: `frames`,
   `width`/`height`, `prompt_node_title`) to the LTX-2 workflow shape. Keep resolution to a
   12GB-safe draft (e.g. 704×1216-ish, modest frame count) and upscale later if needed.
4. Start ComfyUI: `scripts/start-comfyui.ps1` (it checks the model is present first).
5. Test: `venv\Scripts\python -m pipeline.generate kids --style kids` — expect coherent
   animated scenes instead of the creepy 2B output.

## Still open (was next on the list)
- **Voice**: swap edge-tts → **Kokoro TTS** (chosen; free, local, natural). Not wired yet.
  Add a Kokoro backend in `pipeline/tts.py` behind a `voices.engine` switch. Runs on the 5070
  GPU or CPU. This is independent of the video-model work.
- The laptop's `config.json` currently has `comfyui.enabled=true`, `whisper=cpu/int8` — laptop
  4060 is fine for testing the pipeline but NOT for LTX-2 (needs 12GB).

## Reminder
Real Cocomelon is pro 3D animation; even LTX-2/Wan are "AI video" and won't perfectly match
it. LTX-2 on the 5070 is the biggest realistic local jump. If characters still disappoint,
the fallbacks are Wan 2.2, locked-character image→video (Flux → I2V), or paid cloud (Kling/Veo).
