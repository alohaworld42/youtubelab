# youtubelab

Mono-Repo für die automatische YouTube-Inhaltserstellung: Cast-konsistente Kinderlieder-Videos mit KI.

## Struktur

| Pfad | Beschreibung |
|------|-------------|
| `pipeline/` | Kern: Kidsong-Pipeline (Lyrics → Song → Shots → Video → Reel) |
| `studio/` | Web-Studio UI (Dashboard, Review, Channels) |
| `scripts/` | Batch-/Hilfsskripte |
| `prompts/` | LLM-Prompts (Cast-Bible, Songwriting, Director) |
| `templates/` | Web-UI Templates |
| `workflows/` | ComfyUI-Workflows (LTX, Klein, H3, Qwen-Edit) |
| `tests/` | Test-Suite (pytest) |
| `assets/` | Logos, Stile, abhängige Assets |
| `lab/` | **Consistency-Lab** — H3-Phased-Agenten, Keyframe-Workflows, Identity-QC, Bootstrap |
| `h3_r2v/` | **H3 Reference-to-Video** — funktionierender Beat-Render-Flow (MiniMax-H3, Abiray-GGUF, Native CLIPLoader) |
| `cast_refs/` | Canonical-Referenzbilder (Zuri, Kofi, Nala) |

## Schnellstart

### Prod (Kidsong-Pipeline)
```bash
# ComfyUI starten, dann:
python app.py
```

### H3 Reference-to-Video (Beweis-Render)
```powershell
# ComfyUI muss laufen (nicht auto-starten):
& .\h3_r2v\run_proof.ps1 -DryRun    # Prüfen
& .\h3_r2v\run_proof.ps1 -Resume    # Alles rendern (butt) - Resume-Modus
& .\h3_r2v\run_proof.ps1 -Reel      # Reel zusammenschneiden (braucht ffmpeg)
```

### Consistency-Lab (Phased Pipeline)
```powershell
# Stage 0 → 4 via lab.json (Pfade werden via bootstrap/configure.py gesetzt):
& .\lab\run_proof.ps1 -Stage 0      # Setup: Modelle laden
& .\lab\run_proof.ps1 -Stage all    # Kompletter Lauf
```

## Setup-Voraussetzungen

- **GPU:** 8 GB VRAM (MiniMax H3 läuft mit ~5.9 GB)
- **Python:** 3.10+ mit torch (CUDA)
- **ComfyUI:** v0.35+ mit `gguf`-Paket installiert
- **Modelle** (unter ComfyUI-Modellpfad):
  - `diffusion_models/MiniMax-H3-Ref2VA-Pruned-Q4_K_M.gguf` (Abiray)
  - `text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` (Comfy-Org)
  - `vae/minimax_h3_video_vae_fp16.safetensors`, `vae/minimax_h3_audio_vae_fp32.safetensors`

## Neuer PC / Setup

Auf einem frischen Windows-PC genügt:

```powershell
git clone <repo-url>
powershell -ExecutionPolicy Bypass -File setup\download_assets.ps1
```

Das Skript lädt idempotent alle großen Dateien, die **nicht** im Git liegen (`.gitignore: comfyui/, latentsync/, espeak-ng/`):

- **ComfyUI Modelle** (>20 MB): Checkpoints, Diffusion Models, LoRAs, Text Encoders, VAEs, Upscale Models, TTS/DiffRhythm Models
- **ComfyUI Portable** (Windows NVIDIA CUDA 12.1 Release)
- **LatentSync** Repo + Checkpoints (UNet, Whisper, InsightFace Buffalo_L)
- **espeak-ng** MSI + administrative Entpackung

Optionen:
- `-DryRun` — listet nur, was fehlt + Gesamtgröße
- `-Only comfyui|latentsync|espeak|comfyui_portable` — nur eine Komponente

Voraussetzungen: `curl.exe` (Windows 10+), `git`, `7z` (7-Zip), `msiexec`.

## Wichtige Erkenntnisse (GGUF-Kompatibilität)

MiniMax-H3 GGUFs von **unsloth** und **leejet** haben `kv_count=0` (keine Metadaten) → **unbrauchbar** für `CLIPLoaderGGUF`/`UnetLoaderGGUF`.  
Funktionierend: **Abiray/MiniMax-H3-Pruned-GGUF** (Metadaten vorhanden, kv=60) + **nativer CLIPLoader** (`type="minimax"`) mit safetensors-Text-Encoder.

## Autoren

- `alohaworld42`
