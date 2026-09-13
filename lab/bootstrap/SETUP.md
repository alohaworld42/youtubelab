# Neuaufsetzen auf einem anderen Rechner (ohne Claude)

Ziel: das Konsistenz-Lab lauffaehig machen. Modelle (~85 GB) liegen nicht im
Repo, werden per `bootstrap.py` geladen. Reihenfolge strikt einhalten.

## 0. Voraussetzungen
- Windows, NVIDIA-GPU (>= 8 GB), moeglichst >= 16 GB RAM. Mehr RAM = deutlich schneller (H3 laedt sonst staendig zwischen RAM/VRAM um).
- Git, ~120 GB freier Plattenplatz.
- ComfyUI Desktop (>= 0.34.3, hat native MiniMax-H3-Nodes). Merke dir seinen Ordner (enthaelt `main.py`) und sein venv-Python (`<ComfyUI>\.venv\Scripts\python.exe`).

## 1. torch auf +cu126 pruefen (WICHTIG)
ComfyUI Desktop bringt oft `torch+cu130`. Wenn der Treiber < CUDA 12.8 kann, gibt es `access violation` beim Start. Fix (macht der Code-Repo automatisch):
```
git clone -b claude/vram-budget-diffusion-ft5aoa https://github.com/alohaworld42/youtubegenerator.git
& "<ComfyUI>\.venv\Scripts\python.exe" -c "import torch;print(torch.__version__, torch.cuda.is_available())"
```
Zeigt es `False` oder `+cu130`: `youtubegenerator\tools\setup_h3_comfy.ps1 -Comfy "<ComfyUI>" -NoStart` reinstalliert dieselben Versionen als `+cu126`.

## 2. Lab-Repo holen
```
git clone https://github.com/alohaworld42/youtube-consistency-lab.git
cd youtube-consistency-lab
```

## 3. Bootstrap (Nodes, Code-Repo, Patch, Modelle, dino)
Mit dem ComfyUI-venv-Python starten:
```
& "<ComfyUI>\.venv\Scripts\python.exe" bootstrap\bootstrap.py --comfy "<ComfyUI>" --models "<models_dir>" --orca "<clone-ziel youtubegenerator>"
```
- `<models_dir>` = wohin die Modelle sollen (viel Platz). Legt `unet/ text_encoders/ vae/ loras/ diffusion_models/ upscale_models/` an.
- Laedt ~85 GB (Resume-faehig, nie ueberschreiben). Dauer je nach Leitung.
- Klont ComfyUI-GGUF + h3_cond_cache nach `<ComfyUI>\custom_nodes`, wendet den H3-GGUF-Patch an.
- Wenn `--orca` schon aus Schritt 1 existiert: wird wiederverwendet.

## 4. lab.json fuer diesen Rechner schreiben
```
& "<ComfyUI>\.venv\Scripts\python.exe" bootstrap\configure.py --comfy "<ComfyUI>" --models "<models_dir>" --orca "<clone-ziel youtubegenerator>"
```
Prueft jeden Pfad (OK/!!). ffmpeg wird automatisch gesucht; sonst `--ffmpeg "<pfad>"`.

## 5. Lauf starten
```
& "C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe" -ExecutionPolicy Bypass -File "<lab>\run_proof.ps1" -Stage all -RunId proof01
```
`run_proof.ps1` startet den ComfyUI-Server selbst (Flags aus `lab.json.comfy_args`) und faehrt Stage 0-4. Einzelstufen: `-Stage 1` / `-Stage 3` usw. `-Resume` ist Standard.

## Stand & offene Punkte (aus der Session)
- Bewiesen: H3 Reference-to-Video haelt die 3 Cast-Kinder ueber Shots identisch (`runs/proof01/final/s01_V2.mp4`, `s02_V2.mp4`, `s03_V2.mp4`). Der Fix war die gepunktete Autogrow-Verdrahtung `ref_images.ref_image_N` (siehe `agents/02_video/h3_r2v_agent.py`).
- Geliebte Qualitaet = 8-Step-Turbo bei 1344x768x73 (~15 min/Clip). Weniger Steps = Qualitaetsverlust (nicht gewuenscht).
- Verlustfreier Conditioning-Cache validiert (PSNR unendlich): `agents/02_video/h3_phased_agent.py` (encode 1x / sample 1x / decode) — spart aber nur ~5-10%, weil Sampling die Wand ist.
- Naechster offener Test (abgebrochen): klein rendern (864x480) + Upscale (`agents/04_assemble/upscale_video.py`, `4x-UltraSharp`) -> potenziell ~2x, Qualitaet am Bild pruefen. SageAttention (triton-windows installiert) = ~1,3x verlustfrei, braucht Server-Neustart mit `--use-sage-attention`.
- Episoden-Tempo-Hebel ohne Qualitaetsverlust: kurze Beat-Clips (39/56 statt 73 Frames) + Refrain-Wiederverwendung.
- Voller Plan: `docs/PLAN.md`.
