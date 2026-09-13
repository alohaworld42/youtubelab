# Plan: Cast-Konsistenz über Shots (Phase 1 = Beweis an 3 Shots)

## Kontext

Problem: dasselbe Kind sieht in jedem Shot anders aus, Fremdkinder tauchen auf.
Ursache (bestätigt im Code + an den Keyframes von `…-wan22-toon-v4`):
- Jeder Shot = frischer **text-only** Z-Image-Keyframe → Wan 2.2 5B i2v. Kein Pixel-Anker.
- `kidsong.keyframe_first.identity_refs=false`, `reference_anchor.enabled=false` → `canonical.png`-Refs werden **nirgends** benutzt.
- `workflows/flux2_klein_ref.json` verweist auf `flux-2-klein-4b-fp8.safetensors` → nicht auf Platte (nur `flux-2-klein-4b.safetensors` bf16) → Klein-Anker-Pfad auf dem 0.34-Server kaputt.
- `identity_check` (CLIP-Gate) still tot: Repo-venv hat kein torch. `kidsong.shot.height=720` → 704 gerundet.

Entscheidender Befund (Metadaten der 3 „Beweis"-Videos `C:\Users\Aloha\AppData\Local\Comfy-Desktop\ComfyUI-Shared\output\comfy_test\h3_r2v_*.mp4`):
- Die H3-R2V-Graphen senden `"ref_images": [[24,0],…]` (Liste). ComfyUI 0.34.3 erwartet für Autogrow-Inputs **gepunktete Keys** `"ref_images.ref_image_0": ["24",0]` (`comfy_api/latest/_io.py::Autogrow._expand_schema_for_dynamic`, `build_nested_inputs`; offizielles Template `video_minimax_h3_r2v.json`). Listenform → Refs **stillschweigend ignoriert**, LoadImage-Nodes nie ausgeführt.
- → Alle bisherigen „R2V perfect lock"-Renders liefen **ohne Referenzen** (Solo sah nur wegen des Wardrobe-Prompts wie Zuri aus; Gruppe zeigt 3 Nicht-Cast-Kinder). `docs/H3_PIPELINE.md` ist hier falsch. R2V-mit-Refs wurde auf dieser Maschine **noch nie** getestet.
- Auf Platte liegt nur `minimax_h3_fl2va_pruned-Q4_K.gguf` (First/Last-Frame-Variante) + `fl2v_turbo_8step`-LoRA; das offizielle R2V-Template nutzt `ref2va`-Gewichte + `ref2v_turbo`. Der DiT-Code (`comfy\ldm\minimax\model.py`) konsumiert Refs mit beliebigen H3-Gewichten, fl2va ist aber nicht auf Ref-Blöcke trainiert → Ref-Wirkung muss gemessen werden (Probe, s. Stage 2).

Umgebung (verifiziert): RTX 4060 Laptop 8 GB, 15,6 GB RAM, Treiber 616.56, ComfyUI 0.34.3 in `C:\Users\Aloha\Desktop\Projects\youtubegenerator\comfyui\models\ComfyUI (1)\ComfyUI` (`.venv`: torch 2.12.1+cu126, PIL, requests, transformers 5.14, torchvision, huggingface_hub, psutil; kein triton/sage), GGUF-Loader v2.0.0 (H3-gepatcht, **kein `flux2`-Arch** → Klein-GGUF nicht ladbar). Server läuft (pid 34528) mit `--cache-none --disable-pinned-memory --reserve-vram 0.8 --extra-model-paths-config h3_model_paths.yaml`; Input-Dir `<ComfyUI>\input`, Output `<ComfyUI>\output`. Live-Code = Worktree `C:\Users\Aloha\orca\workspaces\youtubegenerator\Youtubegenerator`. ffmpeg: `…\youtubegenerator\comfyui\venv\Lib\site-packages\imageio_ffmpeg\binaries\ffmpeg-win-x86_64-v7.1.exe`. 181 GB frei.
GPU: v4-Job (pid 23632) laut Prüfung bereits tot (10/22 Shots), `/queue` leer, `output\gpu_render.lock` veraltet → User-Entscheidung „jetzt stoppen": bei Start prüfen, lebenden Prozess beenden, Lock via `pipeline.gpu_lock.GPULock.try_acquire()` (steals stale) übernehmen. v4-Outputs bleiben unangetastet.

Recherche (2 Workflows, 30 Agents, Kandidaten adversarial geprüft) — Ergebnis:
Kein Video-Modell auf dieser Box kann Identität tragen (14B-Klasse VACE/Phantom/S2V/Animate/MultiTalk: an der **RAM**-Wand refuted). → Identität wird im **Bild** entschieden (Keyframe aus festen Refs + Mess-Gate); einzige echte Referenz-Video-Option = MiniMax-H3 R2V.
| Kandidat | Identität 3 Toon-Kinder über Shots | 8 GB + 15,6 GB RAM | Rolle |
|---|---|---|---|
| **Qwen-Image-Edit-2511 GGUF** (`TextEncodeQwenImageEditPlus`, 3 Bild-Slots) | stärkste Belege (Changelog: character consistency, 2-Personen-Fusion; Pixar-3D-LoRA „preserves identity") | Q3_K_S 9,0 + VL-7B Q4 4,7 + mmproj 1,4 GB; TE/DiT nie gleichzeitig → plausibel, unbewiesen; Q2_K 7,1 GB Ventil | Keyframe-Engine K2 |
| **FLUX.2 Klein 4B + ReferenceLatent-Kette** | Multi-Ref = Schwachpunkt (Gesichter/Kleidung driften bei 2–3 Refs) | sicher: bf16 7,75 GB auf Platte via `UNETLoader weight_dtype=fp8_e4m3fn` + `qwen_3_4b` (lief so bereits, IMP-014) | Keyframe-Engine K1 |
| **MiniMax-H3 R2V** (Refs korrekt verdrahtet) | supported (Template 2 Refs, `<Picture N>`-Tags); keine Toon-Negativberichte | läuft; 13 min @0,45 MP ohne Refs; 1344x768 mit Refs geschätzt 20–45 min/Clip | Video-Engine V2 (User-Wahl) |
| Wan 2.2 TI2V-5B i2v | hält Identität nur **innerhalb** eines Clips | 12–17 min @1280x704 | V1 — vom User für Phase 1 abgewählt |
| DINOv2-Cosine auf **Crops** | domain-agnostisch; Vollbild vom Hintergrund dominiert → Crops Pflicht | trivial | QC-Gate |
| LTX-2.3/2.5, ID-LoRA, MSR, HunyuanVideo 1.5, Character-LoRA-Training lokal, MultiGPU/DisTorch, iGPU-Vulkan | refuted (RAM/VRAM/single-speaker; iGPU = dieselben 15,6 GB UMA) | — | raus |
! Größter Hebel lt. Recherche: RAM 16 → 32/64 GB (SO-DIMM) würde fast alle refuted 14B-Kandidaten öffnen. Nicht Teil des Plans.
! Audio/Lipsync für 3 Toon-Kinder: kein 8-GB-Pfad → Song wird weiter gedubbt; nicht Teil von Phase 1.
Regeln aus `tasks/lessons.md`: Identität über **Pixel**; Ergebnisse als **Artifact-URL** (Videos als Assets).

## Ziel Phase 1 (User-Entscheidungen eingearbeitet)

Lab in `C:\Users\Aloha\Desktop\Projects\Youtube\`. Keyframes K1 + K2. Video **nur V2 = H3 R2V** (mit Refs, erstmals). 3 Shots: s01 Zuri solo Park · s02 Zuri solo Küche · s03 Zuri+Kofi+Nala im selben Park wie s01. Feste Refs + feste Seeds, 1280x720 echt 16:9, mp4 + Handy (<30 MB) + Kontaktbogen + Metrik. Erfolg = s01↔s02 dasselbe Kind, s03 = genau die 3 Cast-Kinder mit den Gesichtern aus s01, keine Fremden.

## Ordnerstruktur (agentbasiert)

```
C:\Users\Aloha\Desktop\Projects\Youtube\
  README.md                 Startbefehle (volle Pfade), Contract, gemessene Zeiten je Stage
  run_proof.ps1             EINSTIEG: -Stage 0|1|2|3|4|all  -Shots s01,s02,s03  -RunId <id>  -Resume
  lab.json                  Pfade (comfy_dir, python=.venv, orca_root, ffmpeg, lock_file, hf_home), Dims, Seeds, Timeouts
  cast\                     FESTER Referenz-Satz (nie neu würfeln)
    cast.json               id, name, seed, Wardrobe-Text (aus prompts\cast_bible.json: age skin hair top bottom shoes build)
    {zuri,kofi,nala}\canonical.png        Byte-Kopie aus output\_cast_refs
    {zuri,kofi,nala}\ref_576x768.png      ImageOps.fit 3:4, weißer Hintergrund, kein Alpha
  agents\
    common\comfy_lab.py     LabClient(ComfyClient) + Graph-Builder + Lock/Queue/State-Helfer
    common\shots.py         Shot-Spec, Prompt-Bauer (`<Picture N> is <Name>: …`, Count-Clamp, „Image N is …" für Keyframes)
    00_setup\setup_models.py      Downloads (Resume, Größen-Log, nie überschreiben), Cast-Kit, Graph-Validierung, DINO-Warmup
    01_keyframe\klein_agent.py    K1 → keyframes\K1\<shot>_1344x768.png
    01_keyframe\qwen_edit_agent.py K2 → keyframes\K2\<shot>_1344x768.png
    01_keyframe\workflows\klein4b_ref{1,2,3}.json, qwen_edit_2511_ref{1,2,3}.json
    02_video\h3_r2v_agent.py      V2 → clips_V2\<shot>_V2.mp4 (+ s01_norefs, --probe, --guide <keyframe>)
    02_video\workflows\h3_r2v_ref{1,2,3}.json
    03_qc\identity_qc.py          DINOv2-small Crop-Scores → qc\scores.json, qc\report.md, qc\contact_*.png
    04_assemble\assemble.py       1344x768 → 1280x720, h264 crf 18; phone crf 28; A/B-Side-by-Side
  runs\<RunId>\             shots.json  state.json  keyframes\  clips_V2\  qc\  final\  phone\  run.log
  logs\                     PS-Wrapper out/err getrennt
  Claudetempfiles\          Temp (Frames, Zwischenbilder)
  .hf_cache\                dinov2-small
```
Contract: jeder Agent = eigenes CLI (`--run <id> [--shots] [--resume]`), liest `runs\<id>\shots.json`, schreibt nur in seinen Ordner, `state.json` append-only via `pipeline.atomicio.atomic_write_json`. Fertig = Datei existiert und >0 Byte. Nichts wird gelöscht (nur eigene `input\`-Staging-Kopien aufgeräumt).

## Stages des Beweislaufs

Stage 0 — Setup (~20–40 min, kein GPU):
- K1: kein Download (bf16 auf Platte). Fallback bei RAM-Thrash: `qwen_3_4b_fp4_flux2.safetensors` 3,85 GB (Comfy-Org/flux2-klein) → `text_encoders\`.
- K2 (~16 GB): `Qwen-Image-Edit-2511-Q3_K_S.gguf` 9,0 GB → `comfyui\models\unet\`; `Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf` 4,7 GB + `Qwen2.5-VL-7B-Instruct-mmproj-BF16.gguf` 1,4 GB → `text_encoders\`; `qwen_image_vae.safetensors` 0,25 GB → `vae\`; `Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors` 0,85 GB → `loras\`. Q2_K 7,1 GB nur bei Thrash.
- QC: `facebook/dinov2-small` (~90 MB) → `.hf_cache` (Warmup vor den Renders).
- Cast-Kit, Graphen per `/object_info` validieren, 1 Mini-Test je Keyframe-Engine (512x288) → beweist Loader/mmproj-Wiring vor der echten Stage.

Stage 1 — Keyframes (~20–40 min GPU): K1 und K2 rendern je s01/s02/s03 bei **1344x768** (H3-Canvas, Klein /16 ✓), Seeds fix, Refs 576x768 (K1: 1 ReferenceLatent je anwesendes Kind, Graph programmatisch ohne Dummy-Refs; K2: image1..3, Lightning 4 Steps cfg 1). Danach QC-Crop-Scores + Kontaktbogen → Gewinner je Shot (Score; User kann in `state.json` überstimmen). Abbruch, wenn Gesichter schon hier driften.

Stage 2 — H3-Probe (~10 min GPU): s01 bei 864x480x39, 8 Steps, **mit** vs **ohne** Refs (identischer Seed) → DINO `id_zuri`-Differenz. Δ < 0,05 → fl2va ignoriert Refs → erst `minimax-h3-ref2va` pruned Q4 (11,4 GB) + `ref2v_turbo_8step_v1.0_768p` LoRA (~1,9 GB) laden, Probe wiederholen. Kein 1344x768-Render vor bestandener Probe.

Stage 3 — Video V2 (~1,5–3 h GPU): H3 R2V 1344x768, length **90** (17·5+5, 3,75 s @24 fps; kürzer als 124 → −27 % Tokens), res_multistep 8 Steps, Scheduler `beta`, Turbo-LoRA 1.0, `ref_image_size=match`, Refs nur der anwesenden Kinder (`ref_images.ref_image_0..N-1`). Clips: s01, s02, s03 + Kontrolle `s01_norefs`; optional s03 zusätzlich mit Gewinner-Keyframe als Frame-0-Guide (`MiniMaxH3AddGuide` frame_idx 0) → Komposition aus dem Keyframe, Identität aus den Refs. `render_timeout` 4800 s.

Stage 4 — QC + Auslieferung (~5 min): Scores (vs. canonical · Drift im Clip · Cross-Shot s01↔s02 · s01↔s03 · Negativkontrolle = altes v4-`s05`-Keyframe), Kontaktbogen (6 Frames/Clip), `final\*.mp4` (`crop=1344:756:0:6,scale=1280:720:flags=lanczos`, libx264 crf 18, aac), `phone\*.mp4` (crf 28, Größe asserten), Artifact-Seite (Videos als Assets, Tabelle, Kontaktbogen).

Prompt-Regeln: H3-Tokenizer setzt `<Picture i>` in Verbindungsreihenfolge (`text_encoders\minimax.py`): „`<Picture 1>` is Zuri, a 3-year-old girl with deep warm brown skin, round afro puffs … Keep each child's face, hair, skin tone and outfit exactly as in their picture. Exactly N children appear: <Namen>; no other children, no adults. SCENE/ACTION/CAMERA/AUDIO …"; NEGATIVE via `StringConcatenate "\n\nDo not show: "` ≤ 12 Wörter. Keyframes: „Image N is <Name> (<wardrobe>). Using exactly these N children with their exact faces… <Szene>".

## Schritte (Implementierung)

1. Gerüst: Ordner, `lab.json`, `README.md`, `cast\` (Seeds `20260717 + seed_offset`: zuri 21273730, kofi 21274744, nala 21275758; Shot-Seed = Kind-Seed + idx·100; Kontrolle = Seed von s01).
2. `agents\common\comfy_lab.py`: `sys.path.insert(0, orca_root)`; `from pipeline.kidsong.comfy import ComfyClient, ComfyUnreachableError`; `from pipeline.gpu_lock import GPULock`; `from pipeline.atomicio import atomic_write_json`. `LabClient(ComfyClient)`: `__init__` mit `{"comfy": {url, path, "autostart": False, render_timeout}, "_root": LAB}`; `load_workflow(name_or_dict)` (dict → deepcopy, str → `agents\..\workflows\`), damit `render()` unverändert gebaute Graphen nimmt; `queue_idle()` (`/queue` leer), `render_retry()` (`free()` + 1 Retry bei Timeout/Unreachable), `acquire_gpu()` (GPULock, stale-steal, Release im `finally`). Wiederverwendet: `render/_poll_history/_retrieve/stage_input_image/cleanup_staged/free`.
3. `setup_models.py` (huggingface_hub `hf_hub_download` mit Resume, Ziel = `comfyui\models\…`, vorhandene Dateien nie überschreiben, Größen ins Log), Cast-Kit-Bauer (PIL), Validierung, Mini-Tests.
4. Graphen: `klein4b_ref{1,2,3}.json` aus `flux2_klein_ref.json` (Node 1 `UNETLoader flux-2-klein-4b.safetensors fp8_e4m3fn`, 1344x768, Kette `LoadImage→VAEEncode→ReferenceLatent` je Ref, `BasicGuider`, `Flux2Scheduler 4`); `qwen_edit_2511_ref{1,2,3}.json` aus dem Comfy-Template (UnetLoaderGGUF + CLIPLoaderGGUF `qwen_image` + mmproj, `TextEncodeQwenImageEditPlus`, Lightning-LoRA, 4 Steps cfg 1); `h3_r2v_ref{1,2,3}.json` aus `minimax_h3_r2v_group_gguf.json` (Listen-Key raus, gepunktete Keys rein, unbenutzte LoadImage-Nodes entfernt, `beta`, WIDTH/HEIGHT/FRAMES-Titel bleiben).
5. Agenten 1–4 + `shots.json`.
6. `run_proof.ps1`: `$ErrorActionPreference="Continue"`, `$env:PYTHONUTF8="1"`, `[Console]::OutputEncoding=UTF8`, Python = ComfyUI-`.venv`, `& $py @args 2>> logs\proof.err.log | Tee-Object -Append logs\proof.out.log`, `$LASTEXITCODE` je Step, keine Umlaute; Server-Check → Start via `Start-Process $py -ArgumentList "main.py --listen 127.0.0.1 --port 8188 --disable-auto-launch --cache-none --disable-pinned-memory --reserve-vram 0.8 --extra-model-paths-config h3_model_paths.yaml" -WorkingDirectory $Comfy` (nicht `setup_h3_comfy.ps1`: git pull + pip je Lauf); v4-Prozess-Check/Stop; `-Resume` überspringt vorhandene Dateien; `-Stage` schaltet Etappen; Engine-Phasen nie verschränken (`--cache-none` lädt je Prompt von Platte). Aufruf:
   `& "C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe" -ExecutionPolicy Bypass -File "C:\Users\Aloha\Desktop\Projects\Youtube\run_proof.ps1" -Stage all -RunId proof01`
7. End-to-end selbst laufen lassen (Stage 0 → 1 → 2 → 3 → 4), Kontaktbögen ansehen, Artifact publizieren, Zeiten in `README.md` eintragen, Kurzbericht im Chat (Pfade + URL).

## Verifikation

- `run_proof.ps1 -Stage all -RunId proof01` läuft ohne Terminating Error; `runs\proof01\final\` enthält `s01_V2.mp4 s02_V2.mp4 s03_V2.mp4 s01_norefs_V2.mp4` (1280x720, 24 fps, h264), `phone\` < 30 MB je Datei; `keyframes\K1|K2\` je 3 PNG 1344x768.
- Kill mitten im Lauf + `-Resume` → fertige Dateien bleiben, nur fehlende werden gerendert.
- `qc\report.md`: `s01_V2` id_zuri deutlich > `s01_norefs_V2` (Refs wirken); Cross-Shot s01↔s02, s01↔s03 (Zuri-Crop) deutlich über Negativkontrolle; s03 zeigt genau 3 Kinder = Cast.
- Sichtprüfung Kontaktbogen: Haar/Outfit/Hautton/Proportion identisch über s01/s02/s03.
- `…\ComfyUI\user\comfyui.log`: kein `TypeError`, kein OOM; Zeit je Bild/Clip in `run.log`.

## Risiken (geordnet) + billiger Check

1. ! fl2va ignoriert Refs → Stage-2-Probe (2 × ~3 min) entscheidet über ref2va-Download.
2. ! H3-Zeit bei 1344x768x90 unbekannt (Schätzung 20–45 min/Clip) → Probe-Zeit hochrechnen, bei > 45 min auf 1280x736 (0,94 MP) + 8-px-Crop wechseln.
3. ! Klein Multi-Ref-Bleeding (Kofi in Zuris Shirt) → s03-Keyframe im Kontaktbogen prüfen, 1 Reseed erlaubt; K2 als Gegenprobe.
4. ! RAM: `qwen_3_4b` bf16 8 GB + Klein 7,75 GB → `comfyui.log` nach Keyframe 1 auf OOM/Thrash prüfen → fp4-TE laden.
5. ! K2 mmproj/CLIPLoaderGGUF-Wiring unverifiziert → Stage-0-Mini-Test.
6. ? DINO-Schwelle für Toon-Kinder unbekannt → Phase 1 nur messen + Negativkontrolle; Schwelle für Phase-2-Gate daraus ableiten.

## Nicht in Phase 1 (Phase 2 nach Freigabe)

Integration des Siegers in `pipeline\kidsong` (neuer `render_style` H3-R2V bzw. Keyframe+Video, `identity_refs` an, DINO-Reseed-Gate statt totem CLIP-Gate, 720-Konformierung in `edit.assemble`), Multi-View-Sheet pro Kind, Wan-V1 als Tempo-Option, Turbo-Steps, volle Episode, Song/ACE, Lipsync.
