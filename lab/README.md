# Konsistenz-Lab (Cast: Zuri, Kofi, Nala)

Ziel: dieselben 3 Kinder in jedem Shot, referenz-konditioniert, 1280x720.
Phase 1 = Beweis an 3 Shots (s01 Zuri Park, s02 Zuri Kueche, s03 alle 3 im Park).

## Start (voller Pfad, PowerShell 5.1)

```
& "C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe" -ExecutionPolicy Bypass -File "C:\Users\Aloha\Desktop\Projects\Youtube\run_proof.ps1" -Stage all -RunId proof01
```

Stages (einzeln: `-Stage 1` oder `-Stage 2,3`):

| Stage | Was | Dauer (RTX 4060 8 GB) |
|---|---|---|
| 0 | Modelle laden (nur fehlende), DINOv2 warmup, Graphen validieren | 20-40 min, kein GPU |
| 1 | Keyframes K1 (FLUX.2 Klein 4B) + K2 (Qwen-Image-Edit-2511) fuer alle Shots, QC-Kontaktbogen | K1 ~20 s/Bild |
| 2 | H3-Probe: s01 864x480x39 MIT vs OHNE Refs (zeigt, ob Refs wirken) | 2 x ~5 min |
| 3 | H3 Reference-to-Video 1344x768x73: s01, s02, s03 + Kontrolle s01_norefs | 20-45 min/Clip |
| 4 | QC (DINOv2-Scores, Kontaktbogen, report.md), 1280x720-Konformierung, Handy-Version, Side-by-Side | ~5 min |

Optionen: `-Shots s01,s02` · `-Guide K1` (Stage 3 mit Keyframe als Frame-0-Anker) · `-NoResume` · `-KillLockHolder`.
Resume ist Standard: vorhandene Dateien werden uebersprungen. Es wird nichts geloescht.

## Ordner

```
lab.json                    Pfade, Modelle, Aufloesungen, Timeouts, Download-Liste
cast\                       FESTER Referenz-Satz (nie neu wuerfeln): cast.json + <kind>\canonical.png, ref_576x768.png, ref_768x1024_white.png
agents\00_setup\            setup_models.py  (--download --warmup --validate)
agents\01_keyframe\         klein_agent.py (K1), qwen_edit_agent.py (K2), workflows\*.json (Beispielgraphen, --dump-workflows)
agents\02_video\            h3_r2v_agent.py (V2, --probe, --guide), workflows\h3_r2v_ref{1,2,3}.json
agents\03_qc\               identity_qc.py  (DINOv2-small, Crop-Cosine, Kontaktbogen)
agents\04_assemble\         assemble.py     (Crop/Scale 1280x720, crf 18; Handy crf 28; Side-by-Side)
agents\common\              comfy_lab.py (LabClient auf pipeline.kidsong.comfy.ComfyClient), shots.py (Shots, Prompts, Seeds)
runs\<RunId>\               shots.json, state.json (append-only), graphs\ (gesendete API-Graphen), keyframes\K1|K2\, clips_V2\, qc\, final\, phone\, run.log
logs\                       PS-Wrapper (out/err getrennt), ComfyUI-Serverlog, Setup-Log
Claudetempfiles\            Temp (Downloads, Frames)
```

Jeder Agent laeuft auch einzeln, z. B.
```
& "C:\Users\Aloha\Desktop\Projects\youtubegenerator\comfyui\models\ComfyUI (1)\ComfyUI\.venv\Scripts\python.exe" "C:\Users\Aloha\Desktop\Projects\Youtube\agents\01_keyframe\klein_agent.py" --run proof01 --shots s03 --resume
```
(Server muss laufen; `run_proof.ps1` startet ihn sonst mit den 8-GB-Flags.)

## Technik-Notizen

- ComfyUI 0.34.3: Autogrow-Inputs (H3 `ref_images`) im API-Prompt als **gepunktete Keys** `ref_images.ref_image_0` senden. Eine Liste unter `ref_images` wird stillschweigend ignoriert (so liefen die alten "R2V-Beweise" ohne Refs).
- Aufloesung: H3 und Wan brauchen Vielfache von 32 -> 1344x768 (H3-Canvas) rendern, dann Crop 1344x756 + Scale 1280x720 (assemble.py).
- Klein 4B GGUF ist mit dem installierten (H3-gepatchten) GGUF-Loader nicht ladbar -> bf16-Datei mit `weight_dtype=fp8_e4m3fn`.
- GPU-Lock: `output\gpu_render.lock` der Pipeline wird respektiert (lebender Halter -> Abbruch), tote Locks werden uebernommen.
- Seeds: Kind-Seed = 20260717 + seed_offset (cast_bible); Shot-Seed = Kind-Seed + Shotnummer*100; Gruppe = Basis + Summe der Offsets; Kontrolle erbt den Seed.

## Auf einem anderen Rechner neu aufsetzen

Modelle (~85 GB) sind nicht im Repo. Rekonstruktion ohne Claude:
1. `bootstrap/SETUP.md` befolgen (ComfyUI + torch cu126 pruefen).
2. `python bootstrap/bootstrap.py --comfy "<ComfyUI>" --models "<models_dir>" --orca "<youtubegenerator-clone>"` — laedt Modelle, klont Custom-Nodes + Code-Repo, wendet den GGUF-Patch an.
3. `python bootstrap/configure.py --comfy ... --models ... --orca ...` — schreibt `lab.json` fuer diesen Rechner.
4. `run_proof.ps1 -Stage all -RunId proof01`.
