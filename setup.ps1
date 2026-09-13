<# ============================================================================
  setup.ps1 - Einmalige Einrichtung von youtubelab (H3 Ref2VA Video-Pipeline)

  Was das Skript macht:
    1. ComfyUI finden (Parameter > Auto-Erkennung) oder bei -InstallComfyUI
       frisch klonen inkl. venv, torch + gguf.
    2. ComfyUI-GGUF-Knoten sicherstellen (custom_nodes/ComfyUI-GGUF).
    3. Modell-Download (h3_r2v/tools/dl_h3.py, ~35 GB, Resumable).
    4. cast.json Refs auf cast_refs/ im Repo auftren (relativ).
    5. lab.json fuer diesen Rechner schreiben (lab/bootstrap/configure.py).

  Aufruf (Windows PowerShell):
    powershell -ExecutionPolicy Bypass -File .\setup.ps1
    powershell -ExecutionPolicy Bypass -File .\setup.ps1 -InstallComfyUI -ComfyRoot "$env:USERPROFILE\ComfyUI"
    powershell -ExecutionPolicy Bypass -File .\setup.ps1 -SkipModels        # Modelle schon da? Nur config/Checker

  Danach:
    .\h3_r2v\run_proof.ps1 -Shots video_shots.json -ComfyUrl http://127.0.0.1:8188
============================================================================ #>
param(
  [string]$ComfyRoot = "",
  [string]$Python = "",
  [string]$ModelsDir = "",
  [switch]$InstallComfyUI,
  [switch]$SkipModels,
  [switch]$NoLab,
  [switch]$Fast
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'

function Info($m)  { Write-Host "==> " -ForegroundColor Cyan  -NoNewline; Write-Host $m }
function Warn($m)  { Write-Host "!!> " -ForegroundColor Yellow -NoNewline; Write-Host $m }
function Fatal($m) { Write-Host "XX> " -ForegroundColor Red -NoNewline; Write-Host $m; exit 1 }

function Test-ComfyMain($d) {
  return ($d -and (Test-Path (Join-Path $d 'main.py')))
}

# ---- 1. ComfyRoot bestimmen ----
if (-not (Test-ComfyMain $ComfyRoot)) {
  $known = @(
    (Join-Path $env:LOCALAPPDATA 'Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI'),
    (Join-Path $env:USERPROFILE 'ComfyUI\ComfyUI'),
    (Join-Path $root 'ComfyUI')
  )
  foreach ($k in $known) { if (Test-ComfyMain $k) { $ComfyRoot = $k; break } }
}

if (-not (Test-ComfyMain $ComfyRoot)) {
  if ($InstallComfyUI) {
    $ComfyRoot = if ($ComfyRoot) { $ComfyRoot } else { Join-Path $env:USERPROFILE 'ComfyUI' }
    Info "ComfyUI wird nach $ComfyRoot geklont..."
    if (-not (Test-Path $ComfyRoot)) {
      New-Item -ItemType Directory -Force -Path $ComfyRoot | Out-Null
      git clone --depth 1 https://github.com/comfyanonymous/ComfyUI (Join-Path $ComfyRoot 'ComfyUI') 2>$null
      $ComfyRoot = Join-Path $ComfyRoot 'ComfyUI'
      if (-not (Test-ComfyMain $ComfyRoot)) { Fatal "ComfyUI-Klon fehlgeschlagen (git verfuegbar?). -ComfyRoot angeben." }
    }
  } else {
    Fatal "ComfyUI nicht gefunden. Entweder Comfy-Desktop installieren, -ComfyRoot <Pfad> angeben oder -InstallComfyUI fuer frisches Klon."
  }
}
Info "ComfyRoot: $ComfyRoot"

# ---- 2. Python finden (venv des ComfyUI bevorzugt) ----
if (-not $Python) {
  $cands = @(
    (Join-Path $ComfyRoot '.venv\Scripts\python.exe'),
    (Join-Path $ComfyRoot 'venv\Scripts\python.exe'),
    "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe"
  )
  foreach ($c in $cands) { if ($c -and (Test-Path $c)) { $Python = $c; break } }
}
if (-not $Python) { $Python = 'py' }
& $Python --version 2>&1 | ForEach-Object { Info "Python: $_" }

# ---- 2b. Bei frischem Klon: venv + requirements ----
if ($InstallComfyUI -and -not (Test-Path (Join-Path $ComfyRoot '.venv'))) {
  Info "Erstelle venv und installiere Python-Pakete (torch etc.) - das dauert..."
  & $Python -m venv (Join-Path $ComfyRoot '.venv')
  $Python = Join-Path $ComfyRoot '.venv\Scripts\python.exe'
  & $Python -m pip install --upgrade pip 2>&1 | Out-Null
  & $Python -m pip install --pre "torch>=2.6" torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126 2>&1 | Select-Object -Last 5
  & $Python -m pip install "gguf>=0.13.0" "spandrel" "huggingface_hub" "av" "imageio-ffmpeg" "sentencepiece" "protobuf" 2>&1 | Select-Object -Last 5
}

# ---- 3. ComfyUI-GGUF custom node sicherstellen ----
$ggufNode = Join-Path $ComfyRoot 'custom_nodes\ComfyUI-GGUF'
if (-not (Test-Path $ggufNode)) {
  Info "Klone ComfyUI-GGUF custom node..."
  New-Item -ItemType Directory -Force -Path (Join-Path $ComfyRoot 'custom_nodes') | Out-Null
  git clone --depth 1 https://github.com/city96/ComfyUI-GGUF (Join-Path $ComfyRoot 'custom_nodes\ComfyUI-GGUF')
}
if (Test-Path $ggufNode) {
  $req = Join-Path $ggufNode 'requirements.txt'
  if (Test-Path $req) {
    Info "Installiere ComfyUI-GGUF Anforderungen..."
    & $Python -m pip install -r $req 2>&1 | Select-Object -Last 3
  }
} else {
  Warn "ComfyUI-GGUF konnte nicht installiert werden - UnetLoaderGGUF wird fehlen!"
}

# ---- 4. Modell-Download ----
if (-not $ModelsDir) { $ModelsDir = Join-Path $ComfyRoot 'models' }
if ((Test-Path $ModelsDir) -and -not $SkipModels) {
  Info "Lade H3-Modelle nach $ModelsDir (Resumable, >30 GB)..."
  $dl = Join-Path $root 'h3_r2v\tools\dl_h3.py'
  if (-not (Test-Path $dl)) { Fatal "$dl fehlt" }
  & $Python $dl --models-dir $ModelsDir
  if ($LASTEXITCODE -ne 0) { Warn "Modell-Download mit Fehlern beendet (rc=$LASTEXITCODE) - Modelle pruefen!" }
} elseif ($SkipModels) {
  Warn "Modelle uebersprungen (-SkipModels). Stelle sicher, dass Folgendes in $ModelsDir liegt:"
  Warn "  diffusion_models/MiniMax-H3-Ref2VA-Pruned-Q4_K_M.gguf"
  Warn "  text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
  Warn "  vae/minimax_h3_video_vae_fp16.safetensors  +  minimax_h3_audio_vae_fp32.safetensors"
} else {
  Fatal "Models-Verzeichnis $ModelsDir existiert nicht - -SkipModels oder -ModelsDir angeben."
}

# ---- 5. cast.json Refs pruefen (relativ zu Repo-cast_refs) ----
$cast = Join-Path $root 'h3_r2v\config\cast.json'
if (Test-Path $cast) {
  $missing = @()
  $cfg = Get-Content $cast -Raw | ConvertFrom-Json
  foreach ($c in $cfg.cast) {
    $p = [IO.Path]::GetFullPath((Join-Path (Join-Path $root 'h3_r2v\config') $c.ref))
    if (-not (Test-Path $p)) { $missing += "$($c.id): $p" }
  }
  if ($missing.Count -gt 0) {
    Warn "Cast-Refs fehlen: $($missing -join ' | ')"
  } else {
    Info "Cast-Refs OK (relativ zu Repo cast_refs/)."
  }
}

# ---- 6. lab.json fuer diesen Rechner ----
if (-not $NoLab) {
  $labCfg = Join-Path $root 'lab\bootstrap\configure.py'
  if (Test-Path $labCfg) {
    Info "Schreibe lab.json fuer diesen Rechner..."
    & $Python $labCfg --comfy $ComfyRoot --models $ModelsDir --orca $root --python $Python
  } else {
    Warn "configure.py fehlt - lab.json nicht generiert."
  }
}

# ---- Zusammenfassung ----
Write-Host ""
Write-Host "==================== SETUP FERTIG ====================" -ForegroundColor Green
Write-Host "ComfyRoot : $ComfyRoot"
Write-Host "Models    : $ModelsDir"
Write-Host ""
Write-Host "Naechste Schritte:"
if ($InstallComfyUI) { Write-Host "  1. ComfyUI einmal starten (custom nodes werden geladen):"
  Write-Host "     $ComfyRoot\main.py --windows-standalone-build" }
Write-Host "  2. ComfyUI-Server laufen lassen (Port 8188)."
Write-Host "  3. Video rendern:"
Write-Host "     .\h3_r2v\run_proof.ps1 -Shots .\h3_r2v\config\video_shots.json"
Write-Host "     (.\h3_r2v\run_proof.ps1 -Shots .\h3_r2v\config\video_shots.json -Only shot_01  = einzelner Test)"
Write-Host "  4. Reel basteln:  .\h3_r2v\run_proof.ps1 -Reel"
Write-Host "======================================================="