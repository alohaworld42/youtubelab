# start-comfyui.ps1 — ensure a local ComfyUI server is running for AI video
# backgrounds (studio.comfyui tier "aivideo"). Safe to run repeatedly; skips
# quietly (comfyui.enabled stays false effectively) if the install is missing.
param([switch]$Quiet)

$root = Split-Path -Parent $PSScriptRoot
$comfyDir = Join-Path $root "comfyui"
$venvPython = Join-Path $comfyDir "venv\Scripts\python.exe"
$port = 8188

function say($m){ if(-not $Quiet){ Write-Host $m } }

if (-not (Test-Path $venvPython)) {
    say "  [!] ComfyUI nicht installiert unter $comfyDir - AI-Video-Hintergruende uebersprungen."
    return
}

# Already running?
try {
    $r = Invoke-WebRequest -Uri "http://127.0.0.1:$port/system_stats" -UseBasicParsing -TimeoutSec 2
    if ($r.StatusCode -eq 200) { say "  [OK] ComfyUI laeuft bereits"; return }
} catch {}

# Only start if the model files are actually present (avoid a broken empty server).
$ckpt = Join-Path $comfyDir "models\checkpoints\ltx-video-2b-v0.9.5.safetensors"
if (-not (Test-Path $ckpt)) {
    say "  [!] LTX-Video Modell fehlt noch (Download laeuft/nicht gestartet) - ComfyUI-Start uebersprungen."
    return
}

say "  [..] Starte ComfyUI (:$port) fuer lokale AI-Video-Hintergruende..."
$logDir = Join-Path $root "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
Start-Process -FilePath $venvPython `
    -ArgumentList "main.py", "--port", $port, "--preview-method", "none" `
    -WorkingDirectory $comfyDir `
    -RedirectStandardOutput (Join-Path $logDir "comfyui.log") `
    -RedirectStandardError (Join-Path $logDir "comfyui.err.log") `
    -WindowStyle Hidden
say "  [OK] ComfyUI gestartet (Hintergrund) - erster Ladevorgang der Modelle dauert etwas."
