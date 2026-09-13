# run_proof.ps1 -- Konsistenz-Beweislauf (3 Shots) auf dem lokalen ComfyUI-Server.
#
#   & "C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe" -ExecutionPolicy Bypass -File "C:\Users\Aloha\Desktop\Projects\Youtube\run_proof.ps1" -Stage all -RunId proof01
#
# Stages:
#   0  Setup: Modelle laden (nur fehlende), DINOv2 warmup, Graphen gegen /object_info pruefen
#   1  Keyframes: K1 (Klein) + K2 (Qwen-Edit-2511) fuer alle Shots, dann QC-Kontaktbogen
#   2  H3-Probe: s01 klein (864x480x39) MIT und OHNE Refs -> zeigt, ob Refs wirken
#   3  H3 Reference-to-Video 1344x768x73: s01, s02, s03 + Kontrolle s01_norefs (optional -Guide K1|K2)
#   4  QC (Scores, Kontaktbogen, report.md) + Konformierung 1280x720 + Handy-Version + Side-by-Side
#   all = 0,1,2,3,4
#
# -Resume ueberspringt vorhandene Dateien (Standard: an). Nichts wird geloescht.
# -KillLockHolder beendet einen fremden Render, der output\gpu_render.lock haelt (sonst Abbruch).
# Logs: logs\proof_<RunId>.out.log / .err.log ; pro Run: runs\<RunId>\run.log
param(
    [string]$Stage = "all",
    [string]$RunId = "proof01",
    [string]$Shots = "",
    [string]$Guide = "",
    [switch]$NoResume,
    [switch]$KillLockHolder
)
# WICHTIG: nicht "Stop" -- Pythons stderr wuerde in PS 5.1 zum Terminating Error.
$ErrorActionPreference = "Continue"
$env:PYTHONUTF8 = "1"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# Pfade portabel aus lab.json (von bootstrap\configure.py geschrieben).
$Lab    = $PSScriptRoot
$cfg    = Get-Content "$Lab\lab.json" -Raw | ConvertFrom-Json
$Comfy  = $cfg.comfy_dir
$Py     = $cfg.python
$Lock   = $cfg.lock_file
$ComfyArgs = $cfg.comfy_args
$OutLog = "$Lab\logs\proof_$RunId.out.log"
$ErrLog = "$Lab\logs\proof_$RunId.err.log"
$env:HF_HOME = "$Lab\.hf_cache"
New-Item -ItemType Directory -Force "$Lab\logs" | Out-Null

function Test-Server {
    try { return (Invoke-WebRequest -UseBasicParsing "http://127.0.0.1:8188/system_stats" -TimeoutSec 3).StatusCode -eq 200 }
    catch { return $false }
}
function Ensure-Server {
    if (Test-Server) { Write-Host "ComfyUI laeuft." -ForegroundColor DarkGray; return }
    $old = Get-NetTCPConnection -State Listen -LocalPort 8188 -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($old) { Write-Host "Port 8188 belegt aber tot -> Prozess $($old.OwningProcess) beenden"; Stop-Process -Id $old.OwningProcess -Force -ErrorAction SilentlyContinue; Start-Sleep 3 }
    Write-Host "Starte ComfyUI ..." -ForegroundColor Cyan
    $args = $ComfyArgs
    Start-Process -FilePath $Py -ArgumentList $args -WorkingDirectory $Comfy -RedirectStandardOutput "$Lab\logs\comfyui_server.out.log" -RedirectStandardError "$Lab\logs\comfyui_server.err.log" | Out-Null
    for ($i = 0; $i -lt 60; $i++) { Start-Sleep 3; if (Test-Server) { break } }
    if (-not (Test-Server)) { throw "ComfyUI kam nicht auf 127.0.0.1:8188 hoch (siehe logs\comfyui_server.err.log)." }
    Write-Host "Server laeuft." -ForegroundColor Green
}
function Check-Lock {
    if (-not (Test-Path $Lock)) { return }
    try { $info = Get-Content $Lock -Raw | ConvertFrom-Json } catch { return }
    $p = Get-Process -Id $info.pid -ErrorAction SilentlyContinue
    if ($p) {
        if ($KillLockHolder) { Write-Host "Beende fremden Render pid $($info.pid) ($($p.ProcessName))" -ForegroundColor Yellow; Stop-Process -Id $info.pid -Force; Start-Sleep 2 }
        else { throw "GPU-Lock wird von lebendem Prozess $($info.pid) gehalten. Warten oder -KillLockHolder." }
    }
}
function Run-Py([string]$Label, [string[]]$PyArgs) {
    Write-Host "== $Label" -ForegroundColor Cyan
    "== $Label $(Get-Date -Format s)" | Out-File -FilePath $OutLog -Append -Encoding utf8
    & $Py $PyArgs 2>> $ErrLog | Tee-Object -FilePath $OutLog -Append
    $code = $LASTEXITCODE
    if ($code -ne 0) { throw "$Label fehlgeschlagen (exit $code). Siehe $ErrLog" }
}

$resume = @(); if (-not $NoResume) { $resume = @("--resume") }
$shotArg = @(); if ($Shots) { $shotArg = @("--shots", $Shots) }
$stages = if ($Stage -eq "all") { @("0","1","2","3","4") } else { $Stage -split "," }

Write-Host "Run $RunId  Stages: $($stages -join ',')" -ForegroundColor Green
Check-Lock
if ($stages -contains "0") {
    Run-Py "Stage 0 Setup" @("$Lab\agents\00_setup\setup_models.py", "--download", "--warmup")
    Ensure-Server
    Run-Py "Stage 0 Validate" @("$Lab\agents\00_setup\setup_models.py", "--validate")
}
if ($stages -contains "1") {
    Ensure-Server
    Run-Py "Stage 1 K1 Klein keyframes" (@("$Lab\agents\01_keyframe\klein_agent.py", "--run", $RunId) + $resume + $shotArg)
    Run-Py "Stage 1 K2 Qwen-Edit keyframes" (@("$Lab\agents\01_keyframe\qwen_edit_agent.py", "--run", $RunId) + $resume + $shotArg)
    Run-Py "Stage 1 QC keyframes" @("$Lab\agents\03_qc\identity_qc.py", "--run", $RunId, "--what", "keyframes")
}
if ($stages -contains "2") {
    Ensure-Server
    Run-Py "Stage 2 H3 probe (refs vs norefs)" (@("$Lab\agents\02_video\h3_r2v_agent.py", "--run", $RunId, "--probe") + $resume)
    Run-Py "Stage 2 QC probe" @("$Lab\agents\03_qc\identity_qc.py", "--run", $RunId, "--what", "clips")
}
if ($stages -contains "3") {
    Ensure-Server
    $g = @(); if ($Guide) { $g = @("--guide", $Guide) }
    Run-Py "Stage 3 H3 R2V clips" (@("$Lab\agents\02_video\h3_r2v_agent.py", "--run", $RunId) + $resume + $shotArg + $g)
}
if ($stages -contains "4") {
    Run-Py "Stage 4 QC all" @("$Lab\agents\03_qc\identity_qc.py", "--run", $RunId, "--what", "all")
    Run-Py "Stage 4 Assemble" @("$Lab\agents\04_assemble\assemble.py", "--run", $RunId)
}
Write-Host ""
Write-Host "FERTIG. Ergebnisse: $Lab\runs\$RunId\final  (Handy: ...\phone, QC: ...\qc\report.md)" -ForegroundColor Green
