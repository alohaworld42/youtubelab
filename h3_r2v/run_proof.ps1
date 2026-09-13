<#
run_proof.ps1 - Konsistenz-Proof (H3 Reference-to-Video, fixe Refs+Seeds)
Beispiele:
  ... -DryRun
  ... -CaptureTemplate
  ... -Only proof_01
  ... -Resume
  ... -Reel
#>
param(
  [string]$ComfyRoot = "",
  [string]$Python = "",
  [string]$Template = "",
  [string]$Shots = "",
  [string]$Only = "",
  [string]$ComfyUrl = "http://127.0.0.1:8188",
  [int]$Timeout = 2400,
  [switch]$DryRun,
  [switch]$Resume,
  [switch]$Reel,
  [switch]$CaptureTemplate
)

$ErrorActionPreference = 'Continue'
$proj    = $PSScriptRoot
$config  = Join-Path $proj 'config'
$scripts = Join-Path $proj 'scripts'
$outDir  = Join-Path $proj 'output'
$logs    = Join-Path $proj 'logs'
$tmp     = Join-Path $proj 'Claudetempfiles'
foreach ($d in @($config, $scripts, $outDir, $logs, $tmp)) {
  New-Item -ItemType Directory -Force -Path $d | Out-Null
}
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'

# ---- ComfyRoot auto-erkennen (Fallback: bekannte Standardpfade) ----
if (-not $ComfyRoot) {
  $known = @(
    (Join-Path $env:LOCALAPPDATA 'Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI'),
    (Join-Path $env:USERPROFILE 'ComfyUI\ComfyUI'),
    (Join-Path $proj 'ComfyUI')
  )
  foreach ($k in $known) { if ($k -and (Test-Path (Join-Path $k 'main.py'))) { $ComfyRoot = $k; break } }
  if (-not $ComfyRoot) {
    Write-Host "FEHLER: ComfyUI nicht gefunden. -ComfyRoot <Pfad> angeben."
    exit 1
  }
  Write-Host "ComfyRoot auto-erkannt: $ComfyRoot"
}

$driver = Join-Path $scripts 'render_shots_ref.py'
if (-not (Test-Path $driver)) { Write-Host "FEHLER: $driver fehlt"; exit 1 }

# ---- Python finden (Driver ist stdlib-only, jeder Python >= 3.8 geht) ----
if (-not $Python) {
  $cands = @(
    (Join-Path $ComfyRoot '.venv\Scripts\python.exe'),
    (Join-Path $ComfyRoot 'venv\Scripts\python.exe'),
    "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe"
  )
  foreach ($c in $cands) { if ($c -and (Test-Path $c)) { $Python = $c; break } }
  if (-not $Python) { $Python = 'py' }
}
Write-Host "Python: $Python"

function Invoke-Logged([string]$Exe, [string]$ArgStr) {
  $stamp  = Get-Date -Format 'yyyyMMdd_HHmmss'
  $outLog = Join-Path $logs ("run_" + $stamp + ".log")
  $errLog = Join-Path $logs ("run_" + $stamp + ".err.log")
  $p = Start-Process -FilePath $Exe -ArgumentList $ArgStr -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput $outLog -RedirectStandardError $errLog
  if (Test-Path $outLog) { Get-Content $outLog }
  if ($p.ExitCode -ne 0) {
    Write-Host ""
    Write-Host "=== ExitCode $($p.ExitCode) - stderr (letzte 60 Zeilen): ==="
    if (Test-Path $errLog) { Get-Content $errLog -Tail 60 }
  }
  return $p.ExitCode
}

# ---- Modus: Reel (kein ComfyUI noetig) ----
if ($Reel) {
  $reelPy = Join-Path $scripts 'make_proof_reel.py'
  $argStr = '"' + $reelPy + '" --shots-dir "' + (Join-Path $outDir 'shots') + '"' +
            ' --manifest "' + (Join-Path $outDir 'manifest.json') + '"' +
            ' --out-dir "' + (Join-Path $outDir 'reel') + '"' +
            ' --ffmpeg-root "' + $ComfyRoot + '"'
  exit (Invoke-Logged $Python $argStr)
}

function Test-ComfyUp {
  try {
    $r = Invoke-WebRequest -UseBasicParsing -Uri ($ComfyUrl + '/system_stats') -TimeoutSec 3
    return ($r.StatusCode -eq 200)
  } catch { return $false }
}

if (-not (Test-ComfyUp)) {
  Write-Host "ComfyUI laeuft nicht - versuche tools\setup_h3_comfy.ps1 in eigenem Fenster zu starten..."
  $setup = Join-Path $ComfyRoot 'tools\setup_h3_comfy.ps1'
  if (Test-Path $setup) {
    Start-Process -FilePath "C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe" `
      -ArgumentList ('-NoProfile -ExecutionPolicy Bypass -File "' + $setup + '"')
  } else {
    Write-Host "HINWEIS: $setup nicht gefunden - ComfyUI bitte wie gewohnt manuell starten."
  }
  $up = $false
  for ($i = 0; $i -lt 60; $i++) {
    Start-Sleep -Seconds 5
    if (Test-ComfyUp) { $up = $true; break }
  }
  if (-not $up) { Write-Host "FEHLER: ComfyUI nach 300s nicht erreichbar ($ComfyUrl)."; exit 1 }
}
Write-Host "ComfyUI erreichbar."

# ---- Modus: Template aus Historie fangen ----
if ($CaptureTemplate) {
  $tplOut = Join-Path $config 'h3_template.json'
  $argStr = '"' + $driver + '" --comfy ' + $ComfyUrl + ' --capture-template "' + $tplOut + '"'
  exit (Invoke-Logged $Python $argStr)
}

# ---- Template finden ----
$tpl = $Template
if (-not $tpl) {
  $def = Join-Path $config 'h3_template.json'
  if (Test-Path $def) { $tpl = $def }
}
if (-not $tpl) {
  $roots = @(
    (Join-Path $ComfyRoot 'user\default\workflows'),
    'C:\Users\admin\Desktop\Projects\youtubegenerator\output',
    'C:\Users\admin\Desktop\Projects\youtubegenerator\workflows',
    'C:\Users\admin\Desktop\Projects\youtubegenerator\pipeline',
    $proj
  )
  $jsons = @()
  foreach ($r in $roots) {
    if (Test-Path $r) {
      $jsons += Get-ChildItem -Path $r -Recurse -Depth 5 -Filter *.json -File -ErrorAction SilentlyContinue
    }
  }
  $jsons = $jsons | Sort-Object LastWriteTime -Descending | Select-Object -First 30
  foreach ($j in $jsons) {
    $hit = Select-String -Path $j.FullName -Pattern 'MiniMaxH3|ReferenceToVideo' -List -ErrorAction SilentlyContinue
    if ($hit) { $tpl = $j.FullName; break }
  }
  if ($tpl) { Write-Host "Template auto-erkannt: $tpl" }
}
if (-not $tpl) {
  Write-Host "FEHLER: kein H3-Ref2Video-Template gefunden."
  Write-Host "Loesung: alten H3-Test einmal laufen lassen, dann:"
  Write-Host ('  & "C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -ExecutionPolicy Bypass -File "' + $PSScriptRoot + '\run_proof.ps1" -CaptureTemplate')
  exit 1
}

# ---- Renderer starten ----
$shotsF = $Shots
if (-not $shotsF) { $shotsF = Join-Path $config 'proof_shots.json' }
$castF  = Join-Path $config 'cast.json'
$argStr = '"' + $driver + '"' +
  ' --shots "' + $shotsF + '"' +
  ' --cast "' + $castF + '"' +
  ' --out "' + $outDir + '"' +
  ' --comfy ' + $ComfyUrl +
  ' --timeout ' + $Timeout +
  ' --template "' + $tpl + '"'
if ($Only)   { $argStr += ' --only ' + $Only }
if ($DryRun) { $argStr += ' --dry-run' }
if ($Resume) { $argStr += ' --resume' }

Write-Host ""
Write-Host "Starte Renderer..."
$code = Invoke-Logged $Python $argStr
if ($code -eq 0 -and -not $DryRun) {
  Write-Host ""
  Write-Host "Outputs: $outDir\shots\*.mp4"
  Write-Host "Danach Reel: run_proof.ps1 -Reel"
}
exit $code