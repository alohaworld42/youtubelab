# start.ps1 — Bootstrap + Start für Brainrot Studio.
# Wird von Start-BrainrotStudio.bat aufgerufen. Macht beim ersten Lauf alles selbst:
# venv anlegen, Pakete installieren, FFmpeg via winget, dann App starten.

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$Host.UI.RawUI.WindowTitle = "Brainrot Studio"

function Refresh-Path {
    $env:PATH = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' +
                [Environment]::GetEnvironmentVariable('Path', 'User')
}
Refresh-Path
$env:PYTHONIOENCODING = "utf-8"

Write-Host ""
Write-Host "  BRAINROT STUDIO" -ForegroundColor Yellow
Write-Host "  ---------------" -ForegroundColor DarkGray

# --- 1. Python finden -------------------------------------------------------
$py = $null
foreach ($candidate in @("py -3.11", "py -3.12", "py -3.10", "python")) {
    try {
        $v = Invoke-Expression "$candidate --version 2>&1"
        if ($v -match "3\.(10|11|12)") { $py = $candidate; break }
    } catch {}
}
if (-not $py) {
    Write-Host "  [X] Python 3.10-3.12 nicht gefunden." -ForegroundColor Red
    Write-Host "      Download: https://www.python.org/downloads/  (Haken bei 'Add to PATH'!)"
    Read-Host "  Enter zum Beenden"
    exit 1
}
Write-Host "  [OK] Python: $py"

# --- 2. venv + Pakete --------------------------------------------------------
$venvPython = Join-Path $root "venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "  [..] Erster Start: richte Umgebung ein (einmalig, dauert ein paar Minuten)..."
    Invoke-Expression "$py -m venv venv"
}
# Marker: Pakete installiert? flask als Stellvertreter testen.
& $venvPython -c "import flask" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "  [..] Installiere Pakete (einmalig)..."
    & $venvPython -m pip install --disable-pip-version-check --quiet -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  [X] Paket-Installation fehlgeschlagen. Log oben pruefen." -ForegroundColor Red
        Read-Host "  Enter zum Beenden"
        exit 1
    }
}
Write-Host "  [OK] Umgebung bereit"

# --- 3. FFmpeg ----------------------------------------------------------------
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Host "  [..] FFmpeg fehlt - installiere via winget..."
    try {
        winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements
        Refresh-Path
    } catch {}
    if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
        Write-Host "  [!] FFmpeg weiterhin nicht gefunden - Setup-Seite zeigt den Fix." -ForegroundColor Yellow
    } else {
        Write-Host "  [OK] FFmpeg installiert"
    }
} else {
    Write-Host "  [OK] FFmpeg vorhanden"
}

# --- 3b. short-video-maker container (optional render engine) ---------------
try { & (Join-Path $PSScriptRoot "start-svm.ps1") } catch { Write-Host "  [!] short-video-maker Start uebersprungen: $_" -ForegroundColor Yellow }

# --- 3c. ComfyUI (optional local AI video backgrounds) -----------------------
try { & (Join-Path $PSScriptRoot "start-comfyui.ps1") } catch { Write-Host "  [!] ComfyUI Start uebersprungen: $_" -ForegroundColor Yellow }

# --- 4. Start ------------------------------------------------------------------
Write-Host ""
Write-Host "  Starte... Browser oeffnet sich gleich (http://127.0.0.1:5000)" -ForegroundColor Green
Write-Host "  Dieses Fenster offen lassen - hier laeuft der Worker. Schliessen = App aus."
Write-Host ""
& $venvPython app.py
Read-Host "  App beendet. Enter zum Schliessen"
