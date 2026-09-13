# start-svm.ps1 — ensure the short-video-maker container is running.
# Reads PEXELS_API_KEY from keys/pexels_api_key or .env. Safe to run repeatedly.
param([switch]$Quiet)

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
$image = "gyoridavid/short-video-maker:latest-tiny"
$name = "short-video-maker"
$videosDir = Join-Path $root "videos"

function say($m){ if(-not $Quiet){ Write-Host $m } }

function Docker-Ready { $v = docker info --format '{{.ServerVersion}}' 2>$null; return [bool]$v }

# Docker up?
if (-not (Docker-Ready)) {
    $dd = "$env:ProgramFiles\Docker\Docker\Docker Desktop.exe"
    if (Test-Path $dd) {
        say "  [..] Starte Docker Desktop..."
        Start-Process $dd
        foreach($i in 1..40){ if(Docker-Ready){ break }; Start-Sleep 3 }
    }
}
if (-not (Docker-Ready)) { say "  [!] Docker nicht bereit - short-video-maker uebersprungen (Fallback: MoviePy)."; return }

# Already running?
$running = docker ps --filter "name=$name" --filter "status=running" --format "{{.Names}}" 2>$null
if ($running -eq $name) { say "  [OK] short-video-maker laeuft"; return }

# Pull if missing
$have = docker images -q $image 2>$null
if (-not $have) { say "  [..] Lade short-video-maker Image (einmalig)..."; docker pull $image | Out-Null }

# Remove stale stopped container (only if one exists), then run
$exists = docker ps -aq --filter "name=^$name$" 2>$null
if ($exists) { docker rm -f $name 2>$null | Out-Null }

# PEXELS key from keys/ or .env
$pexels = $env:PEXELS_API_KEY
if (-not $pexels) {
    $kf = Join-Path $root "keys\pexels_api_key"
    if (Test-Path $kf) { $pexels = (Get-Content $kf -Raw).Trim() }
}
if (-not $pexels) {
    $envf = Join-Path $root ".env"
    if (Test-Path $envf) { $m = Select-String -Path $envf -Pattern '^PEXELS_API_KEY=(.+)$'; if($m){ $pexels = $m.Matches[0].Groups[1].Value.Trim() } }
}
if (-not $pexels) { say "  [!] PEXELS_API_KEY fehlt - short-video-maker braucht ihn fuer B-Roll. Im Setup eintragen." }

New-Item -ItemType Directory -Force -Path $videosDir | Out-Null
say "  [..] Starte short-video-maker Container (:3123)..."
docker run -d --name $name -p 3123:3123 `
    -e LOG_LEVEL=info `
    -e "PEXELS_API_KEY=$pexels" `
    -v "${videosDir}:/app/data/videos" `
    --restart unless-stopped `
    --entrypoint node `
    $image dist/index.js | Out-Null
# NOTE: the image's default CMD is "pnpm start", which under recent pnpm
# versions runs a dependency-status check that fails without a TTY (corepack
# has no packageManager field pinned and grabs latest pnpm at runtime). The
# package.json "start" script is just "node dist/index.js" anyway, so we call
# node directly and skip pnpm entirely.
say "  [OK] short-video-maker gestartet (erster Start laedt Modelle - dauert einige Minuten)"
