<#
.SYNOPSIS
    Downloads all large assets ignored by git (ComfyUI models, LatentSync, espeak-ng, ComfyUI portable).

.DESCRIPTION
    Idempotent download script for setting up a fresh Windows PC.
    Reads manifest from setup/assets.json and downloads/extracts everything.

.PARAMETER DryRun
    List only what would be downloaded, show total size, do not download.

.PARAMETER Only
    Limit to specific component: comfyui, latentsync, espeak, comfyui_portable

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File setup\download_assets.ps1

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File setup\download_assets.ps1 -DryRun

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File setup\download_assets.ps1 -Only comfyui
#>

param(
    [switch]$DryRun,
    [ValidateSet('comfyui','latentsync','espeak','comfyui_portable')]
    [string]$Only
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$repoRoot = Split-Path -Parent $scriptDir
$manifestPath = Join-Path $scriptDir 'assets.json'

if (-not (Test-Path $manifestPath)) {
    Write-Error "Manifest not found: $manifestPath"
    exit 1
}

$manifest = Get-Content $manifestPath -Raw | ConvertFrom-Json

function Get-HumanSize {
    param([long]$bytes)
    if ($bytes -ge 1GB) { return "{0:N1} GB" -f ($bytes/1GB) }
    if ($bytes -ge 1MB) { return "{0:N1} MB" -f ($bytes/1MB) }
    if ($bytes -ge 1KB) { return "{0:N1} KB" -f ($bytes/1KB) }
    return "$bytes B"
}

function Test-FileExistsAndSize {
    param([string]$path, [long]$expectedBytes)
    if (-not (Test-Path $path)) { return $false }
    $actual = (Get-Item $path).Length
    return $actual -eq $expectedBytes
}

function Download-File {
    param(
        [string]$url,
        [string]$destPath,
        [long]$expectedBytes
    )

    $destDir = Split-Path -Parent $destPath
    if (-not (Test-Path $destDir)) {
        New-Item -ItemType Directory -Path $destDir -Force | Out-Null
    }

    $partPath = "$destPath.part"
    $args = @('-L', '--retry', '5', '--retry-delay', '5', '-C', '-', '-o', $partPath, $url)
    Write-Host "  Downloading: $url"
    Write-Host "  To: $destPath"
    $exitCode = (Start-Process -FilePath 'curl.exe' -ArgumentList $args -Wait -PassThru).ExitCode
    if ($exitCode -ne 0) {
        if (Test-Path $partPath) { Remove-Item $partPath -Force }
        throw "curl failed with exit code $exitCode for $url"
    }
    Move-Item -Path $partPath -Destination $destPath -Force
    Write-Host "  Done: $destPath"
}

function Clone-Or-Update-Repo {
    param(
        [string]$url,
        [string]$commit,
        [string]$targetDir
    )

    $targetPath = Join-Path $repoRoot $targetDir
    if (Test-Path (Join-Path $targetPath '.git')) {
        Write-Host "  Repo exists, fetching updates: $targetDir"
        git -C $targetPath fetch origin
        Write-Host "  Checking out commit: $commit"
        git -C $targetPath checkout $commit
    } else {
        Write-Host "  Cloning: $url -> $targetDir"
        $parent = Split-Path -Parent $targetPath
        if (-not (Test-Path $parent)) { New-Item -ItemType Directory -Path $parent -Force | Out-Null }
        git clone $url $targetPath
        Write-Host "  Checking out commit: $commit"
        git -C $targetPath checkout $commit
    }
}

function Extract-7z {
    param([string]$archivePath, [string]$destDir)
    if (-not (Get-Command '7z' -ErrorAction SilentlyContinue)) {
        throw "7z not found in PATH. Install 7-Zip."
    }
    Write-Host "  Extracting $archivePath to $destDir"
    & 7z x $archivePath -o$destDir -y
}

function Extract-MSI {
    param([string]$msiPath, [string]$destDir)
    Write-Host "  Extracting MSI: $msiPath -> $destDir"
    $absDest = Resolve-Path $destDir
    $absMsi = Resolve-Path $msiPath
    $args = "/a `"$absMsi`" /qn TARGETDIR=`"$absDest`""
    Start-Process -FilePath 'msiexec.exe' -ArgumentList $args -Wait -NoNewWindow
}

# --- MAIN ---

Write-Host "=== Brainrot Studio Asset Downloader ==="
Write-Host "Repo root: $repoRoot"
Write-Host "Manifest: $manifestPath"
Write-Host ""

$stats = @{ ok=0; skipped=0; missing_url=0; failed=0; total_bytes=0 }

# 1. ComfyUI Portable
if (-not $Only -or $Only -eq 'comfyui_portable') {
    $portable = $manifest.repos | Where-Object { $_.name -eq 'comfyui_portable' } | Select-Object -First 1
    if ($portable) {
        $targetDir = Join-Path $repoRoot $portable.target_dir
        $archiveName = 'ComfyUI_windows_portable_nvidia_cu121_or_cpu.7z'
        $archivePath = [System.IO.Path]::Combine($repoRoot, 'comfyui', $archiveName)

        if (-not (Test-Path (Join-Path $targetDir 'ComfyUI'))) {
            $stats.total_bytes += 1.5GB # approximate
            Write-Host "ComfyUI Portable: MISSING (will download ~1.5 GB)"
            if (-not $DryRun) {
                if (-not (Test-Path $archivePath)) {
                    Write-Host "  Downloading ComfyUI portable..."
                    Download-File $portable.url $archivePath 1500000000
                }
                Extract-7z $archivePath (Join-Path $repoRoot 'comfyui')
                Write-Host "  ComfyUI portable extracted."
            }
            $stats.missing_url++
        } else {
            Write-Host "ComfyUI Portable: OK (already extracted)"
            $stats.ok++
        }
    }
}

# 2. Repos (latentsync, comfyui)
$reposToProcess = $manifest.repos | Where-Object { $_.name -ne 'comfyui_portable' }
if ($Only) {
    $reposToProcess = $reposToProcess | Where-Object { $_.name -eq $Only }
}

foreach ($repo in $reposToProcess) {
    $targetPath = Join-Path $repoRoot $repo.target_dir
    $needsClone = -not (Test-Path (Join-Path $targetPath '.git'))
    if ($needsClone) {
        Write-Host "Repo $($repo.name): MISSING (will clone)"
        $stats.missing_url++
        if (-not $DryRun) {
            Clone-Or-Update-Repo $repo.url $repo.commit $repo.target_dir
        }
    } else {
        Write-Host "Repo $($repo.name): OK (exists)"
        $stats.ok++
    }
}

# 3. Files
$filesToProcess = $manifest.files
if ($Only) {
    switch ($Only) {
        'comfyui' { $filesToProcess = $filesToProcess | Where-Object { $_.path -like 'comfyui/*' } }
        'latentsync' { $filesToProcess = $filesToProcess | Where-Object { $_.path -like 'latentsync/*' } }
        'espeak' { $filesToProcess = $filesToProcess | Where-Object { $_.path -like 'espeak-ng/*' } }
        'comfyui_portable' { $filesToProcess = @() }
    }
}

foreach ($file in $filesToProcess) {
    $destPath = Join-Path $repoRoot $file.path
    $expectedBytes = [long]$file.size_bytes
    $exists = Test-FileExistsAndSize $destPath $expectedBytes

    if ($exists) {
        Write-Host "OK: $($file.path) ($(Get-HumanSize $expectedBytes))"
        $stats.skipped++
    } elseif ($file.url) {
        $stats.total_bytes += $expectedBytes
        Write-Host "MISSING: $($file.path) ($(Get-HumanSize $expectedBytes))"
        $stats.missing_url++
        if (-not $DryRun) {
            try {
                Download-File $file.url $destPath $expectedBytes
                $stats.ok++
            } catch {
                Write-Error "FAILED: $($file.path) - $_"
                $stats.failed++
            }
        }
    } else {
        Write-Host "NO URL: $($file.path) ($(Get-HumanSize $expectedBytes)) - $($file.note)"
        $stats.missing_url++
    }
}

# 4. espeak-ng MSI extraction
if (-not $Only -or $Only -eq 'espeak') {
    $msiPath = Join-Path $repoRoot 'espeak-ng\espeak-ng.msi'
    $extractDir = Join-Path $repoRoot 'espeak-ng\eSpeak NG'
    if ((Test-Path $msiPath) -and (-not (Test-Path $extractDir))) {
        Write-Host "Extracting espeak-ng MSI..."
        if (-not $DryRun) {
            Extract-MSI $msiPath $extractDir
        }
    }
}

# 5. LatentSync auxiliary models extraction
if (-not $Only -or $Only -eq 'latentsync') {
    $zipPath = Join-Path $repoRoot 'latentsync\checkpoints\auxiliary\buffalo_l.zip'
    $extractDir = Join-Path $repoRoot 'latentsync\checkpoints\auxiliary\models\buffalo_l'
    $checkFile = Join-Path $extractDir '1k3d68.onnx'
    if ((Test-Path $zipPath) -and (-not (Test-Path $checkFile))) {
        Write-Host "Extracting LatentSync buffalo_l.zip..."
        if (-not $DryRun) {
            if (Get-Command '7z' -ErrorAction SilentlyContinue) {
                & 7z x $zipPath -o$extractDir -y
            } else {
                Expand-Archive -Path $zipPath -DestinationPath $extractDir -Force
            }
        }
    }
}

# Summary
Write-Host ""
Write-Host "=== SUMMARY ==="
Write-Host "OK / already present:  $($stats.ok)"
Write-Host "Skipped (size match):  $($stats.skipped)"
Write-Host "Missing / needs action: $($stats.missing_url)"
if ($stats.failed -gt 0) { Write-Host "FAILED:                $($stats.failed)" }
Write-Host "Total download size:   $(Get-HumanSize $stats.total_bytes)"

if ($DryRun) {
    Write-Host ""
    Write-Host "DRY RUN complete. No files were downloaded."
    if ($stats.missing_url -eq 0) {
        Write-Host "All assets present."
        exit 0
    } else {
        exit 1
    }
}

if ($stats.failed -gt 0) {
    Write-Error "Some downloads failed."
    exit 1
}

Write-Host ""
Write-Host "All done."
exit 0