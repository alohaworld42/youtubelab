<#
.SYNOPSIS
    Downloads all large assets ignored by git (ComfyUI models, LatentSync, espeak-ng, ComfyUI portable, Custom Nodes).

.DESCRIPTION
    Idempotent download script for setting up a fresh Windows PC.
    Reads manifest from setup/assets.json and downloads/extracts everything.

.PARAMETER DryRun
    List only what would be downloaded, show total size, do not download.

.PARAMETER Only
    Limit to specific component: comfyui, latentsync, espeak, comfyui_portable, customnodes

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File setup\download_assets.ps1

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File setup\download_assets.ps1 -DryRun

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File setup\download_assets.ps1 -Only comfyui

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File setup\download_assets.ps1 -Only customnodes
#>

param(
    [switch]$DryRun,
    [ValidateSet('comfyui','latentsync','espeak','comfyui_portable','customnodes')]
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
        [string]$targetDir,
        [switch]$Shallow
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
        $cloneArgs = @('clone')
        if ($Shallow) { $cloneArgs += '--depth', '1' }
        $cloneArgs += $url, $targetPath
        git $cloneArgs
        if ($commit) {
            Write-Host "  Checking out commit: $commit"
            git -C $targetPath checkout $commit
        }
    }
}

function Install-CustomNodeRequirements {
    param([string]$nodeDir)

    $reqPath = Join-Path $nodeDir 'requirements.txt'
    if (-not (Test-Path $reqPath)) {
        Write-Host "  No requirements.txt found, skipping pip install"
        return
    }

    $pythonExe = Join-Path $repoRoot 'comfyui\ComfyUI_windows_portable\python_embeded\python.exe'
    if (-not (Test-Path $pythonExe)) {
        Write-Warning "  Embedded Python not found at $pythonExe, trying system python"
        $pythonExe = 'python'
    }

    Write-Host "  Installing requirements from $reqPath using $pythonExe"
    $args = @('-m', 'pip', 'install', '-r', $reqPath)
    $exitCode = (Start-Process -FilePath $pythonExe -ArgumentList $args -Wait -PassThru).ExitCode
    if ($exitCode -ne 0) {
        throw "pip install failed with exit code $exitCode for $reqPath"
    }
    Write-Host "  Requirements installed successfully"
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
    if (-not $portable) {
        # Fallback: check for portable in custom_nodes or new structure
        $portable = $manifest | Select-Object -ExpandProperty comfyui_portable -ErrorAction SilentlyContinue
    }
    
    # Find portable in repos (new structure) or as top-level property
    if (-not $portable) {
        $portable = $manifest.repos | Where-Object { $_.name -eq 'comfyui_portable' } | Select-Object -First 1
    }
    
    if ($portable) {
        $targetDir = Join-Path $repoRoot $portable.target_dir
        $archiveName = 'ComfyUI_windows_portable_nvidia_cu126.7z'
        $archivePath = Join-Path (Join-Path $repoRoot 'comfyui') $archiveName

        # Check if ComfyUI is already extracted (look for main.py or comfyui_version.py)
        $comfyuiMain = Join-Path $targetDir 'ComfyUI\main.py'
        $comfyuiVer = Join-Path $targetDir 'ComfyUI\comfyui_version.py'
        
        if (-not (Test-Path $comfyuiMain) -and -not (Test-Path $comfyuiVer)) {
            $stats.total_bytes += 1859361599 # actual size of cu126 asset
            Write-Host "ComfyUI Portable (v0.36.0 cu126): MISSING (will download ~1.7 GB)"
            if (-not $DryRun) {
                if (-not (Test-Path $archivePath)) {
                    Write-Host "  Downloading ComfyUI portable..."
                    Download-File $portable.url $archivePath 1859361599
                }
                Extract-7z $archivePath (Join-Path $repoRoot 'comfyui')
                Write-Host "  ComfyUI portable extracted."
            }
            $stats.missing_url++
        } else {
            Write-Host "ComfyUI Portable: OK (already extracted)"
            $stats.ok++
        }
    } else {
        Write-Warning "ComfyUI portable entry not found in manifest"
    }
}

# 2. Repos (latentsync)
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

# 3. Custom Nodes
if (-not $Only -or $Only -eq 'customnodes') {
    if ($manifest.custom_nodes) {
        foreach ($node in $manifest.custom_nodes) {
            $targetPath = Join-Path $repoRoot $node.target_dir
            $needsClone = -not (Test-Path (Join-Path $targetPath '.git'))
            
            if ($needsClone) {
                Write-Host "Custom Node $($node.name): MISSING (will clone)"
                $stats.missing_url++
                if (-not $DryRun) {
                    Clone-Or-Update-Repo $node.url '' $node.target_dir -Shallow
                    if ($node.has_requirements) {
                        Install-CustomNodeRequirements $targetPath
                    }
                }
            } else {
                Write-Host "Custom Node $($node.name): OK (exists)"
                $stats.ok++
                # Still check requirements if not in dry run
                if (-not $DryRun -and $node.has_requirements) {
                    Write-Host "  Checking requirements for $($node.name)..."
                    Install-CustomNodeRequirements $targetPath
                }
            }

            # Handle large models inside custom node
            if ($node.large_models) {
                foreach ($model in $node.large_models) {
                    $destPath = Join-Path $targetPath $model.relative_path
                    $expectedBytes = [long]$model.size_bytes
                    $exists = Test-FileExistsAndSize $destPath $expectedBytes

                    if ($exists) {
                        Write-Host "  Model OK: $($model.relative_path) ($(Get-HumanSize $expectedBytes))"
                        $stats.skipped++
                    } elseif ($model.url) {
                        $stats.total_bytes += $expectedBytes
                        Write-Host "  Model MISSING: $($model.relative_path) ($(Get-HumanSize $expectedBytes))"
                        $stats.missing_url++
                        if (-not $DryRun) {
                            try {
                                Download-File $model.url $destPath $expectedBytes
                                $stats.ok++
                            } catch {
                                Write-Error "  FAILED: $($model.relative_path) - $_"
                                $stats.failed++
                            }
                        }
                    } else {
                        Write-Host "  Model NO URL: $($model.relative_path) ($(Get-HumanSize $expectedBytes)) - $($model.note)"
                        $stats.missing_url++
                    }
                }
            }
        }
    } else {
        Write-Host "No custom_nodes section in manifest"
    }
}

# 4. Files (models, checkpoints, etc.)
$filesToProcess = $manifest.files
if ($Only) {
    switch ($Only) {
        'comfyui' { $filesToProcess = $filesToProcess | Where-Object { $_.path -like 'comfyui/*' } }
        'latentsync' { $filesToProcess = $filesToProcess | Where-Object { $_.path -like 'latentsync/*' } }
        'espeak' { $filesToProcess = $filesToProcess | Where-Object { $_.path -like 'espeak-ng/*' } }
        'comfyui_portable' { $filesToProcess = @() }
        'customnodes' { $filesToProcess = @() }
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

# 5. espeak-ng MSI extraction
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

# 6. LatentSync auxiliary models extraction
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