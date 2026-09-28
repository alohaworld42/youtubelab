<#
.SYNOPSIS
    Verifies all URLs in setup/assets.json return HTTP 200 or are explicitly null.
.DESCRIPTION
    Reads setup/assets.json, extracts all URLs from files[], repos[], and custom_nodes[].large_models[],
    and checks each with curl.exe. URLs that are null are reported as "SKIPPED (null)". All others must return 200.
.OUTPUTS
    Prints a table with URL, HTTP status, and PASS/FAIL. Exits with non-zero code if any URL fails.
#>

param(
    [string]$ManifestPath = "setup\assets.json"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $ManifestPath)) {
    Write-Error "Manifest not found: $ManifestPath"
    exit 1
}

$json = Get-Content $ManifestPath -Raw | ConvertFrom-Json

$allUrls = @()

# Collect from files[]
if ($json.files) {
    foreach ($f in $json.files) {
        $allUrls += [pscustomobject]@{
            Category = "files"
            Path     = $f.path
            Url      = $f.url
            Note     = $f.note
        }
    }
}

# Collect from repos[]
if ($json.repos) {
    foreach ($r in $json.repos) {
        $allUrls += [pscustomobject]@{
            Category = "repos"
            Path     = $r.target_dir
            Url      = $r.url
            Note     = $r.note
        }
    }
}

# Collect from custom_nodes[].large_models[]
if ($json.custom_nodes) {
    foreach ($cn in $json.custom_nodes) {
        if ($cn.large_models) {
            foreach ($lm in $cn.large_models) {
                $allUrls += [pscustomobject]@{
                    Category = "custom_nodes.$($cn.name).large_models"
                    Path     = $lm.relative_path
                    Url      = $lm.url
                    Note     = $lm.note
                }
            }
        }
    }
}

$results = @()
$failCount = 0

Write-Host "Checking $($allUrls.Count) URLs..." -ForegroundColor Cyan
Write-Host ""

foreach ($item in $allUrls) {
    $url = $item.Url
    $label = "$($item.Category) / $($item.Path)"

    if ($null -eq $url -or $url -eq '' -or $url -eq 'null') {
        $status = "SKIPPED (null)"
        $code = "N/A"
        $pass = $true
    }
    else {
        try {
            $code = curl.exe -s -o NUL -I -L -w "%{http_code}" --max-time 30 $url 2>$null
            if ($LASTEXITCODE -ne 0) {
                $code = "CURL_ERROR"
            }
        }
        catch {
            $code = "EXCEPTION"
        }

        if ($code -eq "200") {
            $status = "OK"
            $pass = $true
        }
        else {
            $status = "FAIL (HTTP $code)"
            $pass = $false
            $failCount++
        }
    }

    $result = [pscustomobject]@{
        Label  = $label
        Url    = $url
        Code   = $code
        Status = $status
        Pass   = $pass
    }
    $results += $result

    $color = if ($pass) { "Green" } elseif ($code -eq "N/A") { "Gray" } else { "Red" }
    Write-Host ("{0,-60} {1,-12} {2}" -f $label.Substring(0, [Math]::Min(60, $label.Length)), $code, $status) -ForegroundColor $color
}

Write-Host ""
Write-Host ("Total: {0}, Passed: {1}, Failed: {2}, Skipped: {3}" -f `
    $results.Count, `
    ($results | Where-Object { $_.Pass -and $_.Code -ne "N/A" }).Count, `
    $failCount, `
    ($results | Where-Object { $_.Code -eq "N/A" }).Count) -ForegroundColor Cyan

if ($failCount -gt 0) {
    Write-Host ""
    Write-Host "FAILED URLs:" -ForegroundColor Red
    $results | Where-Object { -not $_.Pass } | ForEach-Object {
        Write-Host "  $($_.Label) -> $($_.Url)" -ForegroundColor Red
    }
    exit 1
}
else {
    Write-Host ""
    Write-Host "All URLs verified successfully." -ForegroundColor Green
    exit 0
}