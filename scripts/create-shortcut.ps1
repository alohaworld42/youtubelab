# Legt eine Desktop-Verknuepfung "Brainrot Studio" an, die den Launcher startet.
$root = Split-Path -Parent $PSScriptRoot
$desktop = [Environment]::GetFolderPath("Desktop")
$lnkPath = Join-Path $desktop "Brainrot Studio.lnk"

$shell = New-Object -ComObject WScript.Shell
$lnk = $shell.CreateShortcut($lnkPath)
$lnk.TargetPath = Join-Path $root "Start-BrainrotStudio.bat"
$lnk.WorkingDirectory = $root
$lnk.IconLocation = "%SystemRoot%\System32\shell32.dll,238"
$lnk.Description = "Brainrot Studio - lokaler YouTube Shorts Generator"
$lnk.Save()

Write-Host "Verknuepfung angelegt: $lnkPath"
