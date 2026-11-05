# Visenlo — One-shot local installer for Windows (PowerShell)
# Usage:
#   powershell -ExecutionPolicy Bypass -File install.ps1
#   OR hosted: irm https://YOUR_WEBSITE/install.ps1 | iex
# What it does completely locally:
#   - Checks Python 3.10+
#   - Creates .venv, installs pip deps
#   - Checks models, runs self-check
#   - Launches Gradio on 127.0.0.1:7860 (100% local)

$ErrorActionPreference = "Stop"
$RepoUrl = if ($env:VISENLO_REPO) { $env:VISENLO_REPO } elseif ($env:REACTORX_REPO) { $env:REACTORX_REPO } else { "https://github.com/Mevinb/Visenlo.git" }
$RepoZip = if ($env:VISENLO_ZIP) { $env:VISENLO_ZIP } elseif ($env:REACTORX_ZIP) { $env:REACTORX_ZIP } else { "https://github.com/Mevinb/Visenlo/archive/refs/heads/main.zip" }

$Root = if ($PSScriptRoot) { $PSScriptRoot } else { (Get-Location).Path }
if (-not (Test-Path "$Root\app.py")) {
  if (Test-Path "$(Get-Location)\app.py") { $Root = Get-Location }
  elseif (Test-Path "$(Get-Location)\Visenlo\app.py") { $Root = "$(Get-Location)\Visenlo" }
  else {
    Write-Host "[Visenlo] Project files not found — fetching from $RepoUrl ..." -ForegroundColor Cyan
    if (Get-Command git -ErrorAction SilentlyContinue) {
      if (Test-Path ".\Visenlo\.git") {
        Write-Host "[Visenlo] Updating existing Visenlo\ ..." -ForegroundColor Cyan
        & git -C ".\Visenlo" pull --ff-only 2>$null
        $Root = "$(Get-Location)\Visenlo"
      } else {
        & git clone $RepoUrl Visenlo
        if ($LASTEXITCODE -ne 0) { Write-Host "[fail] git clone failed" -ForegroundColor Red; exit 1 }
        $Root = "$(Get-Location)\Visenlo"
      }
    } else {
      Write-Host "[Visenlo] git not found, downloading zip ..." -ForegroundColor Yellow
      $zip = "$env:TEMP\visenlo.zip"
      try { Invoke-WebRequest -Uri $RepoZip -OutFile $zip -UseBasicParsing } catch { Write-Host "[fail] Download failed: $_" -ForegroundColor Red; exit 1 }
      $dest = "$env:TEMP\visenlo_extract"
      if (Test-Path $dest) { Remove-Item -Recurse -Force $dest }
      Expand-Archive -Path $zip -DestinationPath $dest -Force
      $found = Get-ChildItem -Path $dest -Recurse -Filter "app.py" | Select-Object -First 1
      if ($found) {
        $dir = $found.DirectoryName
        if (-not (Test-Path ".\Visenlo")) { New-Item -ItemType Directory -Path ".\Visenlo" | Out-Null }
        Copy-Item -Path "$dir\*" -Destination ".\Visenlo" -Recurse -Force
        $Root = "$(Get-Location)\Visenlo"
      } else { Write-Host "[fail] Could not find app.py in zip" -ForegroundColor Red; exit 1 }
    }
  }
}
Write-Host "[Visenlo] Using project at: $Root" -ForegroundColor Cyan

# Reuse the common launcher; a healthy existing install needs no network.
$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
if (Test-Path $VenvPython) {
  & $VenvPython (Join-Path $Root "launcher.py") @args
  exit $LASTEXITCODE
}
foreach ($command in @("python", "python3", "py")) {
  if (Get-Command $command -ErrorAction SilentlyContinue) {
    & $command -c "import sys; sys.exit(sys.version_info < (3, 10))" 2>$null
    if ($LASTEXITCODE -eq 0) {
      & $command (Join-Path $Root "launcher.py") @args
      exit $LASTEXITCODE
    }
  }
}
Write-Host "[fail] Python 3.10 or newer is required. Install Python and run this script again." -ForegroundColor Red
exit 1
