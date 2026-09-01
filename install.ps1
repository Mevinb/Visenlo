# ReactorX — One-shot local installer for Windows (PowerShell)
# Usage:
#   powershell -ExecutionPolicy Bypass -File install.ps1
#   OR hosted: irm https://YOUR_WEBSITE/install.ps1 | iex
# What it does completely locally:
#   - Checks Python 3.10+
#   - Creates .venv, installs pip deps
#   - Checks models, runs self-check
#   - Launches Gradio on 127.0.0.1:7860 (100% local)

$ErrorActionPreference = "Stop"
$RepoUrl = if ($env:REACTORX_REPO) { $env:REACTORX_REPO } else { "https://github.com/Mevinb/Reactor-X.git" }
$RepoZip = if ($env:REACTORX_ZIP) { $env:REACTORX_ZIP } else { "https://github.com/Mevinb/Reactor-X/archive/refs/heads/main.zip" }

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not (Test-Path "$Root\app.py")) {
  if (Test-Path "$(Get-Location)\app.py") { $Root = Get-Location }
  elseif (Test-Path "$(Get-Location)\ReactorX\app.py") { $Root = "$(Get-Location)\ReactorX" }
  else {
    Write-Host "[ReactorX] Project files not found — fetching from $RepoUrl ..." -ForegroundColor Cyan
    if (Get-Command git -ErrorAction SilentlyContinue) {
      if (Test-Path ".\ReactorX\.git") {
        Write-Host "[ReactorX] Updating existing ReactorX\ ..." -ForegroundColor Cyan
        & git -C ".\ReactorX" pull --ff-only 2>$null
        $Root = "$(Get-Location)\ReactorX"
      } else {
        & git clone $RepoUrl ReactorX
        if ($LASTEXITCODE -ne 0) { Write-Host "[fail] git clone failed" -ForegroundColor Red; exit 1 }
        $Root = "$(Get-Location)\ReactorX"
      }
    } else {
      Write-Host "[ReactorX] git not found, downloading zip ..." -ForegroundColor Yellow
      $zip = "$env:TEMP\reactorx.zip"
      try { Invoke-WebRequest -Uri $RepoZip -OutFile $zip -UseBasicParsing } catch { Write-Host "[fail] Download failed: $_" -ForegroundColor Red; exit 1 }
      $dest = "$env:TEMP\reactorx_extract"
      if (Test-Path $dest) { Remove-Item -Recurse -Force $dest }
      Expand-Archive -Path $zip -DestinationPath $dest -Force
      $found = Get-ChildItem -Path $dest -Recurse -Filter "app.py" | Select-Object -First 1
      if ($found) {
        $dir = $found.DirectoryName
        if (-not (Test-Path ".\ReactorX")) { New-Item -ItemType Directory -Path ".\ReactorX" | Out-Null }
        Copy-Item -Path "$dir\*" -Destination ".\ReactorX" -Recurse -Force
        $Root = "$(Get-Location)\ReactorX"
      } else { Write-Host "[fail] Could not find app.py in zip" -ForegroundColor Red; exit 1 }
    }
  }
}
Write-Host "[ReactorX] Using project at: $Root" -ForegroundColor Cyan

function Write-Info($msg) { Write-Host "[ReactorX] $msg" -ForegroundColor Cyan }
function Write-Ok($msg)   { Write-Host "[ok] $msg" -ForegroundColor Green }
function Write-Warn($msg) { Write-Host "[warn] $msg" -ForegroundColor Yellow }
function Write-Fail($msg) { Write-Host "[fail] $msg" -ForegroundColor Red }

# --- 1. Find Python 3.10+ ---
$py = $null
foreach ($cmd in @("python","python3","py")) {
  try {
    $ver = & $cmd -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
    if ($LASTEXITCODE -eq 0) {
      $major = [int]($ver.Split('.')[0]); $minor = [int]($ver.Split('.')[1])
      if ($major -gt 3 -or ($major -eq 3 -and $minor -ge 10)) { $py = $cmd; break }
    }
  } catch {}
}
if (-not $py) {
  # try py launcher explicitly
  try { $v = & py -3.11 -c "import sys; print(sys.version)" 2>$null; if ($LASTEXITCODE -eq 0) { $py = "py -3.11" } } catch {}
}
if (-not $py) {
  Write-Fail "Python 3.10+ not found."
  Write-Host "  Install from https://www.python.org/downloads/ (check 'Add to PATH')"
  Write-Host "  Or via winget: winget install Python.Python.3.11"
  exit 1
}
Write-Ok "Found Python: $(& $py --version 2>&1) ($py)"

# Use py launcher if needed
$PyExec = $py
if ($py -like "py *") {
  $PyArgs = $py.Split(' ',2)[1]
  $PyExec = "py"
}

# --- 2. Create .venv ---
$Venv = Join-Path $Root ".venv"
$VenvPython = Join-Path $Venv "Scripts\python.exe"
if (-not (Test-Path $Venv)) {
  Write-Info "Creating virtual environment at $Venv ..."
  if ($PyExec -eq "py") { & py $PyArgs -m venv $Venv }
  else { & $PyExec -m venv $Venv }
  Write-Ok "Virtual environment created"
} else {
  Write-Ok "Using existing venv at $Venv"
}

if (-not (Test-Path $VenvPython)) {
  Write-Fail "Could not create virtual environment"
  exit 1
}

# --- 3. Install deps ---
Write-Info "Installing dependencies (2-5 minutes first time)..."
& $VenvPython -m pip install --upgrade pip -q
& $VenvPython -m pip uninstall -y onnxruntime opencv-python 2>$null | Out-Null
& $VenvPython -m pip install -r "$Root\requirements.txt"
if ($LASTEXITCODE -ne 0) { Write-Fail "pip install failed"; exit 1 }
Write-Ok "Dependencies installed"

# --- 4. System check ---
Write-Info "System check:"
& $VenvPython -c @"
import platform, sys
print(f'  OS: {platform.system()} {platform.release()}')
print(f'  Python: {sys.version.split()[0]}')
try:
    import onnxruntime as ort
    print(f'  ONNXRuntime: {ort.__version__} providers={ort.get_available_providers()}')
except Exception as e: print(f'  ONNXRuntime: {e}')
try:
    import cv2; print(f'  OpenCV: {cv2.__version__}')
except: pass
"@

# --- 5. Models ---
Write-Info "Checking models..."
if (Test-Path "$Root\scripts\download_models.py") {
  & $VenvPython "$Root\scripts\download_models.py" --check
} else {
  foreach ($f in @("models\inswapper_128.onnx","models\insightface\models\buffalo_l\det_10g.onnx")) {
    if (Test-Path "$Root\$f") { Write-Ok "found $f" } else { Write-Warn "missing $f (see README)" }
  }
}

# --- 6. Self-check ---
if (Test-Path "$Root\scripts\selfcheck.py") {
  Write-Info "Running self-check..."
  & $VenvPython "$Root\scripts\selfcheck.py"
  if ($LASTEXITCODE -eq 0) { Write-Ok "Self-check passed" } else { Write-Warn "Self-check had failures (optional models may be missing)" }
}

# --- 7. Launch ---
Write-Info "Starting ReactorX locally..."
Write-Host ""
Write-Host "  ReactorX will open at http://127.0.0.1:7860" -ForegroundColor Green
Write-Host "  All processing stays 100% on your device. No images leave your machine."
Write-Host "  Press Ctrl+C to stop."
Write-Host ""

$HostArg = "127.0.0.1"
$Port = 7860
# Allow passing --host / --port through
$extra = @()
for ($i=0; $i -lt $args.Count; $i++) {
  if ($args[$i] -eq "--host" -and $i+1 -lt $args.Count) { $HostArg = $args[$i+1]; $i++ }
  elseif ($args[$i] -like "--host=*") { $HostArg = $args[$i].Substring(7) }
  elseif ($args[$i] -eq "--port" -and $i+1 -lt $args.Count) { $Port = $args[$i+1]; $i++ }
  elseif ($args[$i] -like "--port=*") { $Port = $args[$i].Substring(7) }
  else { $extra += $args[$i] }
}

& $VenvPython "$Root\app.py" --host $HostArg --port $Port @extra
