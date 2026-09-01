@echo off
REM ReactorX — One-shot local installer for Windows (CMD)
REM Usage: double-click install.bat  OR  install.bat --port 7860
REM For PowerShell users, prefer install.ps1

setlocal EnableDelayedExpansion
set ROOT=%~dp0
if not exist "%ROOT%app.py" set ROOT=%CD%\

echo [ReactorX] Windows CMD installer — checking Python...

python --version >nul 2>&1
if %errorlevel% neq 0 (
  echo [fail] Python not found in PATH.
  echo   Install from https://www.python.org/downloads/  (check "Add to PATH")
  echo   Or run: winget install Python.Python.3.11
  pause
  exit /b 1
)

for /f "tokens=2" %%v in ('python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"') do set PYVER=%%v
echo [ok] Found Python %PYVER%

if not exist "%ROOT%.venv" (
  echo [ReactorX] Creating virtual environment...
  python -m venv "%ROOT%.venv"
  if errorlevel 1 (
    echo [fail] Could not create venv. Try: python -m pip install --upgrade pip
    pause
    exit /b 1
  )
) else (
  echo [ok] Using existing .venv
)

if not exist "%ROOT%.venv\Scripts\python.exe" (
  echo [fail] venv python not found
  pause
  exit /b 1
)

echo [ReactorX] Installing dependencies (2-5 min first time)...
"%ROOT%.venv\Scripts\python.exe" -m pip install --upgrade pip -q
"%ROOT%.venv\Scripts\python.exe" -m pip uninstall -y onnxruntime opencv-python >nul 2>&1
"%ROOT%.venv\Scripts\python.exe" -m pip install -r "%ROOT%requirements.txt"
if errorlevel 1 (
  echo [fail] pip install failed
  pause
  exit /b 1
)
echo [ok] Dependencies installed

if exist "%ROOT%scripts\download_models.py" (
  "%ROOT%.venv\Scripts\python.exe" "%ROOT%scripts\download_models.py" --check
)

if exist "%ROOT%scripts\selfcheck.py" (
  echo [ReactorX] Running self-check...
  "%ROOT%.venv\Scripts\python.exe" "%ROOT%scripts\selfcheck.py"
)

echo.
echo   ReactorX will open at http://127.0.0.1:7860
echo   All processing stays 100%% on your device.
echo.

REM Pass through args to app.py (default host/port handling via app.py)
"%ROOT%.venv\Scripts\python.exe" "%ROOT%app.py" --host 127.0.0.1 --port 7860 %*
pause
