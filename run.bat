@echo off
REM ReactorX local launcher for Windows (CMD)
REM Usage: run.bat [--host 127.0.0.1] [--port 7860] [--share]

setlocal
set ROOT=%~dp0

if not exist "%ROOT%.venv\Scripts\python.exe" (
  echo [ReactorX] No virtual environment found — running installer...
  call "%ROOT%install.bat" %*
  exit /b %errorlevel%
)

REM Handle --host / --port args manually, default 127.0.0.1:7860
set HOST=127.0.0.1
set PORT=7860
set APP_ARGS=

:parse
if "%~1"=="" goto run
if "%~1"=="--host" (
  set HOST=%~2
  shift & shift & goto parse
)
if "%~1"=="--host=*" (
  set HOST=%~1
  set HOST=!HOST:--host=!
  shift & goto parse
)
if "%~1"=="--port" (
  set PORT=%~2
  shift & shift & goto parse
)
if "%~1"=="--port=*" (
  set PORT=%~1
  set PORT=!PORT:--port=!
  shift & goto parse
)
REM collect remaining args (e.g. --share)
set APP_ARGS=%APP_ARGS% %1
shift
goto parse

:run
echo.
echo   ReactorX running 100%% locally
echo   URL: http://%HOST%:%PORT%
echo   Models: %ROOT%models
echo.

REM Ensure deps are present (quick, idempotent)
"%ROOT%.venv\Scripts\python.exe" -m pip uninstall -y onnxruntime opencv-python >nul 2>&1
"%ROOT%.venv\Scripts\python.exe" -m pip install -q -r "%ROOT%requirements.txt" >nul 2>&1

"%ROOT%.venv\Scripts\python.exe" "%ROOT%app.py" --host %HOST% --port %PORT% %APP_ARGS%
