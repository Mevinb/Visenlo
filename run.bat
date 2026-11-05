@echo off
setlocal
set "VISENLO_ROOT=%~dp0"
if exist "%VISENLO_ROOT%.venv\Scripts\python.exe" goto venv
where py >nul 2>nul
if not errorlevel 1 goto py
where python >nul 2>nul
if not errorlevel 1 goto python
echo Visenlo needs Python 3.11 or 3.12. Install it from python.org with Add Python to PATH enabled.
exit /b 1

:venv
"%VISENLO_ROOT%.venv\Scripts\python.exe" "%VISENLO_ROOT%launcher.py" %*
exit /b %errorlevel%

:py
py -3 "%VISENLO_ROOT%launcher.py" %*
exit /b %errorlevel%

:python
python "%VISENLO_ROOT%launcher.py" %*
exit /b %errorlevel%
