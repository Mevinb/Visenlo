@echo off
REM Installation and offline launches share one maintained Python launcher.
call "%~dp0run.bat" %*
exit /b %errorlevel%
