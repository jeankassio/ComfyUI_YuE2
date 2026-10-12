@echo off
call "%~dp0..\setup-runtime.bat" %*
exit /b %ERRORLEVEL%
