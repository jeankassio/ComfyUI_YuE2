@echo off
setlocal
set "YUE2_PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%YUE2_PYTHON%" set "YUE2_PYTHON=%~dp0..\..\..\python_embeded\python.exe"
if not exist "%YUE2_PYTHON%" (
    echo Python was not found. Run setup-runtime.bat first.
    pause
    exit /b 1
)
"%YUE2_PYTHON%" "%~dp0scripts\convert_official_models.py" %*
set "YUE2_RESULT=%ERRORLEVEL%"
if not "%YUE2_RESULT%"=="0" echo Model adaptation failed. Check the message above.
pause
exit /b %YUE2_RESULT%
