@echo off
setlocal
set "YUE2_PYTHON=%~dp0..\.venv\Scripts\python.exe"
if not exist "%YUE2_PYTHON%" set "YUE2_PYTHON=%~dp0..\..\..\..\python_embeded\python.exe"
if not exist "%YUE2_PYTHON%" (
    echo Python was not found. Run download_models.py with the YuE2 runtime Python.
    pause
    exit /b 1
)
"%YUE2_PYTHON%" "%~dp0download_models.py" --download %*
set "YUE2_RESULT=%ERRORLEVEL%"
if not "%YUE2_RESULT%"=="0" echo Download failed. Check the message above before retrying.
pause
exit /b %YUE2_RESULT%
