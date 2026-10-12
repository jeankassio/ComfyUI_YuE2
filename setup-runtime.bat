@echo off
setlocal
cd /d "%~dp0"
set "YUE2_SETUP_PYTHON=%~dp0..\..\..\python_embeded\python.exe"
if exist "%YUE2_SETUP_PYTHON%" (
    "%YUE2_SETUP_PYTHON%" scripts\setup_runtime.py %*
) else (
    where py >nul 2>nul
    if errorlevel 1 (
        python scripts\setup_runtime.py %*
    ) else (
        py -3 scripts\setup_runtime.py %*
    )
)
if errorlevel 1 (
    echo.
    echo Setup failed. Review the message above.
    pause
    exit /b 1
)
echo.
echo Official YuE2 runtime is ready.
pause
