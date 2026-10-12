@echo off
setlocal
cd /d "%~dp0.."
set "YUE2_SETUP_PYTHON=%~dp0..\..\..\..\python_embeded\python.exe"
if exist "%YUE2_SETUP_PYTHON%" (
    "%YUE2_SETUP_PYTHON%" scripts\setup_sheetsage.py %*
) else (
    where py >nul 2>nul
    if not errorlevel 1 (
        py -3 scripts\setup_sheetsage.py %*
    ) else (
        python scripts\setup_sheetsage.py %*
    )
)
if errorlevel 1 (
    echo.
    echo SheetSage2 setup failed. Read the error above.
    pause
    exit /b 1
)
echo.
echo SheetSage2 setup completed.
pause
