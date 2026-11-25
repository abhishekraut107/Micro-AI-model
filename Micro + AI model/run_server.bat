@echo off
title BioSentinel AI - Surveillance & Digital Twin Platform
cls
echo =========================================================================
echo       BioSentinel AI: Clinical AMR Surveillance & Digital Twin
echo =========================================================================
echo.
echo [1/2] Checking Python environment...
python --version
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH! Please install Python 3.9+.
    pause
    exit /b 1
)

echo.
echo [2/2] Starting BioSentinel AI server on http://127.0.0.1:8000 ...
echo       Press CTRL+C in this terminal to stop the server at any time.
echo.

cd /d "%~dp0"
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload

pause
