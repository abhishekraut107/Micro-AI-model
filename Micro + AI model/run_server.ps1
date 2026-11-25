# BioSentinel AI - PowerShell Startup Script
Write-Host "=========================================================================" -ForegroundColor Cyan
Write-Host "      BioSentinel AI: Clinical AMR Surveillance & Digital Twin           " -ForegroundColor Green
Write-Host "=========================================================================" -ForegroundColor Cyan
Write-Host ""

# Ensure we are in the script's root directory
Set-Location -Path $PSScriptRoot

# Check Python availability
try {
    $pyVersion = python --version
    Write-Host "[✓] Python detected: $pyVersion" -ForegroundColor Green
} catch {
    Write-Host "[X] Python is not found in PATH. Please install Python 3.9+." -ForegroundColor Red
    exit 1
}

Write-Host "[✓] Starting BioSentinel AI server on http://127.0.0.1:8000 ..." -ForegroundColor Cyan
Write-Host "    Press CTRL+C to stop the server.`n" -ForegroundColor Yellow

python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
