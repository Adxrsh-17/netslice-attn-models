# ============================================
# 5G-Traffic Dashboard - PowerShell Startup
# Group 5, AIE Batch B | 22AIE463
# ============================================

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  5G-Traffic Dashboard" -ForegroundColor White
Write-Host "  Group 5, AIE Batch B" -ForegroundColor Gray
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Navigate to the backend directory (relative to this script)
$ScriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Path
$BackendDir = Join-Path $ScriptDir "backend"
Set-Location $BackendDir

# Check Python installation
try {
    $pythonVersion = python --version 2>&1
    Write-Host "[OK] Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Python is not installed or not in PATH." -ForegroundColor Red
    Write-Host "Please install Python 3.9+ from https://www.python.org/" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Create virtual environment if not exists
if (-not (Test-Path "venv")) {
    Write-Host "[INFO] Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
    Write-Host "[OK] Virtual environment created." -ForegroundColor Green
}

# Activate virtual environment
Write-Host "[INFO] Activating virtual environment..." -ForegroundColor Yellow
& ".\venv\Scripts\Activate.ps1"

# Install dependencies
Write-Host "[INFO] Installing dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt --quiet

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Starting FastAPI Server..." -ForegroundColor White
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Dashboard: " -NoNewline; Write-Host "http://localhost:8000"         -ForegroundColor Green
Write-Host "  API Docs:  " -NoNewline; Write-Host "http://localhost:8000/api/docs" -ForegroundColor Green
Write-Host "  Health:    " -NoNewline; Write-Host "http://localhost:8000/api/health" -ForegroundColor Green
Write-Host ""
Write-Host "  Press Ctrl+C to stop the server." -ForegroundColor Yellow
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Open browser and start server
Start-Process "http://localhost:8000"
python main.py
