@echo off
REM ============================================
REM 5G-Traffic Dashboard - Startup Script
REM Group 5, AIE Batch B | 22AIE463
REM ============================================

echo.
echo ============================================
echo   5G-Traffic Dashboard
echo   Group 5, AIE Batch B
echo ============================================
echo.

REM Navigate to backend directory (relative to this script)
cd /d "%~dp0backend"

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.9+ from https://www.python.org/
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist "venv" (
    echo [INFO] Creating virtual environment...
    python -m venv venv
    echo [INFO] Virtual environment created.
)

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo [INFO] Installing dependencies...
pip install -r requirements.txt --quiet

REM Start the server
echo.
echo ============================================
echo   Starting FastAPI Server...
echo ============================================
echo.
echo   Dashboard: http://localhost:8000
echo   API Docs:  http://localhost:8000/api/docs
echo   Health:    http://localhost:8000/api/health
echo.
echo   Press Ctrl+C to stop the server.
echo ============================================
echo.

python main.py

pause
