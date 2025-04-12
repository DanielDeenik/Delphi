@echo off
echo Running Time Series Import for Delphi Trading Intelligence System
echo ============================================================

REM Set environment variables
set PYTHONPATH=%~dp0
set GOOGLE_APPLICATION_CREDENTIALS=%APPDATA%\gcloud\application_default_credentials.json

REM Check if Python is installed
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Python is not installed or not in PATH
    exit /b 1
)

REM Run the import script
python run_time_series_import.py %*

echo ============================================================
echo Import process completed with exit code %ERRORLEVEL%
