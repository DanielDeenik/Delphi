@echo off
REM Batch script for running the integrated import system on Windows
REM This script integrates with the existing codebase

echo Running Delphi Integrated Import System
echo =====================================

REM Check if Python is available
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Python not found in PATH. Please install Python or add it to your PATH.
    exit /b 1
)

REM Pass all arguments to the Python script
python scripts\run_integrated_import.py %*

REM Check exit code
if %ERRORLEVEL% neq 0 (
    echo Import process failed with exit code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

echo Import process completed successfully
exit /b 0
