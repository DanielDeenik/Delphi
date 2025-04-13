@echo off
REM Delphi Trading Intelligence System Launcher
REM This script automatically launches the Delphi application and opens the browser

echo ======================================================
echo    Delphi Trading Intelligence System Launcher
echo ======================================================
echo.

REM Check if Python is installed
python --version > nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python is not installed or not in the PATH.
    echo Please install Python 3.8 or higher and try again.
    goto :end
)

echo [1/4] Setting up environment...
REM Create necessary directories
if not exist logs mkdir logs
if not exist status mkdir status
if not exist templates mkdir templates
if not exist static mkdir static
if not exist config mkdir config

REM Check if config file exists
if not exist config\config.json (
    echo [INFO] Creating default configuration...
    echo {^
  "alpha_vantage": {^
    "api_key": "IAS7UEKOT0HZW0MY",^
    "base_url": "https://www.alphavantage.co/query"^
  },^
  "google_cloud": {^
    "project_id": "delphi-449908",^
    "dataset": "stock_data"^
  },^
  "tickers": [^
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", ^
    "TSLA", "NVDA", "JPM", "V", "JNJ",^
    "WMT", "PG", "MA", "UNH", "HD",^
    "BAC", "PFE", "CSCO", "DIS", "VZ"^
  ]^
} > config\config.json
)

echo [2/4] Starting Delphi application on port 3000...
REM Start the Flask application in the background
start /B python -m trading_ai.cli.dashboard_cli --port 3000 > logs\app_%date:~-4,4%%date:~-7,2%%date:~-10,2%.log 2>&1

REM Check if the application started successfully
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to start the application. Check the logs for details.
    goto :end
)

echo [3/4] Waiting for application to start...
REM Wait for the application to start
timeout /t 5 > nul

echo [4/4] Opening browser...
REM Open the browser to the multi-tab Colab view
REM Try to open the browser, but don't fail if it doesn't work
start "" http://localhost:3000/colab/all 2>nul || (
    echo [WARNING] Could not open browser automatically.
    echo [WARNING] Please open http://localhost:3000/colab/all manually in your browser.
)

echo.
echo ======================================================
echo    Delphi Trading Intelligence System is running
echo    Dashboard URL: http://localhost:3000
echo    Notebooks URL: http://localhost:3000/colab
echo    All Notebooks: http://localhost:3000/colab/all
echo ======================================================
echo.
echo Press Ctrl+C to stop the application
echo.

REM Keep the window open
pause

:end
