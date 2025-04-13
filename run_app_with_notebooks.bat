@echo off
REM Launch the Delphi Trading Intelligence Dashboard on port 3000 and open all notebooks

echo Starting Delphi Trading Intelligence Dashboard on port 3000...
start python -m trading_ai.cli.dashboard_cli --port=3000 --browser

echo Opening all notebooks view...
timeout /t 3 > nul
start http://localhost:3000/colab/all

echo Press any key to exit...
pause
