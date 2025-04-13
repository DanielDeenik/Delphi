@echo off
REM Launch the Delphi Trading Intelligence Dashboard on port 6000 and open all notebooks

echo Starting Delphi Trading Intelligence Dashboard on port 6000...
start python -m trading_ai.cli.dashboard_cli --port=6000 --browser

echo Opening all notebooks view...
timeout /t 3 > nul
start http://localhost:6000/colab/all

echo Press any key to exit...
pause
