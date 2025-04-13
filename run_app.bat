@echo off
REM Launch the Delphi Trading Intelligence Dashboard on port 3000

echo Starting Delphi Trading Intelligence Dashboard on port 3000...
python -m trading_ai.cli.dashboard_cli --port=3000 --browser

echo Press any key to exit...
pause
