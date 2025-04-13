@echo off
REM Generate import scripts for individual tickers
python %~dp0\generate_ticker_scripts.py
pause
