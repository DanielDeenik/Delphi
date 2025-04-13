@echo off
REM Import data for AAPL
python %~dp0\import_aapl.py %*
pause
