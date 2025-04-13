@echo off
REM Import data for TSLA
python %~dp0\import_tsla.py %*
pause
