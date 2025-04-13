@echo off
REM Import data for BAC
python %~dp0\import_bac.py %*
pause
