@echo off
REM Import data for JPM
python %~dp0\import_jpm.py %*
pause
