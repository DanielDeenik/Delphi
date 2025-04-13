@echo off
REM Import data for WMT
python %~dp0\import_wmt.py %*
pause
