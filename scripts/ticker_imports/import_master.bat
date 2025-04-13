@echo off
REM Import data for MASTER
python %~dp0\import_master.py %*
pause
