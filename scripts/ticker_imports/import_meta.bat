@echo off
REM Import data for META
python %~dp0\import_meta.py %*
pause
