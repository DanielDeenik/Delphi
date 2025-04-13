@echo off
REM Import data for PFE
python %~dp0\import_pfe.py %*
pause
