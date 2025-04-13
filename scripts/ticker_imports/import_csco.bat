@echo off
REM Import data for CSCO
python %~dp0\import_csco.py %*
pause
