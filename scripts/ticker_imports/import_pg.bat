@echo off
REM Import data for PG
python %~dp0\import_pg.py %*
pause
