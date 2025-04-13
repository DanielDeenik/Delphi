@echo off
REM Create a desktop shortcut for the Delphi Trading Intelligence System

echo Creating desktop shortcut for Delphi Trading Intelligence System...

REM Get the current directory
set CURRENT_DIR=%~dp0
set SHORTCUT_NAME=Delphi Trading Intelligence.lnk
set DESKTOP_PATH=%USERPROFILE%\Desktop

REM Create the shortcut
echo Set oWS = WScript.CreateObject("WScript.Shell") > CreateShortcut.vbs
echo sLinkFile = "%DESKTOP_PATH%\%SHORTCUT_NAME%" >> CreateShortcut.vbs
echo Set oLink = oWS.CreateShortcut(sLinkFile) >> CreateShortcut.vbs
echo oLink.TargetPath = "%CURRENT_DIR%launch_delphi.bat" >> CreateShortcut.vbs
echo oLink.WorkingDirectory = "%CURRENT_DIR%" >> CreateShortcut.vbs
echo oLink.Description = "Launch Delphi Trading Intelligence System" >> CreateShortcut.vbs
echo oLink.IconLocation = "%SystemRoot%\System32\SHELL32.dll,41" >> CreateShortcut.vbs
echo oLink.Save >> CreateShortcut.vbs

REM Run the VBScript to create the shortcut
cscript //nologo CreateShortcut.vbs
del CreateShortcut.vbs

echo Desktop shortcut created successfully!
echo.
echo You can now launch Delphi Trading Intelligence System by double-clicking the shortcut on your desktop.
echo.

pause
