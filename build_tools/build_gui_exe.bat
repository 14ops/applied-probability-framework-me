@echo off
REM Quick build script for Windows GUI
REM Double-click this file to build the GUI executable

echo ============================================================
echo Applied Probability Framework GUI - EXE Builder
echo ============================================================
echo.

echo Checking Python installation...
python --version
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)

echo.
echo Building GUI executable...
echo.

python build_gui_exe.py

echo.
echo ============================================================
echo Build process completed!
echo ============================================================
echo.
pause

