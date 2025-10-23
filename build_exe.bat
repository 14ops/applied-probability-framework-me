@echo off
REM Quick build script for Windows
REM Double-click this file to build the executable

echo ============================================================
echo Applied Probability Framework - EXE Builder
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
echo Building executable...
echo.

python build_exe.py

echo.
echo ============================================================
echo Build process completed!
echo ============================================================
echo.
pause

