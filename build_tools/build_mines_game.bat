@echo off
REM Build script for MinesGame.exe
REM Double-click this file to build the game executable

echo ============================================================
echo Mines Game - EXE Builder  
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
echo Building MinesGame executable with matrix visualization...
echo.

python build_mines_game.py

echo.
echo ============================================================
echo Build process completed!
echo ============================================================
echo.
pause

