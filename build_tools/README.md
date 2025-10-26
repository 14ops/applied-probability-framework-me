# Build Tools

This folder contains scripts and configuration for building distribution packages.

## Files

### Framework Executable
- **`build_exe.py`** - Python script to build Windows executable
- **`build_exe.bat`** - Windows batch file for one-click building
- **`applied_probability_framework.spec`** - PyInstaller configuration file

### Mines Game Executable (NEW!)
- **`build_mines_game.py`** - Python script to build MinesGame.exe
- **`build_mines_game.bat`** - Windows batch file for one-click game building
- **`mines_game.spec`** - PyInstaller configuration file for game
- **Includes:**
  - ✅ Real-time matrix visualization
  - ✅ Q-Learning matrices display
  - ✅ Experience replay buffer viewer
  - ✅ Evolution parameters monitor
  - ✅ Save/Load learning progress

## Building Windows Executables

### Mines Game (Recommended)

**Quick Build:**
```bash
cd build_tools
python build_mines_game.py
```

Or double-click `build_mines_game.bat` on Windows.

**Manual Build:**
```bash
cd build_tools
pyinstaller --clean --noconfirm mines_game.spec
```

**Output:** `dist/MinesGame/MinesGame.exe`

### Framework GUI

**Quick Build:**
```bash
cd build_tools
python build_exe.py
```

Or double-click `build_exe.bat` on Windows.

**Manual Build:**
```bash
cd build_tools
pyinstaller --clean --noconfirm applied_probability_framework.spec
```

**Output:** `dist/AppliedProbabilityFramework/`

## Documentation

See [docs/build/EXECUTABLE_BUILD.md](../docs/build/EXECUTABLE_BUILD.md) for detailed build instructions.

## Requirements

- Python 3.8+
- PyInstaller 6.0+
- NumPy
- All project dependencies installed

Install PyInstaller:
```bash
pip install pyinstaller numpy
```

## New Features in MinesGame

The MinesGame.exe now includes real-time AI learning visualization:

- **Matrix Visualization Panel** - Opens in a separate window
- **Q-Learning Tab** - Shows learned state-action values
- **Experience Replay Tab** - Displays stored game experiences
- **Evolution Tab** - Monitors parameter evolution
- **Learning Stats Tab** - Overall learning metrics
- **Save/Load Progress** - Persist learning across sessions

To use these features:
1. Build and run MinesGame.exe
2. Select an AI strategy (Takeshi, Lelouch, Senku, etc.)
3. Click "Show Learning Matrices" to open visualization
4. Enable "Auto-Refresh" to see real-time updates
5. Use "Save Learning Progress" to save to disk
6. Use "Load Learning Progress" to continue training


