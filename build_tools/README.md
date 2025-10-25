# Build Tools

This folder contains scripts and configuration for building distribution packages.

## Files

- **`build_exe.py`** - Python script to build Windows executable
- **`build_exe.bat`** - Windows batch file for one-click building
- **`applied_probability_framework.spec`** - PyInstaller configuration file

## Building Windows Executable

### Quick Build

```bash
cd build_tools
python build_exe.py
```

Or double-click `build_exe.bat` on Windows.

### Manual Build

```bash
cd build_tools
pyinstaller --clean --noconfirm applied_probability_framework.spec
```

## Output

The executable will be created in `dist/AppliedProbabilityFramework/`

## Documentation

See [docs/build/EXECUTABLE_BUILD.md](../docs/build/EXECUTABLE_BUILD.md) for detailed build instructions.

## Requirements

- Python 3.8+
- PyInstaller 6.0+
- All project dependencies installed

Install PyInstaller:
```bash
pip install pyinstaller
```


