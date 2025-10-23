# Building Windows Executable (.exe)

This guide will help you create a standalone Windows executable for the Applied Probability Framework.

## Prerequisites

- Windows 10 or later
- Python 3.8 or higher installed
- All dependencies installed from `requirements.txt`

## Quick Start

### Option 1: Automated Build (Recommended)

Simply run the build script:

```bash
python build_exe.py
```

This will:
1. Install PyInstaller if needed
2. Clean previous builds
3. Create the executable
4. Place it in the `dist/AppliedProbabilityFramework/` folder

### Option 2: Manual Build

1. **Install PyInstaller:**
   ```bash
   pip install pyinstaller
   ```

2. **Build the executable:**
   ```bash
   pyinstaller --clean --noconfirm applied_probability_framework.spec
   ```

3. **Find your executable:**
   The `.exe` file will be in `dist/AppliedProbabilityFramework/AppliedProbabilityFramework.exe`

## Using the Executable

After building, you'll have a folder `dist/AppliedProbabilityFramework/` containing:
- `AppliedProbabilityFramework.exe` - The main executable
- Various DLL files and dependencies
- Configuration files

### Running the Application

Open Command Prompt or PowerShell and navigate to the `dist/AppliedProbabilityFramework/` folder:

```bash
# Show help
.\AppliedProbabilityFramework.exe --help

# Run a simulation with default settings
.\AppliedProbabilityFramework.exe run

# Run with custom configuration
.\AppliedProbabilityFramework.exe run --config my_config.json

# Run 10000 simulations in parallel
.\AppliedProbabilityFramework.exe run --num-simulations 10000 --parallel --jobs 8

# Create a default configuration file
.\AppliedProbabilityFramework.exe config --create default_config.json

# List available plugins
.\AppliedProbabilityFramework.exe plugins --list

# Analyze results
.\AppliedProbabilityFramework.exe analyze results/results_mines_basic.json
```

## Distribution

The entire `AppliedProbabilityFramework` folder can be:
- Zipped and shared
- Copied to other Windows computers
- Run without Python installation

**Note:** The folder contains all dependencies, so it might be 100-500 MB in size.

## Troubleshooting

### Build Fails with Import Errors

If PyInstaller can't find certain modules:

1. Make sure all dependencies are installed:
   ```bash
   pip install -r src/python/requirements.txt
   ```

2. Add missing modules to `hiddenimports` in `applied_probability_framework.spec`

### Executable Runs but Shows Errors

1. Check that all `.json` config files are included in the `datas` section of the spec file
2. Run the executable from Command Prompt to see error messages
3. Make sure you're distributing the entire folder, not just the `.exe`

### Antivirus False Positives

Some antivirus software may flag PyInstaller executables. This is a known issue. You may need to:
- Add an exception in your antivirus
- Digitally sign the executable (for professional distribution)

## Creating a Single-File Executable

To create a single `.exe` file (slower startup, but easier to distribute):

Edit the `exe` section in `applied_probability_framework.spec`:

```python
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,      # Add this
    a.zipfiles,      # Add this
    a.datas,         # Add this
    [],
    name='AppliedProbabilityFramework',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    coerce_archive_format=True,
)
```

Then remove the `COLLECT` section at the bottom.

## Advanced Options

### Custom Icon

1. Create or find a `.ico` file
2. Update the spec file:
   ```python
   icon='path/to/your/icon.ico'
   ```

### Reducing File Size

1. Exclude unused libraries in the spec file:
   ```python
   excludes=['tensorflow', 'torch', 'unnecessary_module']
   ```

2. Use UPX compression (already enabled)

3. Remove unused features from your code

### Creating an Installer

Use tools like:
- Inno Setup (free, recommended)
- NSIS
- Advanced Installer

These can create professional installers from your executable folder.

## Support

For issues or questions:
- Check the PyInstaller documentation: https://pyinstaller.org/
- Review error messages carefully
- Ensure all paths in the spec file are correct

