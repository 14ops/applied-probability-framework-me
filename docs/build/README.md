# Build Documentation

This folder contains documentation for building and releasing the Applied Probability Framework.

## Contents

- **[EXECUTABLE_BUILD.md](EXECUTABLE_BUILD.md)** - Instructions for building Windows executable (.exe)
- **[PYPI_RELEASE.md](PYPI_RELEASE.md)** - Instructions for publishing to PyPI

## Quick Links

### Building Windows Executable

```bash
cd build_tools
python build_exe.py
```

See [EXECUTABLE_BUILD.md](EXECUTABLE_BUILD.md) for detailed instructions.

### Build Tools Location

All build scripts are in the `build_tools/` directory at the project root:
- `build_exe.py` - Automated build script
- `build_exe.bat` - Windows batch file
- `applied_probability_framework.spec` - PyInstaller configuration

## Release Notes

Release notes for all versions are in [`docs/releases/`](../releases/)


