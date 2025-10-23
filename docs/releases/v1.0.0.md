# Applied Probability Framework v1.0.0 - Windows Release

## 🎉 First Windows Executable Release!

This is the first official release of the Applied Probability Framework as a standalone Windows executable.

### ✨ What's New

- **Standalone Executable**: No Python installation required!
- **Complete CLI Interface**: Full command-line functionality
- **Professional Build**: Packaged with PyInstaller for optimal performance
- **All Dependencies Included**: NumPy, SciPy, and core libraries bundled

### 📦 Download

**File:** `AppliedProbabilityFramework-v1.0.0-Windows.zip` (20 MB)

**Requirements:**
- Windows 10 or later
- No additional software required

### 🚀 Quick Start

1. Download and extract the zip file
2. Open Command Prompt or PowerShell
3. Navigate to the extracted folder
4. Run: `AppliedProbabilityFramework.exe --help`

### 📖 Usage Examples

```powershell
# Show help
.\AppliedProbabilityFramework.exe --help

# Show version
.\AppliedProbabilityFramework.exe --version

# List available plugins
.\AppliedProbabilityFramework.exe plugins --list

# Create a default configuration
.\AppliedProbabilityFramework.exe config --create my_config.json

# Validate a configuration
.\AppliedProbabilityFramework.exe config --validate my_config.json

# Run simulations (with configured plugins)
.\AppliedProbabilityFramework.exe run --num-simulations 10000 --parallel

# Analyze results
.\AppliedProbabilityFramework.exe analyze results.json
```

### 🎯 Features

- **Monte Carlo Simulations**: Professional probability simulation framework
- **Bayesian Analysis**: Advanced statistical estimation
- **Parallel Processing**: Multi-core simulation support
- **Plugin System**: Extensible architecture for custom strategies
- **Configuration Management**: JSON-based configuration system
- **Result Analysis**: Comprehensive analysis and reporting

### 📋 What's Included

- `AppliedProbabilityFramework.exe` - Main executable (4.7 MB)
- All required DLLs and dependencies
- Configuration files (JSON)
- Python runtime (bundled)
- Total size: ~49 MB extracted, 20 MB compressed

### ⚠️ Known Issues

- Some antivirus software may flag the executable (false positive)
- Optional dependencies (matplotlib, plotly, pandas) not included in this build
- For full ML features, use the Python version

### 🔧 Building from Source

If you prefer to build the executable yourself:

```powershell
# Clone the repository
git clone https://github.com/14ops/applied-probability-framework-me.git
cd applied-probability-framework-me

# Install dependencies
pip install -r src/python/requirements.txt
pip install pyinstaller

# Build
python build_exe.py
```

See `BUILD_EXE_INSTRUCTIONS.md` for detailed build instructions.

### 📝 Changelog

#### Added
- Windows executable build support
- PyInstaller configuration
- Automated build scripts (`build_exe.py`, `build_exe.bat`)
- Comprehensive build documentation
- Release packaging

#### Technical Details
- **Build Tool**: PyInstaller 6.16.0
- **Python Version**: 3.13.7
- **Architecture**: Windows 64-bit
- **Build Type**: One-folder distribution
- **Compression**: UPX enabled

### 🐛 Reporting Issues

Found a bug? Please report it on our [Issues page](https://github.com/14ops/applied-probability-framework-me/issues).

When reporting issues with the executable:
1. Include the exact command you ran
2. Copy any error messages
3. Mention your Windows version
4. Note if antivirus software is running

### 📚 Documentation

- [API Reference](docs/API_REFERENCE.md)
- [Build Instructions](BUILD_EXE_INSTRUCTIONS.md)
- [Quick Start Guide](QUICKSTART.md)
- [Main README](README.md)

### 🙏 Credits

- **Author**: 14ops
- **Framework**: Applied Probability Framework
- **Build System**: PyInstaller

### 📄 License

MIT License - See [LICENSE](LICENSE) file for details

---

**Release Date**: October 23, 2025  
**Version**: 1.0.0  
**Platform**: Windows 10+ (64-bit)

**Download the zip file below and enjoy your standalone executable!** 🚀

