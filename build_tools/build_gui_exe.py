"""
Build script to create Windows executable for Applied Probability Framework GUI.
Run this script to generate a standalone GUI executable.
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

def main():
    print("=" * 60)
    print("Applied Probability Framework GUI - EXE Builder")
    print("=" * 60)
    
    # Check if PyInstaller is installed
    try:
        import PyInstaller
        print("[OK] PyInstaller found")
    except ImportError:
        print("[INSTALLING] PyInstaller not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
        print("[OK] PyInstaller installed")
    
    # Paths
    build_tools_dir = Path(__file__).parent
    project_root = build_tools_dir.parent
    src_python = project_root / "src" / "python"
    dist_dir = project_root / "dist"
    build_dir = project_root / "build"
    spec_file = build_tools_dir / "applied_probability_framework_gui.spec"
    
    # Clean previous builds
    gui_dist = dist_dir / "AppliedProbabilityFrameworkGUI"
    gui_build = build_dir / "applied_probability_framework_gui"
    
    if gui_dist.exists():
        print(f"Cleaning {gui_dist}...")
        shutil.rmtree(gui_dist)
    if gui_build.exists():
        print(f"Cleaning {gui_build}...")
        shutil.rmtree(gui_build)
    
    print("\n" + "=" * 60)
    print("Building GUI executable...")
    print("=" * 60)
    
    # PyInstaller command
    cmd = [
        "pyinstaller",
        "--clean",
        "--noconfirm",
        str(spec_file)
    ]
    
    try:
        subprocess.run(cmd, check=True, cwd=build_tools_dir)
        print("\n" + "=" * 60)
        print("[SUCCESS] Build completed successfully!")
        print("=" * 60)
        print(f"\nExecutable location: {dist_dir / 'AppliedProbabilityFrameworkGUI' / 'AppliedProbabilityFrameworkGUI.exe'}")
        print("\nFeatures included:")
        print("  ✅ Monte Carlo simulation GUI")
        print("  ✅ Strategy selection")
        print("  ✅ Real-time progress monitoring")
        print("  ✅ Results visualization")
        print("\nYou can now distribute the entire 'AppliedProbabilityFrameworkGUI' folder.")
        print("Users can run the .exe without Python installed!")
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Build failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

