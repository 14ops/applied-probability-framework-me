"""
Build script to create Windows executable for Applied Probability Framework.
Run this script to generate a standalone .exe file.
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

def main():
    print("=" * 60)
    print("Applied Probability Framework - EXE Builder")
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
    project_root = Path(__file__).parent
    src_python = project_root / "src" / "python"
    dist_dir = project_root / "dist"
    build_dir = project_root / "build"
    
    # Clean previous builds
    if dist_dir.exists():
        print(f"Cleaning {dist_dir}...")
        shutil.rmtree(dist_dir)
    if build_dir.exists():
        print(f"Cleaning {build_dir}...")
        shutil.rmtree(build_dir)
    
    print("\n" + "=" * 60)
    print("Building executable...")
    print("=" * 60)
    
    # PyInstaller command
    cmd = [
        "pyinstaller",
        "--clean",
        "--noconfirm",
        "applied_probability_framework.spec"
    ]
    
    try:
        subprocess.run(cmd, check=True, cwd=project_root)
        print("\n" + "=" * 60)
        print("[SUCCESS] Build completed successfully!")
        print("=" * 60)
        print(f"\nExecutable location: {dist_dir / 'AppliedProbabilityFramework' / 'AppliedProbabilityFramework.exe'}")
        print("\nYou can now distribute the entire 'AppliedProbabilityFramework' folder.")
        print("Users can run the .exe without Python installed!")
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Build failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

