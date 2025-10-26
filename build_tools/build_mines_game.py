"""
Build script to create MinesGame.exe
Run this script to generate a standalone executable for the Mines Game.
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

def main():
    print("=" * 60)
    print("Mines Game - EXE Builder")
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
    spec_file = build_tools_dir / "mines_game.spec"
    
    # Clean previous builds
    mines_dist = dist_dir / "MinesGame"
    mines_build = build_dir / "mines_game"
    
    if mines_dist.exists():
        print(f"Cleaning {mines_dist}...")
        shutil.rmtree(mines_dist)
    if mines_build.exists():
        print(f"Cleaning {mines_build}...")
        shutil.rmtree(mines_build)
    
    print("\n" + "=" * 60)
    print("Building MinesGame executable...")
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
        print(f"\nExecutable location: {dist_dir / 'MinesGame' / 'MinesGame.exe'}")
        print("\nFeatures included:")
        print("  ✅ Real-time matrix visualization")
        print("  ✅ Q-Learning matrices")
        print("  ✅ Experience replay buffers")
        print("  ✅ Evolution parameters")
        print("  ✅ Save/Load learning progress")
        print("\nYou can now distribute the entire 'MinesGame' folder.")
        print("Users can run the .exe without Python installed!")
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Build failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

