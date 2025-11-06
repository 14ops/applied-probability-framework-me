"""
Launcher script for Paper Trading Dashboard
Run: python run_dashboard.py
"""

import subprocess
import sys
from pathlib import Path
import os

def ensure_env_and_requirements(project_root: Path) -> str:
    """Ensure a venv exists and requirements are installed. Return python exe path."""
    venv_dir = project_root / ".venv"
    if os.name == 'nt':
        python_exe = venv_dir / "Scripts" / "python.exe"
        pip_exe = venv_dir / "Scripts" / "pip.exe"
    else:
        python_exe = venv_dir / "bin" / "python"
        pip_exe = venv_dir / "bin" / "pip"

    # Create venv if missing
    if not python_exe.exists():
        print("🧪 Creating virtual environment (.venv)...")
        subprocess.check_call([sys.executable, "-m", "venv", str(venv_dir)])

    # Install requirements if key packages are missing
    needs_install = False
    for mod in ("streamlit", "alpaca_trade_api"):
        try:
            subprocess.check_call([str(python_exe), "-c", f"import {mod}"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            needs_install = True
            break
    if needs_install:
        print("📦 Installing requirements (this may take a few minutes)...")
        req = project_root / "requirements.txt"
        subprocess.check_call([str(pip_exe), "install", "-r", str(req)])

    return str(python_exe)


if __name__ == "__main__":
    root = Path(__file__).parent
    dashboard_path = root / "stock_ai" / "dashboard_trading.py"

    python_exe = ensure_env_and_requirements(root)

    print("🚀 Starting Paper Trading Dashboard...")
    print("📊 Dashboard will open in your browser")
    print("⏹️  Press Ctrl+C to stop")
    print("-" * 60)

    try:
        subprocess.run([
            python_exe, "-m", "streamlit", "run",
            str(dashboard_path),
            "--server.port", "8501",
            "--server.address", "localhost",
        ])
    except KeyboardInterrupt:
        print("\n✅ Dashboard stopped")

