# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Mines Game GUI.
Beautiful playable game interface.
"""

import os
import sys
from pathlib import Path

# Paths
# Get the directory where this spec file is located
try:
    # When running as a spec file, __file__ is available
    spec_dir = Path(__file__).parent
except NameError:
    # Fallback: assume running from build_tools directory
    spec_dir = Path.cwd()
    if spec_dir.name != 'build_tools':
        spec_dir = spec_dir / 'build_tools'
project_root = spec_dir.parent
src_python = project_root / 'src' / 'python'

# Add source directory to path
sys.path.insert(0, str(src_python))

block_cipher = None

# Define hidden imports
hidden_imports = [
    'tkinter',
    'tkinter.ttk',
    'tkinter.messagebox',
    'tkinter.filedialog',
    'tkinter.scrolledtext',
    'game_simulator',
    'matrix_visualizer',
    'register_plugins',
    'adaptive_takeshi',
    'core',
    'core.base_estimator',
    'core.base_simulator',
    'core.base_strategy',
    'core.config',
    'core.plugin_system',
    'core.evolution_matrix',
    'core.adaptive_strategy',
    'numpy',
]

a = Analysis(
    [str(src_python / 'gui_game.py')],
    pathex=[str(src_python)],
    binaries=[],
    datas=[],
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['matplotlib', 'scipy', 'pandas', 'plotly', 'sklearn', 'tensorflow', 'torch'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='MinesGame',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # NO console window!
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    coerce_archive_format=True,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='MinesGame',
)

