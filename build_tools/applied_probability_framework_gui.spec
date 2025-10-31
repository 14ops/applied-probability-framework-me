# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Applied Probability Framework GUI.
This builds a Windows GUI application with a graphical interface.
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

# Define hidden imports (modules that PyInstaller might miss)
hidden_imports = [
    'numpy',
    'tkinter',
    'tkinter.ttk',
    'tkinter.scrolledtext',
    'tkinter.filedialog',
    'tkinter.messagebox',
    # Framework modules
    'logger',
    'bayesian',
    'game_simulator',
    'strategies',
    'utils',
    'register_plugins',
    'core',
    'core.base_estimator',
    'core.base_simulator',
    'core.base_strategy',
    'core.config',
    'core.parallel_engine',
    'core.plugin_system',
    'core.reproducibility',
]

# Data files to include (JSON configs, etc.)
datas = [
    (str(src_python / 'behavioral_config.json'), '.'),
    (str(src_python / 'drl_config.json'), '.'),
    (str(src_python / 'multi_agent_config.json'), '.'),
    (str(src_python / 'test_config.json'), '.'),
]

# Add visualization files if they exist
vis_dir = src_python / 'visualizations'
if vis_dir.exists():
    datas.append((str(vis_dir), 'visualizations'))

# Add core module
core_dir = src_python / 'core'
if core_dir.exists():
    datas.append((str(core_dir), 'core'))

a = Analysis(
    [str(src_python / 'gui.py')],  # GUI entry point
    pathex=[str(src_python)],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tensorflow',
        'torch',
        'pgmpy',
        'scipy',
        'pandas',
        'matplotlib',
        'plotly',
        'sklearn',
    ],
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
    name='AppliedProbabilityFrameworkGUI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # NO console window for GUI!
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    coerce_archive_format=True,
    icon=None,  # Add an .ico file path here if you have one
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='AppliedProbabilityFrameworkGUI',
)

