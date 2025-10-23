# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Applied Probability Framework.
This file defines how to package the application into a Windows executable.
"""

import os
import sys
from pathlib import Path

# Paths
project_root = Path.cwd()
src_python = project_root / 'src' / 'python'

# Add source directory to path
sys.path.insert(0, str(src_python))

block_cipher = None

# Collect all Python files from src/python
python_files = []
for root, dirs, files in os.walk(src_python):
    for file in files:
        if file.endswith('.py'):
            full_path = os.path.join(root, file)
            python_files.append(full_path)

# Define hidden imports (modules that PyInstaller might miss)
hidden_imports = [
    'numpy',
    'scipy',
    'scipy.stats',
    'scipy.special',
    'pandas',
    'matplotlib',
    'matplotlib.pyplot',
    'plotly',
    'plotly.graph_objs',
    'sklearn',
    'sklearn.ensemble',
    'sklearn.linear_model',
    # Framework modules
    'logger',
    'bayesian',
    'sim_compare',
    'meta_controller',
    'bankroll_manager',
    'game_simulator',
    'mathematical_core',
    'strategies',
    'utils',
    'advanced_strategies',
    'rintaro_okabe_strategy',
    'monte_carlo_tree_search',
    'markov_decision_process',
    'strategy_auto_evolution',
    'confidence_weighted_ensembles',
    'drl_environment',
    'drl_agent',
    'bayesian_mines',
    'human_data_collector',
    'adversarial_agent',
    'adversarial_detector',
    'adversarial_trainer',
    'multi_agent_core',
    'agent_comms',
    'multi_agent_simulator',
    'behavioral_value',
    'behavioral_probability',
    'visualizations.visualization',
    'visualizations.advanced_visualizations',
    'visualizations.comprehensive_visualizations',
    'visualizations.specialized_visualizations',
    'visualizations.realtime_heatmaps',
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
    [str(src_python / 'cli.py')],  # Main entry point
    pathex=[str(src_python)],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tensorflow',  # Exclude heavy ML libraries if not needed
        'torch',
        'pgmpy',
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
    name='AppliedProbabilityFramework',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # Console application
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
    name='AppliedProbabilityFramework',
)

