"""
Main CLI entry point for the Applied Probability Framework.

This module provides the primary command-line interface with comprehensive
subcommands for all framework functionality.

Author: Shihab Belal
License: MIT
Version: 2.0.0
"""

import argparse
import sys
import json
import os
from typing import Optional, Dict, Any
from pathlib import Path

from .commands import (
    run_simulation,
    analyze_results,
    create_config,
    list_strategies,
    benchmark_performance
)
from ..core.base import SimulationConfig
from ..core.probability_engine import ProbabilityEngine
from ..core.statistics import StatisticsEngine


def create_parser() -> argparse.ArgumentParser:
    """
    Create the main argument parser.
    
    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        prog='applied-probability-framework',
        description='Applied Probability and Automation Framework for High-RTP Games',
        epilog='For more information, visit: https://github.com/your-repo/applied-probability-framework',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 2.0.0'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='count',
        default=0,
        help='Increase verbosity level (can be used multiple times)'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output directory for results'
    )
    
    # Create subparsers
    subparsers = parser.add_subparsers(
        dest='command',
        help='Available commands',
        required=True
    )
    
    # Run simulation command
    run_parser = subparsers.add_parser(
        'run',
        help='Run Monte Carlo simulations'
    )
    run_parser.add_argument(
        '--simulations', '-n',
        type=int,
        default=1000,
        help='Number of simulations to run (default: 1000)'
    )
    run_parser.add_argument(
        '--iterations', '-i',
        type=int,
        default=100,
        help='Maximum iterations per simulation (default: 100)'
    )
    run_parser.add_argument(
        '--strategy', '-s',
        type=str,
        required=True,
        help='Strategy to use for simulation'
    )
    run_parser.add_argument(
        '--game-model', '-g',
        type=str,
        required=True,
        help='Game model to simulate'
    )
    run_parser.add_argument(
        '--parallel', '-p',
        action='store_true',
        help='Enable parallel processing'
    )
    run_parser.add_argument(
        '--workers', '-w',
        type=int,
        default=4,
        help='Number of parallel workers (default: 4)'
    )
    run_parser.add_argument(
        '--seed',
        type=int,
        help='Random seed for reproducibility'
    )
    run_parser.add_argument(
        '--timeout',
        type=float,
        help='Timeout in seconds'
    )
    
    # Analyze results command
    analyze_parser = subparsers.add_parser(
        'analyze',
        help='Analyze simulation results'
    )
    analyze_parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Input file with simulation results'
    )
    analyze_parser.add_argument(
        '--output-format', '-f',
        choices=['json', 'csv', 'html', 'pdf'],
        default='json',
        help='Output format for analysis results'
    )
    analyze_parser.add_argument(
        '--confidence-level',
        type=float,
        default=0.95,
        help='Confidence level for statistical analysis (default: 0.95)'
    )
    analyze_parser.add_argument(
        '--convergence-analysis',
        action='store_true',
        help='Perform convergence analysis'
    )
    analyze_parser.add_argument(
        '--sensitivity-analysis',
        action='store_true',
        help='Perform sensitivity analysis'
    )
    
    # Create config command
    config_parser = subparsers.add_parser(
        'config',
        help='Create or manage configuration files'
    )
    config_parser.add_argument(
        '--create',
        action='store_true',
        help='Create a new configuration file'
    )
    config_parser.add_argument(
        '--template',
        choices=['basic', 'advanced', 'custom'],
        default='basic',
        help='Configuration template to use'
    )
    config_parser.add_argument(
        '--output-file', '-o',
        type=str,
        help='Output file for configuration'
    )
    config_parser.add_argument(
        '--validate',
        type=str,
        help='Validate an existing configuration file'
    )
    
    # List strategies command
    list_parser = subparsers.add_parser(
        'list',
        help='List available strategies and game models'
    )
    list_parser.add_argument(
        '--type',
        choices=['strategies', 'models', 'all'],
        default='all',
        help='Type of components to list'
    )
    list_parser.add_argument(
        '--format',
        choices=['table', 'json', 'yaml'],
        default='table',
        help='Output format'
    )
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser(
        'benchmark',
        help='Benchmark framework performance'
    )
    benchmark_parser.add_argument(
        '--test-suite',
        choices=['basic', 'comprehensive', 'stress'],
        default='basic',
        help='Benchmark test suite to run'
    )
    benchmark_parser.add_argument(
        '--iterations',
        type=int,
        default=10,
        help='Number of benchmark iterations'
    )
    benchmark_parser.add_argument(
        '--output-file', '-o',
        type=str,
        help='Output file for benchmark results'
    )
    
    return parser


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If config file is invalid JSON
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """
    Save configuration to file.
    
    Args:
        config: Configuration dictionary
        output_path: Output file path
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)


def setup_logging(verbose_level: int) -> None:
    """
    Setup logging configuration.
    
    Args:
        verbose_level: Verbosity level (0-3)
    """
    import logging
    
    level_map = {
        0: logging.WARNING,
        1: logging.INFO,
        2: logging.DEBUG,
        3: logging.DEBUG
    }
    
    level = level_map.get(verbose_level, logging.INFO)
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def main() -> int:
    """
    Main CLI entry point.
    
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    try:
        # Load configuration if provided
        config = {}
        if args.config:
            config = load_config(args.config)
        
        # Set output directory
        output_dir = args.output or os.getcwd()
        os.makedirs(output_dir, exist_ok=True)
        
        # Execute command
        if args.command == 'run':
            return run_simulation(args, config, output_dir)
        elif args.command == 'analyze':
            return analyze_results(args, config, output_dir)
        elif args.command == 'config':
            return create_config(args, config, output_dir)
        elif args.command == 'list':
            return list_strategies(args, config, output_dir)
        elif args.command == 'benchmark':
            return benchmark_performance(args, config, output_dir)
        else:
            parser.print_help()
            return 1
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        return 130
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose > 0:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())