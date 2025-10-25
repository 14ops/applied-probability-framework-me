"""
Command-line interface for the Applied Probability Framework.

This module provides a professional CLI with comprehensive options for running
simulations, managing configurations, and analyzing results.
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Optional, List
import logging

from core.config import (
    FrameworkConfig, SimulationConfig, GameConfig, 
    StrategyConfig, EstimatorConfig, create_default_config
)
from core.plugin_system import get_registry
from core.parallel_engine import ParallelSimulationEngine

# Import to register all built-in plugins
import register_plugins


def setup_logging(log_level: str) -> None:
    """
    Configure logging for the application.
    
    Args:
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def cmd_run(args: argparse.Namespace) -> int:
    """
    Execute simulation runs.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Exit code (0 for success)
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting simulation run")
    
    # Load configuration
    if args.config:
        logger.info(f"Loading configuration from {args.config}")
        config = FrameworkConfig.from_json(args.config)
    else:
        logger.info("Using default configuration")
        config = create_default_config()
    
    # Override config with CLI arguments
    if args.num_simulations:
        config.simulation.num_simulations = args.num_simulations
    if args.seed is not None:
        config.simulation.seed = args.seed
    if args.parallel is not None:
        config.simulation.parallel = args.parallel
    if args.jobs:
        config.simulation.n_jobs = args.jobs
    if args.output_dir:
        config.simulation.output_dir = args.output_dir
    
    setup_logging(config.simulation.log_level)
    
    # Get registry and create simulator/strategy
    registry = get_registry()
    
    simulator_name = args.simulator or 'mines'
    strategy_name = args.strategy or (config.strategies[0].name if config.strategies else 'basic')
    
    logger.info(f"Using simulator: {simulator_name}, strategy: {strategy_name}")
    
    try:
        simulator = registry.create_simulator(simulator_name, config=config.game.params)
        strategy = registry.create_strategy(strategy_name)
    except KeyError as e:
        logger.error(f"Plugin not found: {e}")
        logger.info(f"Available simulators: {registry.list_simulators()}")
        logger.info(f"Available strategies: {registry.list_strategies()}")
        return 1
    
    # Run simulations
    engine = ParallelSimulationEngine(config, n_jobs=config.simulation.n_jobs)
    
    logger.info(f"Running {config.simulation.num_simulations} simulations...")
    results = engine.run_batch(simulator, strategy, show_progress=not args.quiet)
    
    # Display results
    if not args.quiet:
        print("\n" + "="*60)
        print("SIMULATION RESULTS")
        print("="*60)
        print(f"Total runs:        {results.num_runs}")
        print(f"Mean reward:       {results.mean_reward:.4f}")
        print(f"Std deviation:     {results.std_reward:.4f}")
        print(f"95% CI:            [{results.confidence_interval[0]:.4f}, {results.confidence_interval[1]:.4f}]")
        print(f"Success rate:      {results.success_rate:.2%}")
        print(f"Execution time:    {results.total_duration:.2f}s")
        print(f"Converged:         {'Yes' if results.convergence_achieved else 'No'}")
        print("="*60)
    
    # Save results if requested
    if config.simulation.save_results:
        output_dir = Path(config.simulation.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"results_{simulator_name}_{strategy_name}.json"
        
        results_dict = {
            'config': config.to_dict(),
            'summary': {
                'num_runs': results.num_runs,
                'mean_reward': results.mean_reward,
                'std_reward': results.std_reward,
                'confidence_interval': results.confidence_interval,
                'success_rate': results.success_rate,
                'total_duration': results.total_duration,
                'convergence_achieved': results.convergence_achieved,
            },
            'individual_runs': [
                {
                    'run_id': r.run_id,
                    'seed': r.seed,
                    'reward': r.reward,
                    'steps': r.steps,
                    'success': r.success,
                    'duration': r.duration,
                }
                for r in results.all_results
            ]
        }
        
        with open(output_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        logger.info(f"Results saved to {output_file}")
    
    return 0


def cmd_config(args: argparse.Namespace) -> int:
    """
    Manage configuration files.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Exit code
    """
    if args.create:
        config = create_default_config()
        config.to_json(args.create)
        print(f"Created default configuration: {args.create}")
        return 0
    
    if args.validate:
        try:
            config = FrameworkConfig.from_json(args.validate)
            config.validate()
            print(f"Configuration is valid: {args.validate}")
            return 0
        except Exception as e:
            print(f"Configuration validation failed: {e}")
            return 1
    
    if args.show:
        config = FrameworkConfig.from_json(args.show)
        print(json.dumps(config.to_dict(), indent=2))
        return 0
    
    return 0


def cmd_plugins(args: argparse.Namespace) -> int:
    """
    Manage and list plugins.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Exit code
    """
    registry = get_registry()
    
    if args.discover:
        count = registry.discover_plugins(args.discover)
        print(f"Discovered and registered {count} plugins from {args.discover}")
    
    if args.list:
        print("\nREGISTERED PLUGINS")
        print("="*60)
        
        print("\nSimulators:")
        for name in registry.list_simulators():
            print(f"  - {name}")
        
        print("\nStrategies:")
        for name in registry.list_strategies():
            print(f"  - {name}")
        
        print("\nEstimators:")
        for name in registry.list_estimators():
            print(f"  - {name}")
        
        print()
    
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    """
    Analyze saved results.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Exit code
    """
    results_file = Path(args.results)
    
    if not results_file.exists():
        print(f"Error: Results file not found: {results_file}")
        return 1
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    summary = data['summary']
    
    print("\n" + "="*60)
    print("RESULTS ANALYSIS")
    print("="*60)
    print(f"Configuration: {results_file}")
    print(f"\nTotal runs:        {summary['num_runs']}")
    print(f"Mean reward:       {summary['mean_reward']:.4f}")
    print(f"Std deviation:     {summary['std_reward']:.4f}")
    print(f"95% CI:            [{summary['confidence_interval'][0]:.4f}, {summary['confidence_interval'][1]:.4f}]")
    print(f"Success rate:      {summary['success_rate']:.2%}")
    print(f"Execution time:    {summary['total_duration']:.2f}s")
    print("="*60)
    
    if args.detailed:
        print("\nINDIVIDUAL RUNS (first 10):")
        for run in data['individual_runs'][:10]:
            print(f"  Run {run['run_id']}: reward={run['reward']:.2f}, steps={run['steps']}, success={run['success']}")
    
    return 0


def create_parser() -> argparse.ArgumentParser:
    """
    Create the argument parser for the CLI.
    
    Returns:
        Configured ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog='apf',
        description='Applied Probability Framework - Professional Monte Carlo Simulation System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run simulation with default config
  apf run
  
  # Run with custom configuration
  apf run --config my_config.json
  
  # Run 10000 simulations in parallel
  apf run --num-simulations 10000 --parallel --jobs 8
  
  # Create a default configuration file
  apf config --create default_config.json
  
  # List available plugins
  apf plugins --list
  
  # Analyze results
  apf analyze results/results_mines_basic.json
        """
    )
    
    parser.add_argument('--version', action='version', version='%(prog)s 1.0.0')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Run simulations')
    run_parser.add_argument('-c', '--config', type=str,
                           help='Configuration file (JSON)')
    run_parser.add_argument('-n', '--num-simulations', type=int,
                           help='Number of simulation runs')
    run_parser.add_argument('-s', '--seed', type=int,
                           help='Random seed for reproducibility')
    run_parser.add_argument('--simulator', type=str,
                           help='Simulator to use')
    run_parser.add_argument('--strategy', type=str,
                           help='Strategy to use')
    run_parser.add_argument('-p', '--parallel', action='store_true',
                           help='Run simulations in parallel')
    run_parser.add_argument('-j', '--jobs', type=int,
                           help='Number of parallel jobs (-1 for all cores)')
    run_parser.add_argument('-o', '--output-dir', type=str,
                           help='Output directory for results')
    run_parser.add_argument('-q', '--quiet', action='store_true',
                           help='Suppress progress output')
    run_parser.set_defaults(func=cmd_run)
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Manage configurations')
    config_group = config_parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument('--create', type=str,
                             help='Create default configuration file')
    config_group.add_argument('--validate', type=str,
                             help='Validate configuration file')
    config_group.add_argument('--show', type=str,
                             help='Display configuration file')
    config_parser.set_defaults(func=cmd_config)
    
    # Plugins command
    plugins_parser = subparsers.add_parser('plugins', help='Manage plugins')
    plugins_parser.add_argument('--list', action='store_true',
                               help='List registered plugins')
    plugins_parser.add_argument('--discover', type=str,
                               help='Discover plugins in directory')
    plugins_parser.set_defaults(func=cmd_plugins)
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze results')
    analyze_parser.add_argument('results', type=str,
                               help='Results file to analyze')
    analyze_parser.add_argument('-d', '--detailed', action='store_true',
                               help='Show detailed run information')
    analyze_parser.set_defaults(func=cmd_analyze)
    
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """
    Main entry point for the CLI.
    
    Args:
        argv: Command-line arguments (uses sys.argv if None)
        
    Returns:
        Exit code
    """
    parser = create_parser()
    args = parser.parse_args(argv)
    
    if not hasattr(args, 'func'):
        parser.print_help()
        return 1
    
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

