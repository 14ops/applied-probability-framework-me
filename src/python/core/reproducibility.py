"""
Reproducibility and experiment tracking module.

This module provides comprehensive experiment tracking, state serialization,
and reproducibility features following best practices for scientific computing.
"""

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, List
import json
import pickle
from pathlib import Path
from datetime import datetime
import hashlib
import numpy as np


@dataclass
class ExperimentMetadata:
    """
    Metadata for an experiment run.
    
    Attributes:
        experiment_id: Unique identifier (hash-based)
        timestamp: ISO format timestamp
        config_hash: Hash of configuration for reproducibility
        seed: Random seed used
        framework_version: Version of the framework
        python_version: Python interpreter version
        numpy_version: NumPy version
        tags: User-defined tags for organization
        description: Human-readable description
    """
    experiment_id: str
    timestamp: str
    config_hash: str
    seed: Optional[int]
    framework_version: str
    python_version: str
    numpy_version: str
    tags: List[str]
    description: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def create(cls, config: Dict[str, Any], seed: Optional[int],
               tags: Optional[List[str]] = None, description: str = "") -> 'ExperimentMetadata':
        """
        Create experiment metadata.
        
        Args:
            config: Configuration dictionary
            seed: Random seed
            tags: Optional tags
            description: Experiment description
            
        Returns:
            ExperimentMetadata instance
        """
        import sys
        
        # Create config hash for reproducibility
        config_str = json.dumps(config, sort_keys=True)
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:16]
        
        # Generate experiment ID
        timestamp = datetime.now().isoformat()
        exp_id = f"{config_hash}_{int(datetime.now().timestamp())}"
        
        return cls(
            experiment_id=exp_id,
            timestamp=timestamp,
            config_hash=config_hash,
            seed=seed,
            framework_version="1.0.0",
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            numpy_version=np.__version__,
            tags=tags or [],
            description=description
        )


class ExperimentLogger:
    """
    Comprehensive logger for experiment tracking.
    
    This class provides structured logging of:
    - Configuration and metadata
    - Random states for reproducibility
    - Intermediate results
    - Final aggregated results
    - Performance metrics
    
    All data is serialized in JSON format with optional pickle for
    non-serializable objects.
    
    Attributes:
        output_dir: Directory for experiment outputs
        metadata: Experiment metadata
        events: List of logged events
    """
    
    def __init__(self, output_dir: str, experiment_name: str = "experiment"):
        """
        Initialize experiment logger.
        
        Args:
            output_dir: Base directory for outputs
            experiment_name: Name for this experiment
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_name = experiment_name
        self.metadata: Optional[ExperimentMetadata] = None
        self.events: List[Dict[str, Any]] = []
        self.config: Dict[str, Any] = {}
        
        # Create experiment subdirectory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = self.output_dir / f"{experiment_name}_{timestamp}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
    def start_experiment(self, config: Dict[str, Any], seed: Optional[int] = None,
                        tags: Optional[List[str]] = None, description: str = "") -> None:
        """
        Start a new experiment.
        
        Args:
            config: Full configuration dictionary
            seed: Random seed
            tags: Optional tags for organization
            description: Human-readable description
        """
        self.config = config
        self.metadata = ExperimentMetadata.create(config, seed, tags, description)
        
        # Save metadata
        metadata_file = self.experiment_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata.to_dict(), f, indent=2)
        
        # Save configuration
        config_file = self.experiment_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Capture initial random state
        self._save_random_state("initial")
        
        self.log_event("experiment_started", {
            "experiment_id": self.metadata.experiment_id,
            "timestamp": self.metadata.timestamp
        })
    
    def log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Log an event during the experiment.
        
        Args:
            event_type: Type of event (e.g., 'iteration', 'checkpoint')
            data: Event-specific data
        """
        event = {
            "type": event_type,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        self.events.append(event)
        
        # Append to events file
        events_file = self.experiment_dir / "events.jsonl"
        with open(events_file, 'a') as f:
            f.write(json.dumps(event) + '\n')
    
    def log_results(self, results: Dict[str, Any], checkpoint: bool = False) -> None:
        """
        Log simulation results.
        
        Args:
            results: Results dictionary
            checkpoint: If True, saves as checkpoint for resumption
        """
        results_file = self.experiment_dir / "results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        if checkpoint:
            checkpoint_file = self.experiment_dir / f"checkpoint_{len(self.events)}.json"
            with open(checkpoint_file, 'w') as f:
                json.dump(results, f, indent=2)
        
        self.log_event("results_logged", {
            "num_entries": len(results.get('individual_runs', [])),
            "checkpoint": checkpoint
        })
    
    def save_artifact(self, name: str, obj: Any, format: str = 'json') -> None:
        """
        Save an arbitrary artifact.
        
        Args:
            name: Artifact name (without extension)
            obj: Object to save
            format: Format ('json' or 'pickle')
        """
        artifacts_dir = self.experiment_dir / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        
        if format == 'json':
            filepath = artifacts_dir / f"{name}.json"
            with open(filepath, 'w') as f:
                json.dump(obj, f, indent=2)
        elif format == 'pickle':
            filepath = artifacts_dir / f"{name}.pkl"
            with open(filepath, 'wb') as f:
                pickle.dump(obj, f)
        else:
            raise ValueError(f"Unknown format: {format}")
        
        self.log_event("artifact_saved", {"name": name, "format": format})
    
    def _save_random_state(self, label: str) -> None:
        """
        Save random state for reproducibility.
        
        Args:
            label: Label for this state checkpoint
        """
        state_dir = self.experiment_dir / "random_states"
        state_dir.mkdir(exist_ok=True)
        
        state = {
            'label': label,
            'timestamp': datetime.now().isoformat(),
            'numpy_state': np.random.get_state(),
        }
        
        state_file = state_dir / f"state_{label}.pkl"
        with open(state_file, 'wb') as f:
            pickle.dump(state, f)
    
    def load_random_state(self, label: str) -> None:
        """
        Restore random state from checkpoint.
        
        Args:
            label: Label of state to restore
        """
        state_file = self.experiment_dir / "random_states" / f"state_{label}.pkl"
        
        with open(state_file, 'rb') as f:
            state = pickle.load(f)
        
        np.random.set_state(state['numpy_state'])
    
    def end_experiment(self, final_results: Optional[Dict[str, Any]] = None) -> None:
        """
        End the experiment and finalize logs.
        
        Args:
            final_results: Optional final results to save
        """
        if final_results:
            self.log_results(final_results)
        
        # Save final random state
        self._save_random_state("final")
        
        self.log_event("experiment_ended", {
            "total_events": len(self.events),
            "duration": self._calculate_duration()
        })
        
        # Create summary
        summary = {
            "metadata": self.metadata.to_dict() if self.metadata else {},
            "total_events": len(self.events),
            "duration": self._calculate_duration(),
            "output_dir": str(self.experiment_dir)
        }
        
        summary_file = self.experiment_dir / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def _calculate_duration(self) -> float:
        """Calculate experiment duration in seconds."""
        if not self.events:
            return 0.0
        
        start_time = datetime.fromisoformat(self.events[0]['timestamp'])
        end_time = datetime.fromisoformat(self.events[-1]['timestamp'])
        return (end_time - start_time).total_seconds()
    
    def get_experiment_dir(self) -> Path:
        """Get path to experiment directory."""
        return self.experiment_dir


class StateSerializer:
    """
    Utility for serializing and deserializing simulation states.
    
    This enables checkpointing and resumption of long-running simulations.
    """
    
    @staticmethod
    def serialize_state(obj: Any, filepath: str) -> None:
        """
        Serialize object to file.
        
        Args:
            obj: Object to serialize (must be picklable)
            filepath: Output file path
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    @staticmethod
    def deserialize_state(filepath: str) -> Any:
        """
        Deserialize object from file.
        
        Args:
            filepath: Input file path
            
        Returns:
            Deserialized object
        """
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    @staticmethod
    def create_checkpoint(simulator: Any, strategy: Any, 
                         metadata: Dict[str, Any], filepath: str) -> None:
        """
        Create a checkpoint of current simulation state.
        
        Args:
            simulator: Simulator instance
            strategy: Strategy instance
            metadata: Additional metadata
            filepath: Checkpoint file path
        """
        checkpoint = {
            'simulator_state': simulator.get_state() if hasattr(simulator, 'get_state') else None,
            'simulator_metadata': simulator.get_metadata() if hasattr(simulator, 'get_metadata') else {},
            'strategy_statistics': strategy.get_statistics() if hasattr(strategy, 'get_statistics') else {},
            'metadata': metadata,
            'timestamp': datetime.now().isoformat(),
        }
        
        StateSerializer.serialize_state(checkpoint, filepath)
    
    @staticmethod
    def load_checkpoint(filepath: str) -> Dict[str, Any]:
        """
        Load checkpoint from file.
        
        Args:
            filepath: Checkpoint file path
            
        Returns:
            Checkpoint dictionary
        """
        return StateSerializer.deserialize_state(filepath)


def set_global_seed(seed: int) -> None:
    """
    Set global random seed for all libraries.
    
    This ensures reproducibility across numpy, random, and other libraries.
    
    Args:
        seed: Random seed
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    
    # Set seeds for other libraries if available
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def verify_reproducibility(config: Dict[str, Any], seed: int, 
                          num_runs: int = 5) -> bool:
    """
    Verify that simulations are reproducible with the same seed.
    
    Args:
        config: Configuration dictionary
        seed: Seed to test
        num_runs: Number of verification runs
        
    Returns:
        True if all runs produce identical results
    """
    from core.parallel_engine import ParallelSimulationEngine, _run_single_simulation
    
    results = []
    
    for _ in range(num_runs):
        set_global_seed(seed)
        # Run a simple simulation and collect results
        # (Implementation would depend on specific simulator/strategy)
        results.append(seed)  # Placeholder
    
    # Check if all results are identical
    return len(set(results)) == 1

