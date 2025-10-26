"""
AI Evolution Demonstration

This script demonstrates the matrix-based evolution capabilities of the
Applied Probability Framework. It shows how strategies learn and evolve
through:
1. Q-Learning for action-value estimation
2. Experience Replay for efficient learning
3. Genetic Algorithm-based parameter evolution

Run this to see your AIs actually evolve!
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'python'))

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
import json

from character_strategies import HybridStrategy, RintaroOkabeStrategy
from simulators.mines_simulator import MinesSimulator


def run_evolution_experiment(
    strategy,
    num_episodes: int = 100,
    board_size: int = 5,
    mine_count: int = 3
) -> Dict:
    """
    Run evolution experiment with a strategy.
    
    Args:
        strategy: Strategy instance to evolve
        num_episodes: Number of games to play
        board_size: Size of the mines board
        mine_count: Number of mines
        
    Returns:
        Dictionary with evolution statistics
    """
    print(f"\n{'='*60}")
    print(f"Running Evolution Experiment: {strategy.name}")
    print(f"{'='*60}\n")
    
    simulator = MinesSimulator(
        seed=42,
        config={
            'board_size': board_size,
            'mine_count': mine_count,
            'multiplier_base': 1.1
        }
    )
    
    # Track metrics
    episode_rewards = []
    episode_lengths = []
    win_rates = []
    epsilons = []
    q_values = []
    
    # Run episodes
    for episode in range(num_episodes):
        state = simulator.reset()
        episode_reward = 0.0
        episode_length = 0
        done = False
        
        while not done:
            # Get valid actions
            valid_actions = simulator.get_valid_actions()
            
            if not valid_actions:
                break
            
            # Select action
            action = strategy.select_action(state, valid_actions)
            
            # Take step
            next_state, reward, done, info = simulator.step(action)
            
            # Update strategy
            strategy.update(state, action, reward, next_state, done)
            
            # Track metrics
            episode_reward += reward
            episode_length += 1
            
            state = next_state
        
        # Record episode metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Calculate rolling win rate
        recent_wins = sum(1 for r in episode_rewards[-10:] if r > 0)
        win_rate = recent_wins / min(len(episode_rewards), 10)
        win_rates.append(win_rate)
        
        # Track Q-learning metrics if available
        if hasattr(strategy, 'q_matrix') and strategy.q_matrix:
            epsilons.append(strategy.q_matrix.epsilon)
            # Track average Q-value
            if len(strategy.q_matrix.q_table_a) > 0:
                all_q_values = []
                for state_dict in strategy.q_matrix.q_table_a.values():
                    all_q_values.extend(state_dict.values())
                if all_q_values:
                    q_values.append(np.mean(all_q_values))
                else:
                    q_values.append(0.0)
            else:
                q_values.append(0.0)
        
        # Progress updates
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Win Rate: {win_rate:.2%} | "
                  f"Epsilon: {epsilons[-1]:.4f}" if epsilons else "")
    
    # Get final statistics
    stats = strategy.get_learning_statistics() if hasattr(strategy, 'get_learning_statistics') \
            else strategy.get_statistics()
    
    print(f"\n{'='*60}")
    print("Final Statistics:")
    print(f"{'='*60}")
    print(f"Total Episodes: {num_episodes}")
    print(f"Average Reward: {np.mean(episode_rewards):.2f}")
    print(f"Final Win Rate: {win_rates[-1]:.2%}")
    
    if 'q_learning' in stats:
        print(f"\nQ-Learning Stats:")
        print(f"  Updates: {stats['q_learning']['updates']}")
        print(f"  Q-Table Size: {stats['q_learning']['q_table_size']}")
        print(f"  Final Epsilon: {stats['q_learning']['epsilon']:.4f}")
    
    if 'experience_replay' in stats:
        print(f"\nExperience Replay Stats:")
        print(f"  Buffer Size: {stats['experience_replay']['buffer_size']}")
        print(f"  Total Experiences: {stats['experience_replay']['total_experiences']}")
    
    if 'evolution' in stats:
        print(f"\nEvolution Stats:")
        print(f"  Generation: {stats['evolution']['generation']}")
        print(f"  Best Fitness: {stats['evolution']['best_fitness']:.4f}")
        print(f"  Avg Fitness: {stats['evolution']['avg_fitness']:.4f}")
    
    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'win_rates': win_rates,
        'epsilons': epsilons,
        'q_values': q_values,
        'final_stats': stats
    }


def plot_evolution_results(results: Dict, strategy_name: str) -> None:
    """
    Plot evolution results.
    
    Args:
        results: Results dictionary from run_evolution_experiment
        strategy_name: Name of strategy for plot title
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'AI Evolution: {strategy_name}', fontsize=16, fontweight='bold')
    
    # Plot 1: Episode Rewards
    ax1 = axes[0, 0]
    episodes = range(len(results['episode_rewards']))
    ax1.plot(episodes, results['episode_rewards'], alpha=0.3, label='Episode Reward')
    
    # Add moving average
    window = 10
    if len(results['episode_rewards']) >= window:
        moving_avg = np.convolve(results['episode_rewards'], 
                                np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(results['episode_rewards'])), 
                moving_avg, linewidth=2, label=f'{window}-Episode Moving Avg')
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Reward Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Win Rate
    ax2 = axes[0, 1]
    ax2.plot(episodes, results['win_rates'], linewidth=2, color='green')
    ax2.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='50% Baseline')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Win Rate')
    ax2.set_title('Rolling Win Rate (10 episodes)')
    ax2.set_ylim([0, 1])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Exploration Rate (Epsilon)
    ax3 = axes[1, 0]
    if results['epsilons']:
        ax3.plot(episodes, results['epsilons'], linewidth=2, color='orange')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Epsilon')
        ax3.set_title('Exploration Rate Decay')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'Q-Learning Not Enabled', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Exploration Rate Decay')
    
    # Plot 4: Q-Value Evolution
    ax4 = axes[1, 1]
    if results['q_values']:
        ax4.plot(episodes, results['q_values'], linewidth=2, color='purple')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Average Q-Value')
        ax4.set_title('Q-Value Evolution')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Q-Learning Not Enabled', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Q-Value Evolution')
    
    plt.tight_layout()
    
    # Save plot
    save_path = Path(__file__).parent.parent / 'results' / f'{strategy_name}_evolution.png'
    save_path.parent.mkdir(exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {save_path}")
    
    plt.show()


def compare_strategies() -> None:
    """Compare multiple strategies with evolution."""
    print("\n" + "="*60)
    print("COMPARATIVE EVOLUTION EXPERIMENT")
    print("="*60)
    
    # Test configurations
    configs = {
        'Hybrid (High Q-Learning)': {
            'strategy': HybridStrategy,
            'config': {'q_learning_weight': 0.7, 'learning_rate': 0.15}
        },
        'Hybrid (Balanced)': {
            'strategy': HybridStrategy,
            'config': {'q_learning_weight': 0.5, 'learning_rate': 0.1}
        },
        'Okabe (High Q-Learning)': {
            'strategy': RintaroOkabeStrategy,
            'config': {'q_learning_weight': 0.6, 'learning_rate': 0.15}
        },
    }
    
    all_results = {}
    
    for name, config in configs.items():
        strategy = config['strategy'](config=config['config'])
        results = run_evolution_experiment(strategy, num_episodes=50)
        all_results[name] = results
    
    # Comparative plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle('Strategy Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: Rewards comparison
    ax1 = axes[0]
    for name, results in all_results.items():
        window = 10
        if len(results['episode_rewards']) >= window:
            moving_avg = np.convolve(results['episode_rewards'], 
                                    np.ones(window)/window, mode='valid')
            ax1.plot(range(window-1, len(results['episode_rewards'])), 
                    moving_avg, linewidth=2, label=name)
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Average Reward')
    ax1.set_title('Reward Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Win rates comparison
    ax2 = axes[1]
    for name, results in all_results.items():
        ax2.plot(results['win_rates'], linewidth=2, label=name)
    
    ax2.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='50% Baseline')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Win Rate')
    ax2.set_title('Win Rate Comparison')
    ax2.set_ylim([0, 1])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = Path(__file__).parent.parent / 'results' / 'strategy_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nComparison plot saved to: {save_path}")
    
    plt.show()


def save_evolved_strategy(strategy, name: str) -> None:
    """Save evolved strategy to disk."""
    save_dir = Path(__file__).parent.parent / 'results' / 'evolved_strategies' / name
    
    if hasattr(strategy, 'save_learning_state'):
        strategy.save_learning_state(str(save_dir))
        print(f"\nEvolved strategy saved to: {save_dir}")
    else:
        print("\nStrategy does not support saving learning state")


def main():
    """Main demonstration."""
    print("\n" + "="*60)
    print("AI EVOLUTION DEMONSTRATION")
    print("Applied Probability Framework")
    print("="*60)
    
    # Experiment 1: Hybrid Strategy Evolution
    print("\n\n### EXPERIMENT 1: Hybrid Strategy Evolution ###")
    hybrid_strategy = HybridStrategy(
        config={
            'q_learning_weight': 0.5,  # 50% Q-learning, 50% heuristic
            'learning_rate': 0.1,
            'epsilon': 0.2,  # 20% exploration initially
            'epsilon_decay': 0.995,
            'replay_buffer_size': 5000,
            'evolution_frequency': 10,  # Evolve every 10 episodes
        }
    )
    
    hybrid_results = run_evolution_experiment(hybrid_strategy, num_episodes=100)
    plot_evolution_results(hybrid_results, 'Hybrid_Strategy')
    save_evolved_strategy(hybrid_strategy, 'hybrid')
    
    # Experiment 2: Rintaro Okabe Evolution
    print("\n\n### EXPERIMENT 2: Rintaro Okabe Evolution ###")
    okabe_strategy = RintaroOkabeStrategy(
        config={
            'q_learning_weight': 0.4,  # Reading Steiner bias toward heuristic
            'learning_rate': 0.12,
            'epsilon': 0.15,
            'epsilon_decay': 0.995,
            'replay_buffer_size': 5000,
            'evolution_frequency': 10,
        }
    )
    
    okabe_results = run_evolution_experiment(okabe_strategy, num_episodes=100)
    plot_evolution_results(okabe_results, 'Rintaro_Okabe')
    save_evolved_strategy(okabe_strategy, 'okabe')
    
    # Experiment 3: Strategy Comparison
    print("\n\n### EXPERIMENT 3: Strategy Comparison ###")
    compare_strategies()
    
    print("\n" + "="*60)
    print("EVOLUTION DEMONSTRATION COMPLETE!")
    print("="*60)
    print("\nKey Takeaways:")
    print("1. Q-Learning matrices allow strategies to learn optimal actions")
    print("2. Experience replay improves sample efficiency")
    print("3. Parameter evolution adapts strategy behavior over time")
    print("4. Exploration-exploitation balance (epsilon) decays automatically")
    print("5. Strategies can be saved and loaded to preserve learning")
    print("\nYour AIs are now evolving! 🧠🚀")


if __name__ == '__main__':
    # Check for required dependencies
    try:
        import matplotlib
        main()
    except ImportError:
        print("ERROR: matplotlib is required for visualization")
        print("Install with: pip install matplotlib")
        sys.exit(1)

