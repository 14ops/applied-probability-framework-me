"""
AI Agent Registry and Performance Tracking System
Tracks all AI agents, their strategies, and performance improvements over time.
"""

import json
import time
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import os

@dataclass
class PerformanceMetrics:
    """Performance metrics for an AI agent"""
    win_rate: float = 0.0
    avg_payout: float = 0.0
    total_games: int = 0
    total_profit: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    learning_progress: float = 0.0
    last_updated: str = ""
    
    def to_dict(self):
        return asdict(self)

@dataclass
class AIAgent:
    """Represents an AI agent with its configuration and performance"""
    id: str
    name: str
    strategy_type: str
    description: str
    color: str
    created_at: str
    is_active: bool = True
    performance: PerformanceMetrics = None
    improvements: List[Dict] = None
    
    def __post_init__(self):
        if self.performance is None:
            self.performance = PerformanceMetrics()
        if self.improvements is None:
            self.improvements = []
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'strategy_type': self.strategy_type,
            'description': self.description,
            'color': self.color,
            'created_at': self.created_at,
            'is_active': self.is_active,
            'performance': self.performance.to_dict(),
            'improvements': self.improvements
        }

class AIAgentRegistry:
    """Registry for managing and tracking AI agents and their improvements"""
    
    def __init__(self, data_file: str = "./logs/ai_agents.json"):
        self.data_file = data_file
        self.agents: Dict[str, AIAgent] = {}
        self.performance_history: Dict[str, List[PerformanceMetrics]] = {}
        self.load_agents()
        self.initialize_default_agents()
    
    def load_agents(self):
        """Load agents from persistent storage"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    for agent_data in data.get('agents', []):
                        agent = AIAgent(**agent_data)
                        self.agents[agent.id] = agent
                    self.performance_history = data.get('performance_history', {})
            except Exception as e:
                print(f"Error loading agents: {e}")
    
    def save_agents(self):
        """Save agents to persistent storage"""
        os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
        data = {
            'agents': [agent.to_dict() for agent in self.agents.values()],
            'performance_history': self.performance_history,
            'last_updated': datetime.now().isoformat()
        }
        with open(self.data_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def initialize_default_agents(self):
        """Initialize default AI agents if none exist"""
        if not self.agents:
            default_agents = [
                {
                    'id': 'takeshi_ai',
                    'name': 'Takeshi AI',
                    'strategy_type': 'aggressive',
                    'description': 'High-risk, high-reward strategy with dynamic risk adjustment',
                    'color': '#ef4444',
                    'created_at': datetime.now().isoformat()
                },
                {
                    'id': 'lelouch_ai',
                    'name': 'Lelouch AI',
                    'strategy_type': 'strategic',
                    'description': 'Calculated mastermind with advanced probability analysis',
                    'color': '#8b5cf6',
                    'created_at': datetime.now().isoformat()
                },
                {
                    'id': 'kazuya_ai',
                    'name': 'Kazuya AI',
                    'strategy_type': 'conservative',
                    'description': 'Risk-averse strategy focused on capital preservation',
                    'color': '#10b981',
                    'created_at': datetime.now().isoformat()
                },
                {
                    'id': 'senku_ai',
                    'name': 'Senku AI',
                    'strategy_type': 'analytical',
                    'description': 'Scientific approach with data-driven decision making',
                    'color': '#3b82f6',
                    'created_at': datetime.now().isoformat()
                },
                {
                    'id': 'drl_agent',
                    'name': 'Deep RL Agent',
                    'strategy_type': 'reinforcement_learning',
                    'description': 'Deep reinforcement learning agent with Q-learning',
                    'color': '#f59e0b',
                    'created_at': datetime.now().isoformat()
                },
                {
                    'id': 'bayesian_agent',
                    'name': 'Bayesian Agent',
                    'strategy_type': 'bayesian',
                    'description': 'Bayesian inference-based decision making',
                    'color': '#ec4899',
                    'created_at': datetime.now().isoformat()
                },
                {
                    'id': 'adversarial_agent',
                    'name': 'Adversarial Agent',
                    'strategy_type': 'adversarial',
                    'description': 'Adversarial training for robust decision making',
                    'color': '#6b7280',
                    'created_at': datetime.now().isoformat()
                },
                {
                    'id': 'ensemble_agent',
                    'name': 'Ensemble Agent',
                    'strategy_type': 'ensemble',
                    'description': 'Confidence-weighted ensemble of multiple strategies',
                    'color': '#14b8a6',
                    'created_at': datetime.now().isoformat()
                }
            ]
            
            for agent_data in default_agents:
                agent = AIAgent(**agent_data)
                self.agents[agent.id] = agent
                self.performance_history[agent.id] = []
            
            self.save_agents()
    
    def add_agent(self, agent: AIAgent):
        """Add a new AI agent to the registry"""
        self.agents[agent.id] = agent
        self.performance_history[agent.id] = []
        self.save_agents()
    
    def update_agent_performance(self, agent_id: str, metrics: PerformanceMetrics):
        """Update performance metrics for an agent"""
        if agent_id in self.agents:
            self.agents[agent_id].performance = metrics
            self.performance_history[agent_id].append(metrics)
            
            # Keep only last 100 performance records
            if len(self.performance_history[agent_id]) > 100:
                self.performance_history[agent_id] = self.performance_history[agent_id][-100:]
            
            self.save_agents()
    
    def record_improvement(self, agent_id: str, improvement_type: str, 
                          improvement_value: float, description: str):
        """Record an improvement made by an agent"""
        if agent_id in self.agents:
            improvement = {
                'timestamp': datetime.now().isoformat(),
                'type': improvement_type,
                'value': improvement_value,
                'description': description
            }
            self.agents[agent_id].improvements.append(improvement)
            self.save_agents()
    
    def get_agent(self, agent_id: str) -> Optional[AIAgent]:
        """Get an agent by ID"""
        return self.agents.get(agent_id)
    
    def get_all_agents(self) -> List[AIAgent]:
        """Get all agents"""
        return list(self.agents.values())
    
    def get_active_agents(self) -> List[AIAgent]:
        """Get all active agents"""
        return [agent for agent in self.agents.values() if agent.is_active]
    
    def get_performance_history(self, agent_id: str, days: int = 30) -> List[PerformanceMetrics]:
        """Get performance history for an agent over specified days"""
        if agent_id not in self.performance_history:
            return []
        
        cutoff_date = datetime.now() - timedelta(days=days)
        history = []
        
        for metrics in self.performance_history[agent_id]:
            if datetime.fromisoformat(metrics.last_updated) >= cutoff_date:
                history.append(metrics)
        
        return history
    
    def get_agent_rankings(self) -> List[Dict]:
        """Get agents ranked by performance"""
        rankings = []
        for agent in self.agents.values():
            if agent.performance.total_games > 0:
                rankings.append({
                    'agent': agent,
                    'score': self._calculate_performance_score(agent.performance)
                })
        
        return sorted(rankings, key=lambda x: x['score'], reverse=True)
    
    def _calculate_performance_score(self, metrics: PerformanceMetrics) -> float:
        """Calculate a composite performance score"""
        # Weighted combination of different metrics
        win_rate_score = metrics.win_rate * 0.3
        profit_score = min(metrics.total_profit / 1000, 1.0) * 0.3  # Normalize profit
        sharpe_score = max(metrics.sharpe_ratio, 0) * 0.2
        learning_score = metrics.learning_progress * 0.2
        
        return win_rate_score + profit_score + sharpe_score + learning_score
    
    def simulate_agent_improvements(self):
        """Simulate improvements for demonstration purposes"""
        for agent_id, agent in self.agents.items():
            # Simulate random improvements
            if np.random.random() < 0.3:  # 30% chance of improvement
                improvement_types = [
                    'win_rate_improvement',
                    'risk_management_enhancement',
                    'learning_algorithm_update',
                    'strategy_optimization',
                    'performance_boost'
                ]
                
                improvement_type = np.random.choice(improvement_types)
                improvement_value = np.random.uniform(0.01, 0.05)
                
                descriptions = {
                    'win_rate_improvement': f'Win rate improved by {improvement_value:.1%}',
                    'risk_management_enhancement': f'Risk management enhanced, reducing drawdown by {improvement_value:.1%}',
                    'learning_algorithm_update': f'Learning algorithm updated, improving adaptation speed by {improvement_value:.1%}',
                    'strategy_optimization': f'Strategy parameters optimized, increasing efficiency by {improvement_value:.1%}',
                    'performance_boost': f'Overall performance boosted by {improvement_value:.1%}'
                }
                
                self.record_improvement(
                    agent_id,
                    improvement_type,
                    improvement_value,
                    descriptions[improvement_type]
                )
                
                # Update performance metrics
                current_metrics = agent.performance
                new_metrics = PerformanceMetrics(
                    win_rate=min(current_metrics.win_rate + improvement_value, 1.0),
                    avg_payout=current_metrics.avg_payout * (1 + improvement_value),
                    total_games=current_metrics.total_games + np.random.randint(1, 10),
                    total_profit=current_metrics.total_profit + np.random.uniform(10, 100),
                    max_drawdown=current_metrics.max_drawdown * (1 - improvement_value),
                    sharpe_ratio=current_metrics.sharpe_ratio + improvement_value,
                    learning_progress=min(current_metrics.learning_progress + improvement_value, 1.0),
                    last_updated=datetime.now().isoformat()
                )
                
                self.update_agent_performance(agent_id, new_metrics)

# Global registry instance
ai_registry = AIAgentRegistry()

if __name__ == "__main__":
    # Demo the registry
    print("AI Agent Registry Demo")
    print("=" * 50)
    
    # Simulate some improvements
    ai_registry.simulate_agent_improvements()
    
    # Show all agents
    print("\nAll AI Agents:")
    for agent in ai_registry.get_all_agents():
        print(f"- {agent.name} ({agent.strategy_type}): {agent.performance.win_rate:.1%} win rate")
    
    # Show rankings
    print("\nAgent Rankings:")
    rankings = ai_registry.get_agent_rankings()
    for i, ranking in enumerate(rankings, 1):
        print(f"{i}. {ranking['agent'].name}: {ranking['score']:.3f}")