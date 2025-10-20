"""
AI Agent Performance Tracker
Tracks and manages AI agent performance data for visualization
"""

import json
import os
import time
import random
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import sqlite3
from dataclasses import dataclass, asdict

@dataclass
class AgentPerformance:
    agent_id: str
    win_rate: float
    avg_payout: float
    total_games: int
    improvement: float
    timestamp: str
    metrics: Dict[str, Any]

@dataclass
class AgentConfig:
    agent_id: str
    name: str
    agent_type: str
    status: str
    created_at: str
    last_updated: str
    configuration: Dict[str, Any]

class AIAgentTracker:
    def __init__(self, db_path: str = "ai_agents.db"):
        self.db_path = db_path
        self.init_database()
        self.initialize_default_agents()
    
    def init_database(self):
        """Initialize SQLite database for storing agent data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create agents table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS agents (
                agent_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                agent_type TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                last_updated TEXT NOT NULL,
                configuration TEXT NOT NULL
            )
        ''')
        
        # Create performance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT NOT NULL,
                win_rate REAL NOT NULL,
                avg_payout REAL NOT NULL,
                total_games INTEGER NOT NULL,
                improvement REAL NOT NULL,
                timestamp TEXT NOT NULL,
                metrics TEXT NOT NULL,
                FOREIGN KEY (agent_id) REFERENCES agents (agent_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def initialize_default_agents(self):
        """Initialize default AI agents if they don't exist"""
        if self.get_all_agents():
            return  # Agents already exist
        
        default_agents = [
            {
                "agent_id": "drl_agent",
                "name": "Deep Reinforcement Learning Agent",
                "agent_type": "DRL",
                "status": "active",
                "configuration": {
                    "learning_rate": 0.001,
                    "epsilon": 0.15,
                    "q_table_size": 10000,
                    "episodes": 5000,
                    "memory_size": 100000,
                    "batch_size": 32
                }
            },
            {
                "agent_id": "monte_carlo_agent",
                "name": "Monte Carlo Tree Search Agent",
                "agent_type": "MCTS",
                "status": "active",
                "configuration": {
                    "simulations": 1000,
                    "exploration_constant": 1.4,
                    "max_depth": 20,
                    "iterations": 2500,
                    "time_limit": 1000,
                    "selection_policy": "UCB1"
                }
            },
            {
                "agent_id": "bayesian_agent",
                "name": "Bayesian Probability Agent",
                "agent_type": "Bayesian",
                "status": "active",
                "configuration": {
                    "prior_strength": 0.1,
                    "confidence_level": 0.95,
                    "sample_size": 500,
                    "iterations": 3000,
                    "convergence_threshold": 0.001,
                    "update_frequency": 10
                }
            },
            {
                "agent_id": "adversarial_agent",
                "name": "Adversarial Training Agent",
                "agent_type": "Adversarial",
                "status": "training",
                "configuration": {
                    "generator_learning_rate": 0.0002,
                    "discriminator_learning_rate": 0.0001,
                    "epochs": 100,
                    "batch_size": 32,
                    "discriminator_steps": 1,
                    "generator_steps": 1
                }
            },
            {
                "agent_id": "ensemble_agent",
                "name": "Ensemble Learning Agent",
                "agent_type": "Ensemble",
                "status": "active",
                "configuration": {
                    "num_models": 5,
                    "voting_method": "weighted",
                    "confidence_threshold": 0.8,
                    "retraining_frequency": 100,
                    "diversity_measure": 0.7,
                    "consensus_threshold": 0.6
                }
            }
        ]
        
        for agent_data in default_agents:
            self.add_agent(agent_data)
            # Generate some initial performance data
            self.generate_initial_performance(agent_data["agent_id"])
    
    def add_agent(self, agent_data: Dict[str, Any]) -> bool:
        """Add a new agent to the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            now = datetime.now().isoformat()
            cursor.execute('''
                INSERT OR REPLACE INTO agents 
                (agent_id, name, agent_type, status, created_at, last_updated, configuration)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                agent_data["agent_id"],
                agent_data["name"],
                agent_data["agent_type"],
                agent_data["status"],
                now,
                now,
                json.dumps(agent_data["configuration"])
            ))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error adding agent: {e}")
            return False
    
    def generate_initial_performance(self, agent_id: str, days: int = 30):
        """Generate initial performance data for an agent"""
        base_performance = {
            "drl_agent": {"win_rate": 0.5, "avg_payout": 1.2},
            "monte_carlo_agent": {"win_rate": 0.48, "avg_payout": 1.18},
            "bayesian_agent": {"win_rate": 0.52, "avg_payout": 1.22},
            "adversarial_agent": {"win_rate": 0.45, "avg_payout": 1.15},
            "ensemble_agent": {"win_rate": 0.55, "avg_payout": 1.25}
        }
        
        base = base_performance.get(agent_id, {"win_rate": 0.5, "avg_payout": 1.2})
        
        for i in range(days):
            date = datetime.now() - timedelta(days=days-i-1)
            
            # Simulate gradual improvement with some randomness
            improvement_factor = i * 0.01 + random.uniform(-0.02, 0.02)
            win_rate = max(0.1, min(0.95, base["win_rate"] + improvement_factor))
            avg_payout = max(1.0, min(2.0, base["avg_payout"] + improvement_factor * 0.5))
            
            total_games = random.randint(20, 100)
            improvement = random.uniform(0, 0.05)
            
            metrics = {
                "learning_rate": random.uniform(0.0001, 0.01),
                "confidence": random.uniform(0.7, 0.95),
                "exploration_rate": random.uniform(0.05, 0.3)
            }
            
            self.record_performance(
                agent_id=agent_id,
                win_rate=win_rate,
                avg_payout=avg_payout,
                total_games=total_games,
                improvement=improvement,
                metrics=metrics,
                timestamp=date.isoformat()
            )
    
    def record_performance(self, agent_id: str, win_rate: float, avg_payout: float, 
                          total_games: int, improvement: float, metrics: Dict[str, Any], 
                          timestamp: Optional[str] = None) -> bool:
        """Record performance data for an agent"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if timestamp is None:
                timestamp = datetime.now().isoformat()
            
            cursor.execute('''
                INSERT INTO performance 
                (agent_id, win_rate, avg_payout, total_games, improvement, timestamp, metrics)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                agent_id, win_rate, avg_payout, total_games, improvement, 
                timestamp, json.dumps(metrics)
            ))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error recording performance: {e}")
            return False
    
    def get_all_agents(self) -> List[Dict[str, Any]]:
        """Get all agents from the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM agents ORDER BY created_at')
            rows = cursor.fetchall()
            
            agents = []
            for row in rows:
                agents.append({
                    "agent_id": row[0],
                    "name": row[1],
                    "agent_type": row[2],
                    "status": row[3],
                    "created_at": row[4],
                    "last_updated": row[5],
                    "configuration": json.loads(row[6])
                })
            
            conn.close()
            return agents
        except Exception as e:
            print(f"Error getting agents: {e}")
            return []
    
    def get_agent_performance(self, agent_id: str, days: int = 7) -> List[Dict[str, Any]]:
        """Get performance data for a specific agent"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
            cursor.execute('''
                SELECT * FROM performance 
                WHERE agent_id = ? AND timestamp >= ?
                ORDER BY timestamp
            ''', (agent_id, cutoff_date))
            
            rows = cursor.fetchall()
            
            performance_data = []
            for row in rows:
                performance_data.append({
                    "id": row[0],
                    "agent_id": row[1],
                    "win_rate": row[2],
                    "avg_payout": row[3],
                    "total_games": row[4],
                    "improvement": row[5],
                    "timestamp": row[6],
                    "metrics": json.loads(row[7])
                })
            
            conn.close()
            return performance_data
        except Exception as e:
            print(f"Error getting agent performance: {e}")
            return []
    
    def get_latest_performance(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get the latest performance data for an agent"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM performance 
                WHERE agent_id = ?
                ORDER BY timestamp DESC
                LIMIT 1
            ''', (agent_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return {
                    "id": row[0],
                    "agent_id": row[1],
                    "win_rate": row[2],
                    "avg_payout": row[3],
                    "total_games": row[4],
                    "improvement": row[5],
                    "timestamp": row[6],
                    "metrics": json.loads(row[7])
                }
            return None
        except Exception as e:
            print(f"Error getting latest performance: {e}")
            return None
    
    def get_agent_summary(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get a summary of agent performance"""
        try:
            # Get agent info
            agents = self.get_all_agents()
            agent = next((a for a in agents if a["agent_id"] == agent_id), None)
            if not agent:
                return None
            
            # Get latest performance
            latest_perf = self.get_latest_performance(agent_id)
            if not latest_perf:
                return None
            
            # Get performance history for improvement calculation
            history = self.get_agent_performance(agent_id, days=7)
            if len(history) < 2:
                improvement = 0
            else:
                recent_avg = sum(p["win_rate"] for p in history[-3:]) / min(3, len(history))
                older_avg = sum(p["win_rate"] for p in history[:-3]) / max(1, len(history) - 3)
                improvement = recent_avg - older_avg
            
            return {
                "agent": agent,
                "performance": {
                    "win_rate": latest_perf["win_rate"],
                    "avg_payout": latest_perf["avg_payout"],
                    "total_games": latest_perf["total_games"],
                    "improvement": improvement,
                    "last_update": latest_perf["timestamp"]
                },
                "metrics": latest_perf["metrics"]
            }
        except Exception as e:
            print(f"Error getting agent summary: {e}")
            return None
    
    def get_all_agent_summaries(self) -> List[Dict[str, Any]]:
        """Get summaries for all agents"""
        agents = self.get_all_agents()
        summaries = []
        
        for agent in agents:
            summary = self.get_agent_summary(agent["agent_id"])
            if summary:
                summaries.append(summary)
        
        return summaries
    
    def update_agent_status(self, agent_id: str, status: str) -> bool:
        """Update agent status"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            now = datetime.now().isoformat()
            cursor.execute('''
                UPDATE agents 
                SET status = ?, last_updated = ?
                WHERE agent_id = ?
            ''', (status, now, agent_id))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error updating agent status: {e}")
            return False
    
    def simulate_agent_improvement(self, agent_id: str):
        """Simulate agent improvement by adding new performance data"""
        latest = self.get_latest_performance(agent_id)
        if not latest:
            return
        
        # Simulate small improvements
        win_rate_change = random.uniform(-0.02, 0.05)
        payout_change = random.uniform(-0.05, 0.1)
        
        new_win_rate = max(0.1, min(0.95, latest["win_rate"] + win_rate_change))
        new_avg_payout = max(1.0, min(2.0, latest["avg_payout"] + payout_change))
        new_total_games = latest["total_games"] + random.randint(5, 25)
        new_improvement = (win_rate_change + payout_change) / 2
        
        metrics = {
            "learning_rate": random.uniform(0.0001, 0.01),
            "confidence": random.uniform(0.7, 0.95),
            "exploration_rate": random.uniform(0.05, 0.3),
            "convergence_rate": random.uniform(0.8, 1.0)
        }
        
        self.record_performance(
            agent_id=agent_id,
            win_rate=new_win_rate,
            avg_payout=new_avg_payout,
            total_games=new_total_games,
            improvement=new_improvement,
            metrics=metrics
        )
    
    def export_data(self, filepath: str) -> bool:
        """Export all data to JSON file"""
        try:
            data = {
                "agents": self.get_all_agents(),
                "performance_data": {},
                "exported_at": datetime.now().isoformat()
            }
            
            agents = self.get_all_agents()
            for agent in agents:
                data["performance_data"][agent["agent_id"]] = self.get_agent_performance(
                    agent["agent_id"], days=30
                )
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            return True
        except Exception as e:
            print(f"Error exporting data: {e}")
            return False

# Global tracker instance
tracker = AIAgentTracker()

def get_ai_agent_data():
    """Get AI agent data for the frontend"""
    return tracker.get_all_agent_summaries()

def simulate_agent_updates():
    """Simulate agent performance updates"""
    agents = tracker.get_all_agents()
    for agent in agents:
        if agent["status"] == "active":
            tracker.simulate_agent_improvement(agent["agent_id"])

if __name__ == "__main__":
    # Initialize and test the tracker
    print("Initializing AI Agent Tracker...")
    
    # Get all agent summaries
    summaries = tracker.get_all_agent_summaries()
    print(f"Found {len(summaries)} agents:")
    
    for summary in summaries:
        agent = summary["agent"]
        perf = summary["performance"]
        print(f"- {agent['name']}: {perf['win_rate']:.2%} win rate, {perf['avg_payout']:.2f}x payout")
    
    # Simulate some updates
    print("\nSimulating agent updates...")
    simulate_agent_updates()
    
    # Export data
    tracker.export_data("ai_agent_data.json")
    print("Data exported to ai_agent_data.json")