"""
AI Visualization System
Comprehensive visualization of AI agents, their performance, and improvements over time.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.animation import FuncAnimation
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any
import os
from ai_agent_registry import ai_registry, AIAgent, PerformanceMetrics

class AIVisualizationSystem:
    """Comprehensive visualization system for AI agents and improvements"""
    
    def __init__(self, output_dir: str = "./visualizations/ai_dashboard"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.setup_plotting_style()
    
    def setup_plotting_style(self):
        """Setup consistent plotting style"""
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # Set plotly template
        import plotly.io as pio
        pio.templates.default = "plotly_dark"
    
    def create_agent_overview_dashboard(self) -> str:
        """Create comprehensive agent overview dashboard"""
        agents = ai_registry.get_all_agents()
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Performance Metrics Comparison',
                'Win Rate Over Time',
                'Profit Distribution',
                'Learning Progress',
                'Improvement Timeline',
                'Agent Rankings'
            ],
            specs=[
                [{"type": "bar"}, {"type": "scatter"}],
                [{"type": "histogram"}, {"type": "bar"}],
                [{"type": "timeline"}, {"type": "bar"}]
            ]
        )
        
        # 1. Performance Metrics Comparison
        agent_names = [agent.name for agent in agents]
        win_rates = [agent.performance.win_rate for agent in agents]
        avg_payouts = [agent.performance.avg_payout for agent in agents]
        
        fig.add_trace(
            go.Bar(x=agent_names, y=win_rates, name='Win Rate', marker_color='#1f77b4'),
            row=1, col=1
        )
        
        # 2. Win Rate Over Time (simulated)
        for i, agent in enumerate(agents):
            history = ai_registry.get_performance_history(agent.id, days=30)
            if history:
                dates = [datetime.fromisoformat(m.last_updated) for m in history]
                win_rates = [m.win_rate for m in history]
                fig.add_trace(
                    go.Scatter(x=dates, y=win_rates, name=agent.name, 
                             line=dict(color=agent.color)),
                    row=1, col=2
                )
        
        # 3. Profit Distribution
        profits = [agent.performance.total_profit for agent in agents]
        fig.add_trace(
            go.Histogram(x=profits, name='Profit Distribution', marker_color='#ff7f0e'),
            row=2, col=1
        )
        
        # 4. Learning Progress
        learning_progress = [agent.performance.learning_progress for agent in agents]
        fig.add_trace(
            go.Bar(x=agent_names, y=learning_progress, name='Learning Progress', 
                   marker_color='#2ca02c'),
            row=2, col=2
        )
        
        # 5. Improvement Timeline
        improvement_data = []
        for agent in agents:
            for improvement in agent.improvements[-10:]:  # Last 10 improvements
                improvement_data.append({
                    'Agent': agent.name,
                    'Date': improvement['timestamp'],
                    'Type': improvement['type'],
                    'Value': improvement['value']
                })
        
        if improvement_data:
            df_improvements = pd.DataFrame(improvement_data)
            df_improvements['Date'] = pd.to_datetime(df_improvements['Date'])
            
            for agent_name in df_improvements['Agent'].unique():
                agent_data = df_improvements[df_improvements['Agent'] == agent_name]
                fig.add_trace(
                    go.Scatter(x=agent_data['Date'], y=agent_data['Value'], 
                             mode='markers+lines', name=f'{agent_name} Improvements',
                             marker=dict(size=8)),
                    row=3, col=1
                )
        
        # 6. Agent Rankings
        rankings = ai_registry.get_agent_rankings()
        ranking_names = [r['agent'].name for r in rankings]
        ranking_scores = [r['score'] for r in rankings]
        
        fig.add_trace(
            go.Bar(x=ranking_names, y=ranking_scores, name='Performance Score',
                   marker_color='#d62728'),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            title="AI Agent Performance Dashboard",
            height=1200,
            showlegend=True,
            template="plotly_dark"
        )
        
        # Save the dashboard
        dashboard_path = os.path.join(self.output_dir, "ai_agent_dashboard.html")
        fig.write_html(dashboard_path)
        
        return dashboard_path
    
    def create_improvement_timeline(self) -> str:
        """Create detailed improvement timeline visualization"""
        agents = ai_registry.get_all_agents()
        
        fig = go.Figure()
        
        for agent in agents:
            if agent.improvements:
                improvements_df = pd.DataFrame(agent.improvements)
                improvements_df['timestamp'] = pd.to_datetime(improvements_df['timestamp'])
                
                # Group improvements by type
                for improvement_type in improvements_df['type'].unique():
                    type_data = improvements_df[improvements_df['type'] == improvement_type]
                    
                    fig.add_trace(go.Scatter(
                        x=type_data['timestamp'],
                        y=type_data['value'],
                        mode='markers+lines',
                        name=f'{agent.name} - {improvement_type.replace("_", " ").title()}',
                        marker=dict(
                            size=8,
                            color=agent.color,
                            symbol='circle'
                        ),
                        line=dict(color=agent.color, width=2)
                    ))
        
        fig.update_layout(
            title="AI Agent Improvements Timeline",
            xaxis_title="Time",
            yaxis_title="Improvement Value",
            template="plotly_dark",
            height=600
        )
        
        timeline_path = os.path.join(self.output_dir, "improvement_timeline.html")
        fig.write_html(timeline_path)
        
        return timeline_path
    
    def create_performance_heatmap(self) -> str:
        """Create performance heatmap for all agents"""
        agents = ai_registry.get_all_agents()
        
        # Create performance matrix
        metrics = ['win_rate', 'avg_payout', 'total_profit', 'max_drawdown', 'sharpe_ratio', 'learning_progress']
        agent_names = [agent.name for agent in agents]
        
        performance_matrix = []
        for agent in agents:
            row = [
                agent.performance.win_rate,
                agent.performance.avg_payout,
                agent.performance.total_profit,
                abs(agent.performance.max_drawdown),  # Use absolute value for heatmap
                agent.performance.sharpe_ratio,
                agent.performance.learning_progress
            ]
            performance_matrix.append(row)
        
        # Normalize the matrix
        performance_matrix = np.array(performance_matrix)
        normalized_matrix = (performance_matrix - performance_matrix.min(axis=0)) / (performance_matrix.max(axis=0) - performance_matrix.min(axis=0))
        
        fig = go.Figure(data=go.Heatmap(
            z=normalized_matrix,
            x=metrics,
            y=agent_names,
            colorscale='Viridis',
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="AI Agent Performance Heatmap",
            xaxis_title="Performance Metrics",
            yaxis_title="AI Agents",
            template="plotly_dark",
            height=600
        )
        
        heatmap_path = os.path.join(self.output_dir, "performance_heatmap.html")
        fig.write_html(heatmap_path)
        
        return heatmap_path
    
    def create_learning_curves(self) -> str:
        """Create learning curves for all agents"""
        agents = ai_registry.get_all_agents()
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Win Rate Learning Curve', 'Profit Learning Curve', 
                          'Risk Management Learning', 'Overall Performance'],
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        for agent in agents:
            history = ai_registry.get_performance_history(agent.id, days=30)
            if history:
                dates = [datetime.fromisoformat(m.last_updated) for m in history]
                
                # Win rate curve
                win_rates = [m.win_rate for m in history]
                fig.add_trace(
                    go.Scatter(x=dates, y=win_rates, name=f'{agent.name} Win Rate',
                             line=dict(color=agent.color)),
                    row=1, col=1
                )
                
                # Profit curve
                profits = [m.total_profit for m in history]
                fig.add_trace(
                    go.Scatter(x=dates, y=profits, name=f'{agent.name} Profit',
                             line=dict(color=agent.color)),
                    row=1, col=2
                )
                
                # Risk management (inverse of max drawdown)
                risk_scores = [1 - abs(m.max_drawdown) for m in history]
                fig.add_trace(
                    go.Scatter(x=dates, y=risk_scores, name=f'{agent.name} Risk Score',
                             line=dict(color=agent.color)),
                    row=2, col=1
                )
                
                # Overall performance score
                performance_scores = [ai_registry._calculate_performance_score(m) for m in history]
                fig.add_trace(
                    go.Scatter(x=dates, y=performance_scores, name=f'{agent.name} Performance',
                             line=dict(color=agent.color)),
                    row=2, col=2
                )
        
        fig.update_layout(
            title="AI Agent Learning Curves",
            template="plotly_dark",
            height=800
        )
        
        curves_path = os.path.join(self.output_dir, "learning_curves.html")
        fig.write_html(curves_path)
        
        return curves_path
    
    def create_agent_comparison_radar(self) -> str:
        """Create radar chart comparing all agents"""
        agents = ai_registry.get_all_agents()
        
        fig = go.Figure()
        
        for agent in agents:
            # Normalize metrics to 0-1 scale
            win_rate = agent.performance.win_rate
            avg_payout = min(agent.performance.avg_payout / 10, 1.0)  # Normalize payout
            total_profit = min(agent.performance.total_profit / 1000, 1.0)  # Normalize profit
            risk_management = 1 - abs(agent.performance.max_drawdown)  # Higher is better
            sharpe_ratio = min(max(agent.performance.sharpe_ratio, 0), 2) / 2  # Normalize Sharpe
            learning_progress = agent.performance.learning_progress
            
            fig.add_trace(go.Scatterpolar(
                r=[win_rate, avg_payout, total_profit, risk_management, sharpe_ratio, learning_progress],
                theta=['Win Rate', 'Avg Payout', 'Total Profit', 'Risk Management', 'Sharpe Ratio', 'Learning Progress'],
                fill='toself',
                name=agent.name,
                line_color=agent.color
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="AI Agent Performance Comparison (Radar Chart)",
            template="plotly_dark",
            height=600
        )
        
        radar_path = os.path.join(self.output_dir, "agent_comparison_radar.html")
        fig.write_html(radar_path)
        
        return radar_path
    
    def create_realtime_dashboard(self) -> str:
        """Create real-time dashboard for live monitoring"""
        agents = ai_registry.get_all_agents()
        
        # Create a comprehensive real-time dashboard
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                'Live Performance', 'Recent Improvements', 'Agent Status',
                'Performance Trends', 'Risk Analysis', 'Learning Progress'
            ],
            specs=[
                [{"type": "bar"}, {"type": "scatter"}, {"type": "indicator"}],
                [{"type": "scatter"}, {"type": "bar"}, {"type": "bar"}]
            ]
        )
        
        # Live Performance
        agent_names = [agent.name for agent in agents]
        current_scores = [ai_registry._calculate_performance_score(agent.performance) for agent in agents]
        
        fig.add_trace(
            go.Bar(x=agent_names, y=current_scores, name='Current Performance',
                   marker_color=[agent.color for agent in agents]),
            row=1, col=1
        )
        
        # Recent Improvements (last 24 hours)
        recent_improvements = []
        for agent in agents:
            for improvement in agent.improvements[-5:]:  # Last 5 improvements
                if datetime.fromisoformat(improvement['timestamp']) > datetime.now() - timedelta(hours=24):
                    recent_improvements.append({
                        'agent': agent.name,
                        'value': improvement['value'],
                        'type': improvement['type']
                    })
        
        if recent_improvements:
            df_recent = pd.DataFrame(recent_improvements)
            for agent_name in df_recent['agent'].unique():
                agent_data = df_recent[df_recent['agent'] == agent_name]
                fig.add_trace(
                    go.Scatter(x=agent_data['type'], y=agent_data['value'],
                             mode='markers', name=f'{agent_name} Recent',
                             marker=dict(size=10)),
                    row=1, col=2
                )
        
        # Agent Status (indicator)
        active_agents = len([a for a in agents if a.is_active])
        total_agents = len(agents)
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=active_agents,
                title={'text': "Active Agents"},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={'axis': {'range': [None, total_agents]},
                       'bar': {'color': "darkblue"},
                       'steps': [{'range': [0, total_agents/2], 'color': "lightgray"},
                                {'range': [total_agents/2, total_agents], 'color': "gray"}],
                       'threshold': {'line': {'color': "red", 'width': 4},
                                   'thickness': 0.75, 'value': total_agents}}),
            row=1, col=3
        )
        
        # Performance Trends
        for agent in agents:
            history = ai_registry.get_performance_history(agent.id, days=7)
            if history:
                dates = [datetime.fromisoformat(m.last_updated) for m in history]
                scores = [ai_registry._calculate_performance_score(m) for m in history]
                fig.add_trace(
                    go.Scatter(x=dates, y=scores, name=agent.name,
                             line=dict(color=agent.color)),
                    row=2, col=1
                )
        
        # Risk Analysis
        risk_scores = [abs(agent.performance.max_drawdown) for agent in agents]
        fig.add_trace(
            go.Bar(x=agent_names, y=risk_scores, name='Risk Level',
                   marker_color='red'),
            row=2, col=2
        )
        
        # Learning Progress
        learning_scores = [agent.performance.learning_progress for agent in agents]
        fig.add_trace(
            go.Bar(x=agent_names, y=learning_scores, name='Learning Progress',
                   marker_color='green'),
            row=2, col=3
        )
        
        fig.update_layout(
            title="AI Agent Real-time Dashboard",
            template="plotly_dark",
            height=800
        )
        
        realtime_path = os.path.join(self.output_dir, "realtime_dashboard.html")
        fig.write_html(realtime_path)
        
        return realtime_path
    
    def generate_all_visualizations(self) -> Dict[str, str]:
        """Generate all visualizations and return paths"""
        print("Generating AI visualization dashboard...")
        
        visualizations = {
            'overview_dashboard': self.create_agent_overview_dashboard(),
            'improvement_timeline': self.create_improvement_timeline(),
            'performance_heatmap': self.create_performance_heatmap(),
            'learning_curves': self.create_learning_curves(),
            'agent_comparison_radar': self.create_agent_comparison_radar(),
            'realtime_dashboard': self.create_realtime_dashboard()
        }
        
        print(f"Visualizations generated in: {self.output_dir}")
        for name, path in visualizations.items():
            print(f"  - {name}: {path}")
        
        return visualizations

# Global visualization system instance
viz_system = AIVisualizationSystem()

if __name__ == "__main__":
    # Demo the visualization system
    print("AI Visualization System Demo")
    print("=" * 50)
    
    # Simulate some data
    ai_registry.simulate_agent_improvements()
    
    # Generate all visualizations
    viz_paths = viz_system.generate_all_visualizations()
    
    print("\nAll visualizations generated successfully!")
    print("Open the HTML files in your browser to view the dashboards.")