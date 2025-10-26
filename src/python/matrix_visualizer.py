"""
Matrix Visualization Panel for Real-Time AI Learning Display

This module provides a comprehensive visualization of AI learning matrices:
- Q-Learning state-action values
- Experience Replay buffer contents
- Parameter Evolution populations

Updates in real-time as the AI plays and learns.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
from typing import Optional, Dict, Any
import time


class MatrixVisualizationPanel:
    """
    Real-time visualization panel for AI learning matrices.
    
    Features:
    - Tabbed interface for different matrix types
    - Auto-refresh during gameplay
    - Color-coded values
    - Scrollable displays for large datasets
    """
    
    # Color scheme
    BG_COLOR = "#1a1f3a"
    PANEL_COLOR = "#0f1429"
    TEXT_COLOR = "#ffffff"
    POSITIVE_COLOR = "#4caf50"
    NEGATIVE_COLOR = "#f44336"
    NEUTRAL_COLOR = "#9e9e9e"
    HIGHLIGHT_COLOR = "#ffd700"
    
    def __init__(self, parent, bg_color=None, panel_color=None):
        """
        Initialize the visualization panel.
        
        Args:
            parent: Parent tkinter widget
            bg_color: Background color (optional)
            panel_color: Panel color (optional)
        """
        self.parent = parent
        self.bg_color = bg_color or self.BG_COLOR
        self.panel_color = panel_color or self.PANEL_COLOR
        
        # Reference to current strategy
        self.current_strategy = None
        self.is_visible = True
        self.auto_refresh = False
        self.refresh_interval = 500  # ms
        self.last_refresh = 0
        
        # Create main container
        self.container = tk.Frame(parent, bg=self.bg_color)
        
        # Create notebook (tabbed interface)
        self.notebook = ttk.Notebook(self.container)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create tabs
        self.create_q_learning_tab()
        self.create_experience_replay_tab()
        self.create_evolution_tab()
        self.create_stats_tab()
        
        # Control panel at bottom
        self.create_control_panel()
        
    def create_q_learning_tab(self):
        """Create Q-Learning matrix visualization tab."""
        tab = tk.Frame(self.notebook, bg=self.panel_color)
        self.notebook.add(tab, text="Q-Learning Matrix")
        
        # Header with statistics
        header = tk.Frame(tab, bg=self.panel_color)
        header.pack(fill=tk.X, padx=10, pady=5)
        
        self.q_stats_label = tk.Label(header, text="No Q-Matrix loaded",
                                      font=("Arial", 10, "bold"),
                                      bg=self.panel_color, fg=self.TEXT_COLOR)
        self.q_stats_label.pack(side=tk.LEFT)
        
        # Instructions
        info = tk.Label(tab, text="Displays learned state-action values. Green = positive, Red = negative.",
                       font=("Arial", 8), bg=self.panel_color, fg=self.NEUTRAL_COLOR)
        info.pack(pady=2)
        
        # Scrollable text area for Q-values
        text_frame = tk.Frame(tab, bg=self.panel_color)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.q_text = scrolledtext.ScrolledText(text_frame, height=20, width=60,
                                                font=("Consolas", 8),
                                                bg="#1e1e1e", fg=self.TEXT_COLOR,
                                                insertbackground=self.TEXT_COLOR)
        self.q_text.pack(fill=tk.BOTH, expand=True)
        
        # Configure tags for color coding
        self.q_text.tag_configure("positive", foreground=self.POSITIVE_COLOR)
        self.q_text.tag_configure("negative", foreground=self.NEGATIVE_COLOR)
        self.q_text.tag_configure("neutral", foreground=self.NEUTRAL_COLOR)
        self.q_text.tag_configure("highlight", foreground=self.HIGHLIGHT_COLOR, font=("Consolas", 8, "bold"))
        
    def create_experience_replay_tab(self):
        """Create Experience Replay buffer visualization tab."""
        tab = tk.Frame(self.notebook, bg=self.panel_color)
        self.notebook.add(tab, text="Experience Replay")
        
        # Header
        header = tk.Frame(tab, bg=self.panel_color)
        header.pack(fill=tk.X, padx=10, pady=5)
        
        self.replay_stats_label = tk.Label(header, text="No Replay Buffer loaded",
                                           font=("Arial", 10, "bold"),
                                           bg=self.panel_color, fg=self.TEXT_COLOR)
        self.replay_stats_label.pack(side=tk.LEFT)
        
        # Instructions
        info = tk.Label(tab, text="Recent experiences stored for learning. High-priority experiences are highlighted.",
                       font=("Arial", 8), bg=self.panel_color, fg=self.NEUTRAL_COLOR)
        info.pack(pady=2)
        
        # Scrollable text area
        text_frame = tk.Frame(tab, bg=self.panel_color)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.replay_text = scrolledtext.ScrolledText(text_frame, height=20, width=60,
                                                     font=("Consolas", 8),
                                                     bg="#1e1e1e", fg=self.TEXT_COLOR,
                                                     insertbackground=self.TEXT_COLOR)
        self.replay_text.pack(fill=tk.BOTH, expand=True)
        
        # Configure tags
        self.replay_text.tag_configure("high_priority", foreground=self.HIGHLIGHT_COLOR, font=("Consolas", 8, "bold"))
        self.replay_text.tag_configure("positive_reward", foreground=self.POSITIVE_COLOR)
        self.replay_text.tag_configure("negative_reward", foreground=self.NEGATIVE_COLOR)
        
    def create_evolution_tab(self):
        """Create Parameter Evolution visualization tab."""
        tab = tk.Frame(self.notebook, bg=self.panel_color)
        self.notebook.add(tab, text="Parameter Evolution")
        
        # Header
        header = tk.Frame(tab, bg=self.panel_color)
        header.pack(fill=tk.X, padx=10, pady=5)
        
        self.evolution_stats_label = tk.Label(header, text="No Evolution Matrix loaded",
                                              font=("Arial", 10, "bold"),
                                              bg=self.panel_color, fg=self.TEXT_COLOR)
        self.evolution_stats_label.pack(side=tk.LEFT)
        
        # Instructions
        info = tk.Label(tab, text="Parameter populations and fitness scores. Best individuals highlighted.",
                       font=("Arial", 8), bg=self.panel_color, fg=self.NEUTRAL_COLOR)
        info.pack(pady=2)
        
        # Scrollable text area
        text_frame = tk.Frame(tab, bg=self.panel_color)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.evolution_text = scrolledtext.ScrolledText(text_frame, height=20, width=60,
                                                        font=("Consolas", 8),
                                                        bg="#1e1e1e", fg=self.TEXT_COLOR,
                                                        insertbackground=self.TEXT_COLOR)
        self.evolution_text.pack(fill=tk.BOTH, expand=True)
        
        # Configure tags
        self.evolution_text.tag_configure("best", foreground=self.HIGHLIGHT_COLOR, font=("Consolas", 8, "bold"))
        self.evolution_text.tag_configure("good", foreground=self.POSITIVE_COLOR)
        self.evolution_text.tag_configure("poor", foreground=self.NEGATIVE_COLOR)
        
    def create_stats_tab(self):
        """Create overall learning statistics tab."""
        tab = tk.Frame(self.notebook, bg=self.panel_color)
        self.notebook.add(tab, text="Learning Stats")
        
        # Stats display
        stats_frame = tk.Frame(tab, bg=self.panel_color)
        stats_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        self.stats_text = scrolledtext.ScrolledText(stats_frame, height=20, width=60,
                                                    font=("Consolas", 9),
                                                    bg="#1e1e1e", fg=self.TEXT_COLOR,
                                                    insertbackground=self.TEXT_COLOR)
        self.stats_text.pack(fill=tk.BOTH, expand=True)
        
        # Configure tags
        self.stats_text.tag_configure("header", foreground=self.HIGHLIGHT_COLOR, font=("Consolas", 10, "bold"))
        self.stats_text.tag_configure("label", foreground=self.NEUTRAL_COLOR)
        self.stats_text.tag_configure("value", foreground=self.TEXT_COLOR, font=("Consolas", 9, "bold"))
        
    def create_control_panel(self):
        """Create control panel with refresh controls."""
        control = tk.Frame(self.container, bg=self.panel_color, relief=tk.RAISED, borderwidth=2)
        control.pack(fill=tk.X, padx=5, pady=5)
        
        # Auto-refresh toggle
        self.auto_refresh_var = tk.BooleanVar(value=False)
        auto_check = tk.Checkbutton(control, text="Auto-Refresh",
                                    variable=self.auto_refresh_var,
                                    command=self.toggle_auto_refresh,
                                    bg=self.panel_color, fg=self.TEXT_COLOR,
                                    selectcolor="#2d3561", font=("Arial", 9))
        auto_check.pack(side=tk.LEFT, padx=10, pady=5)
        
        # Manual refresh button
        refresh_btn = tk.Button(control, text="🔄 Refresh Now",
                               command=self.refresh_all,
                               bg="#1e88e5", fg="white",
                               font=("Arial", 9, "bold"),
                               relief=tk.FLAT, padx=10, pady=5,
                               cursor="hand2")
        refresh_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Last update label
        self.last_update_label = tk.Label(control, text="Last update: Never",
                                          font=("Arial", 8),
                                          bg=self.panel_color, fg=self.NEUTRAL_COLOR)
        self.last_update_label.pack(side=tk.RIGHT, padx=10)
        
    def set_strategy(self, strategy):
        """
        Set the strategy to visualize.
        
        Args:
            strategy: Strategy instance with learning matrices
        """
        self.current_strategy = strategy
        self.refresh_all()
        
    def toggle_auto_refresh(self):
        """Toggle auto-refresh on/off."""
        self.auto_refresh = self.auto_refresh_var.get()
        if self.auto_refresh:
            self.schedule_refresh()
            
    def schedule_refresh(self):
        """Schedule next auto-refresh."""
        if self.auto_refresh and self.is_visible:
            self.refresh_all()
            self.parent.after(self.refresh_interval, self.schedule_refresh)
            
    def refresh_all(self):
        """Refresh all matrix displays."""
        if not self.current_strategy:
            return
            
        self.refresh_q_learning()
        self.refresh_experience_replay()
        self.refresh_evolution()
        self.refresh_stats()
        
        # Update timestamp
        self.last_refresh = time.time()
        self.last_update_label.config(text=f"Last update: {time.strftime('%H:%M:%S')}")
        
    def refresh_q_learning(self):
        """Refresh Q-Learning matrix display."""
        self.q_text.delete(1.0, tk.END)
        
        if not self.current_strategy or not hasattr(self.current_strategy, 'q_matrix'):
            self.q_text.insert(tk.END, "Q-Learning not enabled for this strategy.\n")
            return
            
        q_matrix = self.current_strategy.q_matrix
        if not q_matrix:
            self.q_text.insert(tk.END, "Q-Matrix not initialized.\n")
            return
            
        # Update header stats
        total_states = len(q_matrix.q_table_a)
        total_updates = q_matrix.updates
        epsilon = q_matrix.epsilon
        avg_reward = q_matrix.total_reward / max(1, total_updates)
        
        stats_text = f"States: {total_states} | Updates: {total_updates} | ε: {epsilon:.4f} | Avg Reward: {avg_reward:.4f}"
        self.q_stats_label.config(text=stats_text)
        
        # Display top states by visit count
        self.q_text.insert(tk.END, "Top States by Experience:\n", "header")
        self.q_text.insert(tk.END, "=" * 70 + "\n")
        
        # Collect states with their total visit counts
        state_visits = {}
        for state_hash in q_matrix.q_table_a.keys():
            total_visits = sum(q_matrix.visit_counts[state_hash].values())
            if total_visits > 0:
                state_visits[state_hash] = total_visits
                
        # Sort by visit count
        sorted_states = sorted(state_visits.items(), key=lambda x: x[1], reverse=True)[:20]
        
        if not sorted_states:
            self.q_text.insert(tk.END, "\nNo experiences yet. Play some games to see Q-values!\n", "neutral")
            return
            
        for state_hash, visits in sorted_states:
            self.q_text.insert(tk.END, f"\nState: {state_hash[:12]}... (visited {visits} times)\n", "highlight")
            
            # Get Q-values for this state
            q_values_a = q_matrix.q_table_a[state_hash]
            q_values_b = q_matrix.q_table_b[state_hash] if q_matrix.use_double_q else {}
            
            # Display actions and Q-values
            for action_key, q_a in list(q_values_a.items())[:5]:  # Show top 5 actions
                q_b = q_values_b.get(action_key, 0.0)
                avg_q = (q_a + q_b) / 2.0 if q_matrix.use_double_q else q_a
                
                # Color code based on value
                tag = "positive" if avg_q > 0.1 else "negative" if avg_q < -0.1 else "neutral"
                
                self.q_text.insert(tk.END, f"  Action {action_key}: Q={avg_q:.4f}", tag)
                if q_matrix.use_double_q:
                    self.q_text.insert(tk.END, f" (A:{q_a:.3f}, B:{q_b:.3f})")
                self.q_text.insert(tk.END, "\n")
                
    def refresh_experience_replay(self):
        """Refresh Experience Replay buffer display."""
        self.replay_text.delete(1.0, tk.END)
        
        if not self.current_strategy or not hasattr(self.current_strategy, 'replay_buffer'):
            self.replay_text.insert(tk.END, "Experience Replay not enabled for this strategy.\n")
            return
            
        replay_buffer = self.current_strategy.replay_buffer
        if not replay_buffer:
            self.replay_text.insert(tk.END, "Replay Buffer not initialized.\n")
            return
            
        # Update header stats
        buffer_size = len(replay_buffer.buffer)
        max_size = replay_buffer.max_size
        is_prioritized = replay_buffer.prioritized
        
        stats_text = f"Buffer: {buffer_size}/{max_size} | Prioritized: {is_prioritized}"
        self.replay_stats_label.config(text=stats_text)
        
        # Display recent experiences
        self.replay_text.insert(tk.END, "Recent Experiences:\n", "header")
        self.replay_text.insert(tk.END, "=" * 70 + "\n")
        
        if buffer_size == 0:
            self.replay_text.insert(tk.END, "\nNo experiences stored yet.\n", "neutral")
            return
            
        # Show last 20 experiences
        recent_experiences = list(replay_buffer.buffer)[-20:]
        
        for i, exp in enumerate(reversed(recent_experiences)):
            # Get priority if available
            priority = replay_buffer.priorities[-i-1] if i < len(replay_buffer.priorities) else 0.0
            is_high_priority = priority > 0.5
            
            header_tag = "high_priority" if is_high_priority else ""
            reward_tag = "positive_reward" if exp.reward > 0 else "negative_reward" if exp.reward < 0 else ""
            
            self.replay_text.insert(tk.END, f"\nExp #{buffer_size - i}", header_tag)
            if is_high_priority:
                self.replay_text.insert(tk.END, " ⭐", "high_priority")
            self.replay_text.insert(tk.END, f" (Priority: {priority:.3f})\n")
            
            self.replay_text.insert(tk.END, f"  State: {exp.state_hash[:12]}...\n")
            self.replay_text.insert(tk.END, f"  Action: {exp.action}\n")
            self.replay_text.insert(tk.END, f"  Reward: {exp.reward:.4f}", reward_tag)
            self.replay_text.insert(tk.END, f" | Done: {exp.done}\n")
            
    def refresh_evolution(self):
        """Refresh Parameter Evolution display."""
        self.evolution_text.delete(1.0, tk.END)
        
        if not self.current_strategy or not hasattr(self.current_strategy, 'evolution_matrix'):
            self.evolution_text.insert(tk.END, "Parameter Evolution not enabled for this strategy.\n")
            return
            
        evolution_matrix = self.current_strategy.evolution_matrix
        if not evolution_matrix:
            self.evolution_text.insert(tk.END, "Evolution Matrix not initialized.\n")
            return
            
        # Update header stats
        generation = evolution_matrix.generation
        pop_size = len(evolution_matrix.population)
        mutation_rate = evolution_matrix.mutation_rate
        
        stats_text = f"Generation: {generation} | Population: {pop_size} | Mutation: {mutation_rate:.3f}"
        self.evolution_stats_label.config(text=stats_text)
        
        # Display population
        self.evolution_text.insert(tk.END, "Parameter Population:\n", "header")
        self.evolution_text.insert(tk.END, "=" * 70 + "\n")
        
        # Sort by fitness
        sorted_pop = sorted(enumerate(evolution_matrix.population),
                          key=lambda x: evolution_matrix.fitness_scores[x[0]],
                          reverse=True)
        
        for rank, (idx, params) in enumerate(sorted_pop[:10]):  # Show top 10
            fitness = evolution_matrix.fitness_scores[idx]
            
            # Tag based on fitness
            if rank == 0:
                tag = "best"
                prefix = "🏆 "
            elif rank < 3:
                tag = "good"
                prefix = "⭐ "
            else:
                tag = ""
                prefix = "   "
                
            self.evolution_text.insert(tk.END, f"\n{prefix}Individual #{idx + 1}", tag)
            self.evolution_text.insert(tk.END, f" (Rank: {rank + 1}, Fitness: {fitness:.4f})\n", tag)
            
            # Display parameters
            for param_name, value in params.items():
                self.evolution_text.insert(tk.END, f"  {param_name}: {value:.4f}\n")
                
    def refresh_stats(self):
        """Refresh overall learning statistics."""
        self.stats_text.delete(1.0, tk.END)
        
        if not self.current_strategy:
            self.stats_text.insert(tk.END, "No strategy loaded.\n")
            return
            
        # Header
        self.stats_text.insert(tk.END, "Learning Statistics\n", "header")
        self.stats_text.insert(tk.END, "=" * 60 + "\n\n")
        
        # Strategy name
        self.stats_text.insert(tk.END, "Strategy: ", "label")
        self.stats_text.insert(tk.END, f"{self.current_strategy.name}\n\n", "value")
        
        # Q-Learning stats
        if hasattr(self.current_strategy, 'q_matrix') and self.current_strategy.q_matrix:
            q = self.current_strategy.q_matrix
            self.stats_text.insert(tk.END, "Q-Learning:\n", "header")
            self.stats_text.insert(tk.END, f"  Total States Visited: ", "label")
            self.stats_text.insert(tk.END, f"{len(q.q_table_a)}\n", "value")
            self.stats_text.insert(tk.END, f"  Total Updates: ", "label")
            self.stats_text.insert(tk.END, f"{q.updates}\n", "value")
            self.stats_text.insert(tk.END, f"  Current Epsilon: ", "label")
            self.stats_text.insert(tk.END, f"{q.epsilon:.4f}\n", "value")
            self.stats_text.insert(tk.END, f"  Learning Rate: ", "label")
            self.stats_text.insert(tk.END, f"{q.alpha:.4f}\n", "value")
            self.stats_text.insert(tk.END, f"  Discount Factor: ", "label")
            self.stats_text.insert(tk.END, f"{q.gamma:.4f}\n", "value")
            self.stats_text.insert(tk.END, f"  Total Reward: ", "label")
            self.stats_text.insert(tk.END, f"{q.total_reward:.2f}\n\n", "value")
        
        # Replay buffer stats
        if hasattr(self.current_strategy, 'replay_buffer') and self.current_strategy.replay_buffer:
            rb = self.current_strategy.replay_buffer
            self.stats_text.insert(tk.END, "Experience Replay:\n", "header")
            self.stats_text.insert(tk.END, f"  Buffer Size: ", "label")
            self.stats_text.insert(tk.END, f"{len(rb.buffer)}/{rb.max_size}\n", "value")
            self.stats_text.insert(tk.END, f"  Prioritized: ", "label")
            self.stats_text.insert(tk.END, f"{rb.prioritized}\n", "value")
            if rb.buffer:
                avg_reward = sum(exp.reward for exp in rb.buffer) / len(rb.buffer)
                self.stats_text.insert(tk.END, f"  Avg Experience Reward: ", "label")
                self.stats_text.insert(tk.END, f"{avg_reward:.4f}\n\n", "value")
        
        # Evolution stats
        if hasattr(self.current_strategy, 'evolution_matrix') and self.current_strategy.evolution_matrix:
            em = self.current_strategy.evolution_matrix
            self.stats_text.insert(tk.END, "Parameter Evolution:\n", "header")
            self.stats_text.insert(tk.END, f"  Generation: ", "label")
            self.stats_text.insert(tk.END, f"{em.generation}\n", "value")
            self.stats_text.insert(tk.END, f"  Population Size: ", "label")
            self.stats_text.insert(tk.END, f"{len(em.population)}\n", "value")
            self.stats_text.insert(tk.END, f"  Mutation Rate: ", "label")
            self.stats_text.insert(tk.END, f"{em.mutation_rate:.4f}\n", "value")
            if em.fitness_scores:
                best_fitness = max(em.fitness_scores)
                avg_fitness = sum(em.fitness_scores) / len(em.fitness_scores)
                self.stats_text.insert(tk.END, f"  Best Fitness: ", "label")
                self.stats_text.insert(tk.END, f"{best_fitness:.4f}\n", "value")
                self.stats_text.insert(tk.END, f"  Avg Fitness: ", "label")
                self.stats_text.insert(tk.END, f"{avg_fitness:.4f}\n", "value")
                
    def pack(self, **kwargs):
        """Pack the container."""
        self.container.pack(**kwargs)
        
    def grid(self, **kwargs):
        """Grid the container."""
        self.container.grid(**kwargs)
        
    def pack_forget(self):
        """Hide the panel."""
        self.container.pack_forget()
        self.is_visible = False
        
    def grid_forget(self):
        """Hide the panel."""
        self.container.grid_forget()
        self.is_visible = False


if __name__ == "__main__":
    # Test the visualizer
    root = tk.Tk()
    root.title("Matrix Visualizer Test")
    root.geometry("800x600")
    root.configure(bg="#1a1f3a")
    
    panel = MatrixVisualizationPanel(root)
    panel.pack(fill=tk.BOTH, expand=True)
    
    root.mainloop()

