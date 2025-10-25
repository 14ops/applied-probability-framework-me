"""
GUI application for Applied Probability Framework.
Professional Windows interface for running Monte Carlo simulations.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading
import json
from pathlib import Path
from datetime import datetime

# Import framework components
from core.config import create_default_config
from core.plugin_system import get_registry
from core.parallel_engine import ParallelSimulationEngine
import register_plugins  # Register all plugins


class SimulationGUI:
    """Main GUI application for the framework."""
    
    def __init__(self, root):
        """Initialize the GUI."""
        self.root = root
        self.root.title("Applied Probability Framework - v1.0.0")
        self.root.geometry("900x700")
        self.root.resizable(True, True)
        
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Get registry
        self.registry = get_registry()
        
        # Variables
        self.running = False
        self.results = None
        
        # Create UI
        self.create_widgets()
        
        # Load available plugins
        self.load_plugins()
    
    def create_widgets(self):
        """Create all GUI widgets."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Title
        title = ttk.Label(main_frame, text="Applied Probability Framework", 
                         font=('Arial', 16, 'bold'))
        title.grid(row=0, column=0, pady=(0, 10))
        
        # Settings Frame
        settings_frame = ttk.LabelFrame(main_frame, text="Simulation Settings", padding="10")
        settings_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        settings_frame.columnconfigure(1, weight=1)
        
        # Simulator selection
        ttk.Label(settings_frame, text="Simulator:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.simulator_var = tk.StringVar()
        self.simulator_combo = ttk.Combobox(settings_frame, textvariable=self.simulator_var, 
                                           state='readonly', width=30)
        self.simulator_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=5, padx=(5, 0))
        
        # Strategy selection
        ttk.Label(settings_frame, text="Strategy:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.strategy_var = tk.StringVar()
        self.strategy_combo = ttk.Combobox(settings_frame, textvariable=self.strategy_var,
                                          state='readonly', width=30)
        self.strategy_combo.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5, padx=(5, 0))
        
        # Number of simulations
        ttk.Label(settings_frame, text="Simulations:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.num_sims_var = tk.StringVar(value="1000")
        num_sims_entry = ttk.Entry(settings_frame, textvariable=self.num_sims_var, width=32)
        num_sims_entry.grid(row=2, column=1, sticky=(tk.W, tk.E), pady=5, padx=(5, 0))
        
        # Parallel execution
        self.parallel_var = tk.BooleanVar(value=True)
        parallel_check = ttk.Checkbutton(settings_frame, text="Run in Parallel", 
                                        variable=self.parallel_var)
        parallel_check.grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        # Control buttons frame
        button_frame = ttk.Frame(settings_frame)
        button_frame.grid(row=4, column=0, columnspan=2, pady=(10, 0))
        
        self.run_button = ttk.Button(button_frame, text="▶ Run Simulation", 
                                     command=self.run_simulation, width=20)
        self.run_button.grid(row=0, column=0, padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="⏹ Stop", 
                                      command=self.stop_simulation, state='disabled', width=15)
        self.stop_button.grid(row=0, column=1, padx=5)
        
        self.save_button = ttk.Button(button_frame, text="💾 Save Results", 
                                      command=self.save_results, state='disabled', width=15)
        self.save_button.grid(row=0, column=2, padx=5)
        
        # Progress Frame
        progress_frame = ttk.LabelFrame(main_frame, text="Progress", padding="10")
        progress_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        progress_frame.columnconfigure(0, weight=1)
        progress_frame.rowconfigure(1, weight=1)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, 
                                           maximum=100, mode='determinate')
        self.progress_bar.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Results text area
        self.results_text = scrolledtext.ScrolledText(progress_frame, height=20, width=80,
                                                      font=('Consolas', 9))
        self.results_text.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=3, column=0, sticky=(tk.W, tk.E))
    
    def load_plugins(self):
        """Load available plugins into dropdowns."""
        # Load simulators
        simulators = self.registry.list_simulators()
        self.simulator_combo['values'] = simulators
        if simulators:
            self.simulator_var.set(simulators[0])
        
        # Load strategies
        strategies = self.registry.list_strategies()
        self.strategy_combo['values'] = strategies
        if strategies:
            self.strategy_var.set(strategies[0])
    
    def log(self, message):
        """Add message to results text area."""
        self.results_text.insert(tk.END, message + "\n")
        self.results_text.see(tk.END)
        self.root.update_idletasks()
    
    def run_simulation(self):
        """Run the simulation in a background thread."""
        if self.running:
            return
        
        # Validate inputs
        try:
            num_sims = int(self.num_sims_var.get())
            if num_sims <= 0:
                raise ValueError("Must be positive")
        except ValueError:
            messagebox.showerror("Invalid Input", "Number of simulations must be a positive integer")
            return
        
        simulator_name = self.simulator_var.get()
        strategy_name = self.strategy_var.get()
        
        if not simulator_name or not strategy_name:
            messagebox.showerror("Missing Selection", "Please select a simulator and strategy")
            return
        
        # Clear previous results
        self.results_text.delete(1.0, tk.END)
        self.progress_var.set(0)
        self.results = None
        
        # Update UI state
        self.running = True
        self.run_button['state'] = 'disabled'
        self.stop_button['state'] = 'normal'
        self.save_button['state'] = 'disabled'
        self.status_var.set("Running simulations...")
        
        # Log start
        self.log("=" * 60)
        self.log(f"Starting simulation: {simulator_name} + {strategy_name}")
        self.log(f"Number of runs: {num_sims}")
        self.log(f"Parallel: {self.parallel_var.get()}")
        self.log("=" * 60)
        self.log("")
        
        # Run in background thread
        thread = threading.Thread(target=self._run_simulation_thread,
                                 args=(simulator_name, strategy_name, num_sims))
        thread.daemon = True
        thread.start()
    
    def _run_simulation_thread(self, simulator_name, strategy_name, num_sims):
        """Background thread for running simulations."""
        try:
            # Create config
            config = create_default_config()
            config.simulation.num_simulations = num_sims
            config.simulation.parallel = self.parallel_var.get()
            
            # Create simulator and strategy
            simulator = self.registry.create_simulator(simulator_name)
            strategy = self.registry.create_strategy(strategy_name)
            
            # Create engine
            engine = ParallelSimulationEngine(config, n_jobs=-1 if self.parallel_var.get() else 1)
            
            # Run simulations with progress tracking
            class ProgressTracker:
                def __init__(self, gui):
                    self.gui = gui
                    self.count = 0
                    self.total = num_sims
                
                def update(self, n=1):
                    self.count += n
                    progress = (self.count / self.total) * 100
                    self.gui.root.after(0, self.gui.progress_var.set, progress)
                    if self.count % 50 == 0 or self.count == self.total:
                        msg = f"Progress: {self.count}/{self.total} ({progress:.1f}%)"
                        self.gui.root.after(0, self.gui.log, msg)
            
            tracker = ProgressTracker(self)
            
            # Simple progress tracking by running in batches
            results = engine.run_batch(simulator, strategy, num_runs=num_sims, show_progress=False)
            
            # Update progress to 100%
            self.root.after(0, self.progress_var.set, 100)
            
            # Display results
            self.root.after(0, self._display_results, results)
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            self.root.after(0, self.log, f"\n{error_msg}")
            self.root.after(0, messagebox.showerror, "Simulation Error", error_msg)
        finally:
            self.root.after(0, self._simulation_finished)
    
    def _display_results(self, results):
        """Display simulation results."""
        self.results = results
        
        self.log("")
        self.log("=" * 60)
        self.log("SIMULATION RESULTS")
        self.log("=" * 60)
        self.log(f"Total runs:        {results.num_runs}")
        self.log(f"Mean reward:       {results.mean_reward:.4f}")
        self.log(f"Std deviation:     {results.std_reward:.4f}")
        self.log(f"95% CI:            [{results.confidence_interval[0]:.4f}, {results.confidence_interval[1]:.4f}]")
        self.log(f"Success rate:      {results.success_rate * 100:.2f}%")
        self.log(f"Execution time:    {results.total_duration:.2f}s")
        self.log("=" * 60)
        self.log("")
        self.log("✅ Simulation completed successfully!")
        
        self.status_var.set("Simulation completed")
        self.save_button['state'] = 'normal'
    
    def _simulation_finished(self):
        """Called when simulation finishes."""
        self.running = False
        self.run_button['state'] = 'normal'
        self.stop_button['state'] = 'disabled'
    
    def stop_simulation(self):
        """Stop the running simulation."""
        # Note: This is a simple implementation
        # For proper stopping, we'd need to implement cancellation in the engine
        self.running = False
        self.log("\n⚠️ Stop requested (simulation will finish current batch)")
        self.status_var.set("Stopping...")
    
    def save_results(self):
        """Save results to JSON file."""
        if not self.results:
            messagebox.showwarning("No Results", "No results to save")
            return
        
        # Ask for save location
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfile=f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        if filename:
            try:
                # Create results dict
                results_dict = {
                    'num_runs': self.results.num_runs,
                    'mean_reward': self.results.mean_reward,
                    'std_reward': self.results.std_reward,
                    'confidence_interval': list(self.results.confidence_interval),
                    'success_rate': self.results.success_rate,
                    'total_duration': self.results.total_duration,
                    'convergence_achieved': self.results.convergence_achieved,
                    'timestamp': datetime.now().isoformat(),
                    'individual_results': [
                        {
                            'run_id': r.run_id,
                            'seed': r.seed,
                            'reward': r.reward,
                            'steps': r.steps,
                            'success': r.success,
                            'duration': r.duration
                        }
                        for r in self.results.all_results
                    ]
                }
                
                with open(filename, 'w') as f:
                    json.dump(results_dict, f, indent=2)
                
                self.log(f"\n💾 Results saved to: {filename}")
                messagebox.showinfo("Saved", f"Results saved successfully to:\n{filename}")
                
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save results:\n{str(e)}")


def main():
    """Main entry point for GUI application."""
    root = tk.Tk()
    app = SimulationGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

