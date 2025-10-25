"""
Mines Game GUI - Beautiful graphical interface for the Mines game.
Styled like a modern casino game with interactive gameplay.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import random
import time
from collections import deque
from game_simulator import GameSimulator
from core.plugin_system import get_registry
import register_plugins  # Register all strategies


class MinesGameGUI:
    """Beautiful GUI for playing the Mines game interactively."""
    
    # Colors matching the screenshot
    BG_COLOR = "#1a1f3a"
    PANEL_COLOR = "#0f1429"
    BUTTON_COLOR = "#1e88e5"
    BUTTON_HOVER = "#2196f3"
    CELL_COLOR = "#2d3561"
    CELL_HOVER = "#3d4571"
    SAFE_COLOR = "#4caf50"
    MINE_COLOR = "#f44336"
    TEXT_COLOR = "#ffffff"
    BORDER_COLOR = "#3d4571"
    
    def __init__(self, root):
        """Initialize the game GUI."""
        self.root = root
        self.root.title("Mines Game - Applied Probability Framework")
        self.root.configure(bg=self.BG_COLOR)
        self.root.geometry("1400x950")  # Increased size to ensure all buttons are visible
        self.root.minsize(1200, 800)  # Set minimum size
        
        # Game state
        self.grid_size = 5
        self.num_mines = 3
        self.game = None
        self.cells = []
        self.game_active = False
        self.clicks_made = 0
        self.current_multiplier = 1.0
        self.ai_playing = False
        self.current_strategy = None
        
        # Profit and statistics tracking
        self.starting_balance = 1000.0
        self.current_balance = 1000.0
        self.bet_amount = 10.0
        self.current_profit = 0.0
        self.bet_deducted = False
        self.strategy_stats = self.load_stats()
        
        # Auto run & win-rate over time
        self.autorun = False
        self.autorun_after_id = None
        self.win_history = deque()  # (ts, won:1/0)
        
        # Get registry for strategies
        self.registry = get_registry()
        
        # Create UI
        self.create_widgets()
        
        # Keyboard shortcuts
        try:
            self.root.bind('<Key-c>', lambda e: self.cash_out())
            self.root.bind('<Key-C>', lambda e: self.cash_out())
            self.root.bind('<Return>', lambda e: self.start_game())
            self.root.bind('<space>', lambda e: self.start_game())
        except Exception:
            pass
        
    def create_widgets(self):
        """Create all UI elements."""
        # Main container
        main_frame = tk.Frame(self.root, bg=self.BG_COLOR)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Top status bar: shows Balance and Final Multiplier
        top_bar = tk.Frame(main_frame, bg=self.BG_COLOR)
        top_bar.pack(fill=tk.X, side=tk.TOP, pady=(0, 10))
        
        self.top_balance_label = tk.Label(top_bar, text=f"Balance: ${self.current_balance:.2f}",
                                          font=("Arial", 12, "bold"), bg=self.BG_COLOR,
                                          fg="#ffc107")
        self.top_balance_label.pack(side=tk.LEFT)
        
        self.top_final_mult_label = tk.Label(top_bar, text="",
                                             font=("Arial", 12, "bold"), bg=self.BG_COLOR,
                                             fg="#ffd700")
        self.top_final_mult_label.pack(side=tk.RIGHT)
        
        # Left panel - scrollable controls
        scroll_container = tk.Frame(main_frame, bg=self.BG_COLOR, width=300)
        scroll_container.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        scroll_container.pack_propagate(False)

        # Body (scrollable area)
        body_frame = tk.Frame(scroll_container, bg=self.BG_COLOR)
        body_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(body_frame, bg=self.PANEL_COLOR, highlightthickness=0)
        vscroll = tk.Scrollbar(body_frame, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vscroll.set)

        vscroll.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        left_panel = tk.Frame(canvas, bg=self.PANEL_COLOR, width=280)
        canvas.create_window((0, 0), window=left_panel, anchor='nw')

        # Update scrollregion when content changes
        left_panel.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        # Mouse wheel scrolling (Windows)
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # Footer (pinned, non-scrollable)
        footer = tk.Frame(scroll_container, bg=self.PANEL_COLOR)
        footer.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Cash Out button in footer (always visible, pinned at bottom)
        self.cashout_button = tk.Button(footer, text="💰 CASH OUT & SECURE PROFIT",
                                       command=self.cash_out,
                                       bg="#4caf50", fg="white",
                                       font=("Arial", 16, "bold"), relief=tk.RAISED,
                                       padx=25, pady=18, cursor="hand2", state=tk.NORMAL,
                                       activebackground="#45a049", borderwidth=4,
                                       highlightbackground="#4caf50", highlightthickness=2)
        self.cashout_button.pack(pady=20, padx=20, fill=tk.X)
        
        # Add hover effect for Cash Out button
        def on_cashout_enter(e):
            self.cashout_button.config(bg="#45a049", relief=tk.SUNKEN, font=("Arial", 17, "bold"))
        def on_cashout_leave(e):
            self.cashout_button.config(bg="#4caf50", relief=tk.RAISED, font=("Arial", 16, "bold"))
        self.cashout_button.bind("<Enter>", on_cashout_enter)
        self.cashout_button.bind("<Leave>", on_cashout_leave)
        
        # Title in left panel
        title_label = tk.Label(left_panel, text="MINES", font=("Arial", 20, "bold"),
                              bg=self.PANEL_COLOR, fg=self.TEXT_COLOR)
        title_label.pack(pady=(20, 30))
        
        # Grid Size selector
        tk.Label(left_panel, text="Grid Size", font=("Arial", 10),
                bg=self.PANEL_COLOR, fg=self.TEXT_COLOR).pack(pady=(10, 5))
        
        grid_frame = tk.Frame(left_panel, bg=self.PANEL_COLOR)
        grid_frame.pack(pady=5)
        
        self.grid_buttons = {}
        for size in [3, 4, 5, 6]:
            btn = tk.Button(grid_frame, text=f"{size}x{size}", 
                          command=lambda s=size: self.set_grid_size(s),
                          bg=self.BUTTON_COLOR if size == 5 else self.CELL_COLOR,
                          fg=self.TEXT_COLOR, font=("Arial", 9, "bold"),
                          relief=tk.FLAT, padx=8, pady=5, cursor="hand2")
            btn.pack(side=tk.LEFT, padx=2)
            self.grid_buttons[size] = btn
        
        # Play Mode
        tk.Label(left_panel, text="Play Mode", font=("Arial", 10),
                bg=self.PANEL_COLOR, fg=self.TEXT_COLOR).pack(pady=(20, 5))
        
        self.play_mode_var = tk.StringVar(value="manual")
        mode_frame = tk.Frame(left_panel, bg=self.PANEL_COLOR)
        mode_frame.pack(pady=5)
        
        tk.Radiobutton(mode_frame, text="Manual", variable=self.play_mode_var, 
                      value="manual", bg=self.PANEL_COLOR, fg=self.TEXT_COLOR,
                      selectcolor=self.CELL_COLOR, font=("Arial", 9)).pack(anchor=tk.W)
        tk.Radiobutton(mode_frame, text="AI Plays", variable=self.play_mode_var, 
                      value="ai", bg=self.PANEL_COLOR, fg=self.TEXT_COLOR,
                      selectcolor=self.CELL_COLOR, font=("Arial", 9)).pack(anchor=tk.W)
        tk.Radiobutton(mode_frame, text="AI Hints", variable=self.play_mode_var, 
                      value="hints", bg=self.PANEL_COLOR, fg=self.TEXT_COLOR,
                      selectcolor=self.CELL_COLOR, font=("Arial", 9)).pack(anchor=tk.W)
        
        # Strategy selection with character info
        strategy_frame = tk.Frame(left_panel, bg=self.PANEL_COLOR)
        strategy_frame.pack(pady=(10, 5), fill=tk.X, padx=20)

        tk.Label(strategy_frame, text="Strategy", font=("Arial", 10, "bold"),
                bg=self.PANEL_COLOR, fg=self.TEXT_COLOR).pack(anchor=tk.W)

        self.strategy_var = tk.StringVar()
        self.strategy_combo = ttk.Combobox(strategy_frame, textvariable=self.strategy_var,
                                          state='readonly', width=20)
        self.strategy_combo.pack(pady=(2, 0), fill=tk.X)

        # Character description
        self.strategy_desc = tk.Label(strategy_frame, text="Select a strategy above",
                                     font=("Arial", 8), bg=self.PANEL_COLOR,
                                     fg="#cccccc", wraplength=200, justify=tk.LEFT)
        self.strategy_desc.pack(pady=(5, 0), fill=tk.X)

        # Bet Amount
        bet_frame = tk.LabelFrame(left_panel, text="Bet Amount", bg=self.PANEL_COLOR,
                                   fg=self.TEXT_COLOR, font=("Arial", 10, "bold"))
        bet_frame.pack(pady=12, padx=20, fill=tk.X)

        self.bet_var = tk.StringVar(value=f"{self.bet_amount:.2f}")
        bet_entry = tk.Entry(bet_frame, textvariable=self.bet_var, justify='right')
        bet_entry.pack(fill=tk.X, pady=(6, 8))
        
        # AI Bet Info Label
        self.ai_bet_info = tk.Label(bet_frame, text="", bg=self.PANEL_COLOR,
                                     fg="#ffd700", font=("Arial", 8, "italic"), wraplength=240)
        self.ai_bet_info.pack(fill=tk.X, pady=(2, 4))

        # Quick bet controls
        quick_frame = tk.Frame(bet_frame, bg=self.PANEL_COLOR)
        quick_frame.pack(fill=tk.X)
        tk.Button(quick_frame, text="1/2", width=6, command=self._bet_half,
                  bg=self.CELL_COLOR, fg=self.TEXT_COLOR, relief=tk.FLAT).pack(side=tk.LEFT, padx=2)
        tk.Button(quick_frame, text="2x", width=6, command=self._bet_double,
                  bg=self.CELL_COLOR, fg=self.TEXT_COLOR, relief=tk.FLAT).pack(side=tk.LEFT, padx=2)
        tk.Button(quick_frame, text="Max", width=6, command=self._bet_max,
                  bg=self.CELL_COLOR, fg=self.TEXT_COLOR, relief=tk.FLAT).pack(side=tk.LEFT, padx=2)

        # Primary Play button (prominent and near the top)
        self.start_button = tk.Button(left_panel, text="▶ PLAY", command=self.start_game,
                                      bg="#1e88e5", fg="white",
                                      font=("Arial", 16, "bold"), relief=tk.RAISED,
                                      padx=25, pady=15, cursor="hand2",
                                      activebackground="#1565c0", borderwidth=3,
                                      highlightbackground="#1e88e5", highlightthickness=2)
        self.start_button.pack(pady=14, padx=20, fill=tk.X)
        
        # Add hover effect for Play button
        def on_play_enter(e):
            if self.start_button['state'] != tk.DISABLED:
                self.start_button.config(bg="#1565c0", relief=tk.SUNKEN)
        def on_play_leave(e):
            if self.start_button['state'] != tk.DISABLED:
                self.start_button.config(bg="#1e88e5", relief=tk.RAISED)
        self.start_button.bind("<Enter>", on_play_enter)
        self.start_button.bind("<Leave>", on_play_leave)

        # Strategy Tuning
        tuning = tk.LabelFrame(left_panel, text="Strategy Tuning", bg=self.PANEL_COLOR,
                               fg=self.TEXT_COLOR, font=("Arial", 10, "bold"))
        tuning.pack(pady=10, padx=20, fill=tk.X)
        tk.Label(tuning, text="Aggression", bg=self.PANEL_COLOR, fg=self.TEXT_COLOR,
                 font=("Arial", 9)).pack(anchor=tk.W)
        self.aggr_var = tk.DoubleVar(value=0.50)
        tk.Scale(tuning, from_=0.30, to=0.70, resolution=0.01, orient=tk.HORIZONTAL,
                 variable=self.aggr_var, bg=self.PANEL_COLOR, fg=self.TEXT_COLOR,
                 troughcolor=self.CELL_COLOR, highlightthickness=0).pack(fill=tk.X)
        tk.Label(tuning, text="Max Risk (Kazuya)", bg=self.PANEL_COLOR, fg=self.TEXT_COLOR,
                 font=("Arial", 9)).pack(anchor=tk.W)
        self.maxrisk_var = tk.DoubleVar(value=0.20)
        tk.Scale(tuning, from_=0.05, to=0.35, resolution=0.01, orient=tk.HORIZONTAL,
                 variable=self.maxrisk_var, bg=self.PANEL_COLOR, fg=self.TEXT_COLOR,
                 troughcolor=self.CELL_COLOR, highlightthickness=0).pack(fill=tk.X)

        # Auto-evolution controls
        evo_frame = tk.LabelFrame(left_panel, text="Auto-Evolution", bg=self.PANEL_COLOR,
                                  fg=self.TEXT_COLOR, font=("Arial", 10, "bold"))
        evo_frame.pack(pady=10, padx=20, fill=tk.X)
        self.evo_enabled = tk.BooleanVar(value=True)
        tk.Checkbutton(evo_frame, text="Enable Evolution", variable=self.evo_enabled,
                       bg=self.PANEL_COLOR, fg=self.TEXT_COLOR, selectcolor=self.CELL_COLOR).pack(anchor=tk.W)
        tk.Label(evo_frame, text="Learning Rate", bg=self.PANEL_COLOR, fg=self.TEXT_COLOR,
                 font=("Arial", 9)).pack(anchor=tk.W)
        self.evo_rate = tk.DoubleVar(value=1.0)
        tk.Scale(evo_frame, from_=0.5, to=2.0, resolution=0.1, orient=tk.HORIZONTAL,
                 variable=self.evo_rate, bg=self.PANEL_COLOR, fg=self.TEXT_COLOR,
                 troughcolor=self.CELL_COLOR, highlightthickness=0).pack(fill=tk.X)

        # Fair Play Seed
        seed_frame = tk.LabelFrame(left_panel, text="Fair Play Seed", bg=self.PANEL_COLOR,
                                   fg=self.TEXT_COLOR, font=("Arial", 10, "bold"))
        seed_frame.pack(pady=10, padx=20, fill=tk.X)
        self.seed_var = tk.StringVar(value="")
        tk.Entry(seed_frame, textvariable=self.seed_var, justify='left').pack(fill=tk.X, pady=4)
        self.fair_hash_label = tk.Label(seed_frame, text="Hash: —", bg=self.PANEL_COLOR,
                                        fg="#888888", font=("Arial", 8), wraplength=220, justify=tk.LEFT)
        self.fair_hash_label.pack(fill=tk.X)

        # Logging
        log_frame = tk.LabelFrame(left_panel, text="Logging", bg=self.PANEL_COLOR,
                                  fg=self.TEXT_COLOR, font=("Arial", 10, "bold"))
        log_frame.pack(pady=10, padx=20, fill=tk.X)
        self.logging_enabled = tk.BooleanVar(value=True)
        tk.Checkbutton(log_frame, text="Save sessions to logs/sessions.jsonl",
                       variable=self.logging_enabled, bg=self.PANEL_COLOR,
                       fg=self.TEXT_COLOR, selectcolor=self.CELL_COLOR).pack(anchor=tk.W)

        # Auto Run
        ar_frame = tk.LabelFrame(left_panel, text="Auto Run", bg=self.PANEL_COLOR,
                                  fg=self.TEXT_COLOR, font=("Arial", 10, "bold"))
        ar_frame.pack(pady=10, padx=20, fill=tk.X)
        self.autorun_interval = tk.IntVar(value=1200)
        tk.Label(ar_frame, text="Interval (ms)", bg=self.PANEL_COLOR, fg=self.TEXT_COLOR,
                 font=("Arial", 9)).pack(anchor=tk.W)
        tk.Entry(ar_frame, textvariable=self.autorun_interval, justify='right').pack(fill=tk.X, pady=(2,6))
        self.winrate_window = tk.IntVar(value=60)
        tk.Label(ar_frame, text="Win-Rate Window (s)", bg=self.PANEL_COLOR, fg=self.TEXT_COLOR,
                 font=("Arial", 9)).pack(anchor=tk.W)
        tk.Scale(ar_frame, from_=10, to=300, orient=tk.HORIZONTAL, resolution=10,
                 variable=self.winrate_window, bg=self.PANEL_COLOR, fg=self.TEXT_COLOR,
                 troughcolor=self.CELL_COLOR, highlightthickness=0, command=lambda _=None: self.update_time_winrate()).pack(fill=tk.X)
        self.autorun_button = tk.Button(ar_frame, text="▶ START AUTO RUN", command=self.toggle_auto_run,
                                        bg="#ff6f00", fg="white", relief=tk.RAISED,
                                        font=("Arial", 11, "bold"), cursor="hand2",
                                        padx=15, pady=10, borderwidth=2,
                                        activebackground="#e65100")
        self.autorun_button.pack(fill=tk.X, pady=(6,2))
        
        # Add hover effect for Auto Run button
        def on_autorun_enter(e):
            if not self.autorun:
                self.autorun_button.config(bg="#e65100", relief=tk.SUNKEN)
        def on_autorun_leave(e):
            if not self.autorun:
                self.autorun_button.config(bg="#ff6f00", relief=tk.RAISED)
        self.autorun_button.bind("<Enter>", on_autorun_enter)
        self.autorun_button.bind("<Leave>", on_autorun_leave)
        # Rolling Win Rate Display (clickable to refresh)
        self.time_winrate_button = tk.Button(ar_frame, text="📊 Win Rate (last 60s): 0.0%",
                                             command=self.update_time_winrate,
                                             bg="#1e88e5", fg="white", relief=tk.RAISED,
                                             font=("Arial", 10, "bold"), cursor="hand2",
                                             padx=10, pady=8, borderwidth=2,
                                             activebackground="#1565c0")
        self.time_winrate_button.pack(fill=tk.X, pady=(6,2))
        
        # Add hover effect for Win Rate button
        def on_winrate_enter(e):
            self.time_winrate_button.config(bg="#1565c0", relief=tk.SUNKEN)
        def on_winrate_leave(e):
            self.time_winrate_button.config(bg="#1e88e5", relief=tk.RAISED)
        self.time_winrate_button.bind("<Enter>", on_winrate_enter)
        self.time_winrate_button.bind("<Leave>", on_winrate_leave)

        # Character descriptions
        self.character_descriptions = {
            'takeshi': '🔥 High-risk, high-reward aggressive play. Escalates betting when advantage is high.',
            'lelouch': '🧠 Strategic mastermind. Long-term planning with psychological advantage tactics.',
            'kazuya': '🛡️ Conservative survivor. Extreme risk aversion with capital preservation focus.',
            'senku': '🔬 Analytical scientist. Pure data-driven EV maximization with systematic exploration.',
            'okabe': '⚡ Mad scientist. Meta-game manipulation with worldline convergence and intuition.',
            'hybrid': '⭐ Ultimate fusion. Dynamic blending of Senku and Lelouch based on game phase.',
            'conservative': '📊 Basic conservative strategy using Bayesian estimation.',
            'random': '🎲 Random action selection for baseline comparison.',
            'aggressive': '💥 Basic aggressive strategy that takes more risks.'
        }

        # Load strategies with display names (avoiding duplicates)
        self.strategy_map = {
            'takeshi': 'Takeshi Kovacs - Aggressive Berserker',
            'lelouch': 'Lelouch vi Britannia - Strategic Mastermind',
            'kazuya': 'Kazuya Kinoshita - Conservative Survivor',
            'senku': 'Senku Ishigami - Analytical Scientist',
            'okabe': 'Rintaro Okabe - Mad Scientist',
            'hybrid': 'Hybrid Strategy - Ultimate Fusion',
            'conservative': 'Conservative (Basic)',
            'random': 'Random',
            'aggressive': 'Aggressive (Basic)'
        }
        
        strategies = self.registry.list_strategies()
        # Remove duplicates by keeping only preferred keys
        seen_display = set()
        unique_strategies = []
        preferred_keys = ['takeshi', 'lelouch', 'kazuya', 'senku', 'okabe', 'hybrid', 'conservative', 'random', 'aggressive']
        
        for key in preferred_keys:
            if key in strategies:
                unique_strategies.append(key)
        
        # Add any remaining strategies not in preferred list
        for s in strategies:
            display_name = self.strategy_map.get(s, s)
            if display_name not in seen_display and s not in unique_strategies:
                unique_strategies.append(s)
            seen_display.add(display_name)
        
        display_names = [self.strategy_map.get(s, s) for s in unique_strategies]
        self.strategy_combo['values'] = display_names
        if display_names:
            self.strategy_var.set(display_names[0])

        # Bind strategy selection to update description
        self.strategy_var.trace('w', self.update_strategy_description)
        
        # Number of Mines
        tk.Label(left_panel, text="Number of Mines", font=("Arial", 10),
                bg=self.PANEL_COLOR, fg=self.TEXT_COLOR).pack(pady=(20, 5))
        
        mines_frame = tk.Frame(left_panel, bg=self.PANEL_COLOR)
        mines_frame.pack(pady=5, fill=tk.X, padx=20)
        
        self.mines_var = tk.IntVar(value=3)
        self.mines_slider = tk.Scale(mines_frame, from_=1, to=10, orient=tk.HORIZONTAL,
                                     variable=self.mines_var, bg=self.PANEL_COLOR,
                                     fg=self.TEXT_COLOR, troughcolor=self.CELL_COLOR,
                                     highlightthickness=0, command=self.update_mines_label)
        self.mines_slider.pack(fill=tk.X)
        
        self.mines_label = tk.Label(left_panel, text="💣 3 Mines", font=("Arial", 10, "bold"),
                                    bg=self.PANEL_COLOR, fg=self.MINE_COLOR)
        self.mines_label.pack(pady=5)
        
        # (Start button moved above near Bet Amount)
        
        # Stats display
        stats_frame = tk.Frame(left_panel, bg=self.PANEL_COLOR)
        stats_frame.pack(pady=20, fill=tk.X, padx=20)
        
        tk.Label(stats_frame, text="Clicks Made:", font=("Arial", 9),
                bg=self.PANEL_COLOR, fg=self.TEXT_COLOR).pack()
        self.clicks_label = tk.Label(stats_frame, text="0", font=("Arial", 14, "bold"),
                                     bg=self.PANEL_COLOR, fg=self.SAFE_COLOR)
        self.clicks_label.pack(pady=2)
        
        tk.Label(stats_frame, text="Safe Cells Left:", font=("Arial", 9),
                bg=self.PANEL_COLOR, fg=self.TEXT_COLOR).pack(pady=(10, 0))
        self.safe_cells_label = tk.Label(stats_frame, text="22", font=("Arial", 14, "bold"),
                                         bg=self.PANEL_COLOR, fg=self.BUTTON_COLOR)
        self.safe_cells_label.pack(pady=2)
        
        # Profit display - make it more prominent
        profit_frame = tk.Frame(stats_frame, bg=self.PANEL_COLOR)
        profit_frame.pack(pady=(5, 0), fill=tk.X, padx=20)

        tk.Label(profit_frame, text="💰 CURRENT PROFIT", font=("Arial", 9, "bold"),
                bg=self.PANEL_COLOR, fg=self.SAFE_COLOR).pack()
        self.profit_label = tk.Label(profit_frame, text="$0.00", font=("Arial", 16, "bold"),
                                     bg=self.PANEL_COLOR, fg=self.SAFE_COLOR)
        self.profit_label.pack(pady=2)

        # Balance display
        balance_frame = tk.Frame(stats_frame, bg=self.PANEL_COLOR)
        balance_frame.pack(pady=(5, 0), fill=tk.X, padx=20)

        tk.Label(balance_frame, text="🏦 TOTAL BALANCE", font=("Arial", 9, "bold"),
                bg=self.PANEL_COLOR, fg="#ffc107").pack()
        self.balance_label = tk.Label(balance_frame, text="$1000.00", font=("Arial", 16, "bold"),
                                      bg=self.PANEL_COLOR, fg="#ffc107")
        self.balance_label.pack(pady=2)
        
        # AI Betting Dashboard
        ai_bet_dashboard = tk.LabelFrame(left_panel, text="🤖 AI Betting Dashboard", 
                                         bg=self.PANEL_COLOR, fg=self.TEXT_COLOR,
                                         font=("Arial", 10, "bold"))
        ai_bet_dashboard.pack(pady=15, padx=20, fill=tk.X)
        
        # Dictionary to store AI bet labels
        self.ai_bet_labels = {}
        
        # Define all AI characters with their colors
        ai_characters = [
            ("🔥 Takeshi", "#ff6f00", "takeshi"),
            ("🧠 Lelouch", "#9c27b0", "lelouch"),
            ("🛡️ Kazuya", "#4caf50", "kazuya"),
            ("🔬 Senku", "#2196f3", "senku"),
            ("🎭 Okabe", "#ff5722", "okabe"),
            ("⚡ Hybrid", "#ffc107", "hybrid")
        ]
        
        for emoji_name, color, key in ai_characters:
            char_frame = tk.Frame(ai_bet_dashboard, bg=self.PANEL_COLOR)
            char_frame.pack(fill=tk.X, pady=2, padx=5)
            
            # Character name
            name_label = tk.Label(char_frame, text=emoji_name, font=("Arial", 9, "bold"),
                                 bg=self.PANEL_COLOR, fg=color, width=12, anchor='w')
            name_label.pack(side=tk.LEFT)
            
            # Bet amount
            bet_label = tk.Label(char_frame, text="—", font=("Arial", 9),
                                bg=self.PANEL_COLOR, fg=self.TEXT_COLOR, anchor='e')
            bet_label.pack(side=tk.RIGHT, fill=tk.X, expand=True)
            
            self.ai_bet_labels[key] = bet_label
        
        # Auto-refresh indicator
        self.ai_refresh_label = tk.Label(ai_bet_dashboard, text="🔄 Auto-updating every 3s",
                                         bg=self.PANEL_COLOR, fg="#888888",
                                         font=("Arial", 7, "italic"))
        self.ai_refresh_label.pack(pady=(8, 5))
        
        # Start automatic refresh
        self.start_ai_bet_auto_refresh()
        
        # Strategy Statistics
        strategy_stats_frame = tk.LabelFrame(left_panel, text="Strategy Stats", 
                                             bg=self.PANEL_COLOR, fg=self.TEXT_COLOR,
                                             font=("Arial", 10, "bold"))
        strategy_stats_frame.pack(pady=20, padx=20, fill=tk.BOTH)
        
        self.wins_label = tk.Label(strategy_stats_frame, text="Wins: 0", font=("Arial", 9),
                                   bg=self.PANEL_COLOR, fg=self.SAFE_COLOR)
        self.wins_label.pack(pady=2)
        
        self.losses_label = tk.Label(strategy_stats_frame, text="Losses: 0", font=("Arial", 9),
                                     bg=self.PANEL_COLOR, fg=self.MINE_COLOR)
        self.losses_label.pack(pady=2)
        
        self.winrate_label = tk.Label(strategy_stats_frame, text="Win Rate: 0.0%", font=("Arial", 9),
                                      bg=self.PANEL_COLOR, fg=self.TEXT_COLOR)
        self.winrate_label.pack(pady=2)
        
        self.total_profit_label = tk.Label(strategy_stats_frame, text="Total Profit: $0.00", font=("Arial", 9),
                                           bg=self.PANEL_COLOR, fg=self.TEXT_COLOR)
        self.total_profit_label.pack(pady=2)
        
        # Multiplier display - make it more prominent
        multiplier_frame = tk.Frame(stats_frame, bg=self.PANEL_COLOR)
        multiplier_frame.pack(pady=(10, 0), fill=tk.X, padx=20)

        tk.Label(multiplier_frame, text="🎯 CURRENT MULTIPLIER", font=("Arial", 9, "bold"),
                bg=self.PANEL_COLOR, fg="#ffd700").pack()
        self.multiplier_label = tk.Label(multiplier_frame, text="1.00x", font=("Arial", 18, "bold"),
                                         bg=self.PANEL_COLOR, fg="#ffd700")
        self.multiplier_label.pack(pady=2)
        
        # Results Dashboard
        dash = tk.LabelFrame(left_panel, text="Results Dashboard", bg=self.PANEL_COLOR,
                              fg=self.TEXT_COLOR, font=("Arial", 10, "bold"))
        dash.pack(pady=10, padx=20, fill=tk.BOTH)
        self.dash_games = tk.Label(dash, text="Games: 0", bg=self.PANEL_COLOR, fg=self.TEXT_COLOR)
        self.dash_games.pack(anchor=tk.W)
        self.dash_avg_mult = tk.Label(dash, text="Avg Mult: 0.00x", bg=self.PANEL_COLOR, fg=self.TEXT_COLOR)
        self.dash_avg_mult.pack(anchor=tk.W)
        self.dash_roi = tk.Label(dash, text="ROI: 0.0%", bg=self.PANEL_COLOR, fg=self.TEXT_COLOR)
        self.dash_roi.pack(anchor=tk.W)
        self.metrics = {'games': 0, 'sum_mult': 0.0, 'sum_bet': 0.0, 'sum_profit': 0.0}

        # Rules panel
        rules_frame = tk.LabelFrame(left_panel, text="Rules", bg=self.PANEL_COLOR,
                                     fg=self.TEXT_COLOR, font=("Arial", 10, "bold"))
        rules_frame.pack(pady=10, padx=20, fill=tk.BOTH)
        rules_text = (
            "1) Reveal safe tiles to increase the payout multiplier.\n"
            "2) Cash out at any time to lock the current multiplier.\n"
            "3) Hitting a bomb ends the game and loses the wager.\n"
            "4) More configured bombs = higher multipliers per reveal."
        )
        tk.Label(rules_frame, text=rules_text, justify=tk.LEFT, anchor='w',
                 bg=self.PANEL_COLOR, fg=self.TEXT_COLOR, wraplength=220,
                 font=("Arial", 9)).pack(fill=tk.X, padx=6, pady=6)

        # AI Control buttons
        ai_button_frame = tk.Frame(left_panel, bg=self.PANEL_COLOR)
        ai_button_frame.pack(pady=10, padx=20, fill=tk.X)
        
        self.hint_button = tk.Button(ai_button_frame, text="💡 Get Hint",
                                     command=self.get_hint,
                                     bg="#ff9800", fg=self.TEXT_COLOR,
                                     font=("Arial", 10, "bold"), relief=tk.FLAT,
                                     padx=10, pady=8, cursor="hand2", state=tk.DISABLED)
        self.hint_button.pack(fill=tk.X, pady=2)
        
        self.auto_play_button = tk.Button(ai_button_frame, text="🤖 AI Auto-Play",
                                         command=self.toggle_ai_play,
                                         bg="#9c27b0", fg=self.TEXT_COLOR,
                                         font=("Arial", 10, "bold"), relief=tk.FLAT,
                                         padx=10, pady=8, cursor="hand2", state=tk.DISABLED)
        self.auto_play_button.pack(fill=tk.X, pady=2)

        # Right panel - game grid
        self.grid_container = tk.Frame(main_frame, bg=self.BG_COLOR)
        self.grid_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create initial grid
        self.create_grid()
        
    def create_grid(self):
        """Create the game grid."""
        # Clear existing grid
        for widget in self.grid_container.winfo_children():
            widget.destroy()
        
        self.cells = []
        
        # Center the grid
        grid_frame = tk.Frame(self.grid_container, bg=self.BG_COLOR)
        grid_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        
        # Calculate cell size based on grid
        cell_size = max(80, 500 // self.grid_size)
        
        for row in range(self.grid_size):
            row_cells = []
            for col in range(self.grid_size):
                cell_frame = tk.Frame(grid_frame, bg=self.BORDER_COLOR, 
                                     width=cell_size, height=cell_size)
                cell_frame.grid(row=row, column=col, padx=2, pady=2)
                cell_frame.pack_propagate(False)
                
                cell = tk.Button(cell_frame, text="", 
                               bg=self.CELL_COLOR, fg=self.TEXT_COLOR,
                               font=("Arial", 24), relief=tk.FLAT,
                               command=lambda r=row, c=col: self.click_cell(r, c),
                               cursor="hand2", state=tk.DISABLED)
                cell.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)
                
                # Hover effects
                cell.bind("<Enter>", lambda e, b=cell: self.on_hover(e, b))
                cell.bind("<Leave>", lambda e, b=cell: self.on_leave(e, b))
                
                row_cells.append(cell)
            self.cells.append(row_cells)
    
    def on_hover(self, event, button):
        """Handle mouse hover."""
        if button['state'] != tk.DISABLED and self.game_active:
            button.config(bg=self.CELL_HOVER)
    
    def on_leave(self, event, button):
        """Handle mouse leave."""
        if button['state'] != tk.DISABLED and self.game_active:
            button.config(bg=self.CELL_COLOR)
    
    def set_grid_size(self, size):
        """Set the grid size."""
        if self.game_active:
            messagebox.showwarning("Game Active", "Finish current game first!")
            return
        
        self.grid_size = size
        
        # Update button colors
        for s, btn in self.grid_buttons.items():
            btn.config(bg=self.BUTTON_COLOR if s == size else self.CELL_COLOR)
        
        # Update mines slider max
        max_mines = (size * size) // 2
        self.mines_slider.config(to=max_mines)
        if self.mines_var.get() > max_mines:
            self.mines_var.set(max_mines)
        
        self.create_grid()
        self.update_safe_cells_label()
    
    def update_mines_label(self, value=None):
        """Update the mines count label."""
        num_mines = self.mines_var.get()
        self.mines_label.config(text=f"💣 {num_mines} Mine{'s' if num_mines != 1 else ''}")
        self.update_safe_cells_label()
    
    def update_safe_cells_label(self):
        """Update safe cells remaining label."""
        if self.game_active and self.game:
            safe_left = self.game.max_clicks - self.clicks_made
        else:
            safe_left = (self.grid_size * self.grid_size) - self.mines_var.get()
        self.safe_cells_label.config(text=str(safe_left))
    
    def start_game(self):
        """Start a new game."""
        self.num_mines = self.mines_var.get()
        # Apply optional seed for fair play
        seed_text = (self.seed_var.get() or "").strip()
        seed_val = None
        if seed_text:
            try:
                seed_val = int(seed_text)
            except Exception:
                seed_val = None
        self.game = GameSimulator(board_size=self.grid_size, mine_count=self.num_mines, seed=seed_val)
        self.game.initialize_game()
        
        self.game_active = True
        self.clicks_made = 0
        self.current_multiplier = 1.0
        self.ai_playing = False
        
        # Initialize strategy - convert display name back to key
        display_name = self.strategy_var.get()
        # Find the key from display name
        strategy_key = None
        for key, val in self.strategy_map.items():
            if val == display_name:
                strategy_key = key
                break
        
        if not strategy_key:
            strategy_key = display_name  # Fallback
        
        try:
            self.current_strategy = self.registry.create_strategy(strategy_key)
            # Apply learning to the strategy
            self.apply_learning_to_strategy(self.current_strategy, strategy_key)
            # Apply user tuning
            self.apply_user_tuning(self.current_strategy)
        except Exception as e:
            print(f"Error creating strategy: {e}")
            self.current_strategy = None
        
        # Read bet amount and reset profit tracking for this game
        # If AI is playing, let the strategy determine bet size
        if self.ai_playing and self.current_strategy:
            self.bet_amount = self.calculate_ai_bet_size(strategy_key)
            self.bet_var.set(f"{self.bet_amount:.2f}")
            
            # Show AI bet reasoning
            win_history_list = list(self.win_history)
            recent_games = win_history_list[-10:] if len(win_history_list) > 0 else []
            win_rate = sum(recent_games) / len(recent_games) if recent_games else 0.5
            bankroll_pct = (self.bet_amount / self.current_balance * 100) if self.current_balance > 0 else 0
            self.ai_bet_info.config(text=f"🤖 AI: {bankroll_pct:.1f}% of bankroll (Win rate: {win_rate*100:.0f}%)")
        else:
            self.ai_bet_info.config(text="")
            try:
                value = float(self.bet_var.get())
                if value <= 0:
                    value = 10.0
                self.bet_amount = min(max(1.0, value), self.current_balance)
                self.bet_var.set(f"{self.bet_amount:.2f}")
            except Exception:
                self.bet_amount = 10.0
                self.bet_var.set("10.00")
        self.current_profit = 0.0
        self.bet_deducted = False
        # Clear final multiplier banner
        if hasattr(self, 'top_final_mult_label'):
            self.top_final_mult_label.config(text="")
        # Update fair-play hash
        try:
            fair_hash = self.compute_fair_hash()
            self.fair_hash_label.config(text=f"Hash: {fair_hash}")
        except Exception:
            self.fair_hash_label.config(text="Hash: —")
        
        # Deduct wager upfront from balance
        if self.current_balance < self.bet_amount:
            messagebox.showwarning("Insufficient Balance", "Your balance is too low for this bet amount.")
            self.start_button.config(state=tk.NORMAL)
            return
        self.current_balance -= self.bet_amount
        self.bet_deducted = True

        # Update displays
        self.update_stats_display(strategy_key)
        self.update_profit_display()
        
        # Update AI betting dashboard
        self.update_all_ai_bets()
        
        # Enable all cells
        for row in self.cells:
            for cell in row:
                cell.config(state=tk.NORMAL, text="", bg=self.CELL_COLOR)
        
        # Update UI based on mode
        play_mode = self.play_mode_var.get()
        self.start_button.config(state=tk.DISABLED)
        self.cashout_button.config(bg="#4caf50")  # Keep it green and enabled
        self.clicks_label.config(text="0")
        self.multiplier_label.config(text="1.00x")
        self.update_safe_cells_label()
        
        # Enable AI buttons if strategy available
        if self.current_strategy:
            self.hint_button.config(state=tk.NORMAL)
            self.auto_play_button.config(state=tk.NORMAL)
        
        # Auto-start AI if mode is AI
        if play_mode == "ai" and self.current_strategy:
            self.root.after(500, self.start_ai_play)
    
    def click_cell(self, row, col):
        """Handle cell click."""
        if not self.game_active:
            return
        
        # Click the cell
        success, result = self.game.click_cell(row, col)
        
        if not success:
            return  # Already clicked
        
        self.clicks_made += 1
        
        if result == "Mine":
            # Hit a mine - game over!
            self.reveal_mine(row, col)
            self.game_over(won=False)
        else:
            # Safe cell!
            self.reveal_safe(row, col)
            
            # Update multiplier (exponential based on probability)
            self.current_multiplier = self.calculate_multiplier()
            
            # Update current profit estimate
            self.current_profit = self.bet_amount * self.current_multiplier - self.bet_amount
            
            # Check if won (all safe cells revealed)
            if self.clicks_made >= self.game.max_clicks:
                self.game_over(won=True)
            else:
                # Update stats
                self.clicks_label.config(text=str(self.clicks_made))
                self.multiplier_label.config(text=f"{self.current_multiplier:.2f}x")
                self.update_safe_cells_label()
                self.update_profit_display()  # Show potential profit in real-time
    
    def calculate_multiplier(self):
        """
        Calculate multiplier based on probability of success.
        Uses exponential growth similar to real Mines games.
        
        Formula: Each safe click multiplies by (total_remaining / safe_remaining)
        This represents the risk taken at each step.
        """
        total_cells = self.grid_size * self.grid_size
        mines = self.num_mines
        
        multiplier = 1.0
        
        # Calculate multiplier for each click made
        for click in range(self.clicks_made):
            remaining_cells = total_cells - click
            remaining_safe = remaining_cells - mines
            
            if remaining_safe > 0:
                # Probability of hitting safe cell: safe / total
                # Odds multiplier: total / safe (inverted probability)
                click_multiplier = remaining_cells / remaining_safe
                multiplier *= click_multiplier
            else:
                break
        
        # Apply house edge (typically 1-4% in casino games)
        house_edge = 0.97  # 3% house edge
        multiplier *= house_edge
        
        return round(multiplier, 2)

    def compute_fair_hash(self):
        """Compute a fair-play hash from seed and mine positions."""
        import hashlib
        seed = self.game.seed if hasattr(self.game, 'seed') else None
        mines_sorted = sorted(list(self.game.mines)) if hasattr(self.game, 'mines') else []
        payload = f"seed={seed}|mines={mines_sorted}".encode('utf-8')
        return hashlib.sha256(payload).hexdigest()

    def update_strategy_description(self, *args):
        """Update the character description when strategy is selected."""
        display_name = self.strategy_var.get()
        # Find the strategy key from display name
        strategy_key = None
        for key, val in self.strategy_map.items():
            if val == display_name:
                strategy_key = key
                break

        if strategy_key and strategy_key in self.character_descriptions:
            self.strategy_desc.config(text=self.character_descriptions[strategy_key])
        else:
            self.strategy_desc.config(text="Select a strategy above")

    def reveal_safe(self, row, col):
        """Reveal a safe cell."""
        cell = self.cells[row][col]
        cell.config(text="💎", bg=self.SAFE_COLOR, state=tk.DISABLED)
    
    def reveal_mine(self, row, col):
        """Reveal a mine."""
        cell = self.cells[row][col]
        cell.config(text="💣", bg=self.MINE_COLOR, state=tk.DISABLED)
    
    def reveal_all_mines(self):
        """Reveal all mines on the board."""
        for r, c in self.game.mines:
            if not self.game.revealed[r][c]:
                cell = self.cells[r][c]
                cell.config(text="💣", bg=self.MINE_COLOR, state=tk.DISABLED)
    
    def game_over(self, won):
        """Handle game over."""
        self.game_active = False
        self.ai_playing = False  # Stop AI
        
        # Calculate profit/loss with upfront bet deduction model
        if won:
            # Credit full payout; bet was already deducted
            payout = self.bet_amount * self.current_multiplier
            profit = payout - self.bet_amount
            self.current_balance += payout
            self.current_profit = profit
        else:
            # Bet already deducted; just record loss
            profit = -self.bet_amount
            self.current_profit = profit
        
        # Update displays
        self.update_profit_display()
        # Show final multiplier on top bar
        if hasattr(self, 'top_final_mult_label'):
            self.top_final_mult_label.config(text=f"Final Multiplier: {self.current_multiplier:.2f}x")
        # Update AI betting dashboard after game ends
        self.update_all_ai_bets()
        # Update dashboard metrics
        try:
            self.metrics['games'] += 1
            self.metrics['sum_mult'] += float(self.current_multiplier)
            self.metrics['sum_bet'] += float(self.bet_amount)
            self.metrics['sum_profit'] += float(self.current_profit)
            avg_mult = (self.metrics['sum_mult'] / max(1, self.metrics['games']))
            roi = (self.metrics['sum_profit'] / max(1e-9, self.metrics['sum_bet'])) * 100.0
            self.dash_games.config(text=f"Games: {self.metrics['games']}")
            self.dash_avg_mult.config(text=f"Avg Mult: {avg_mult:.2f}x")
            self.dash_roi.config(text=f"ROI: {roi:.1f}%")
        except Exception:
            pass
        
        # Get strategy key and update stats
        display_name = self.strategy_var.get()
        strategy_key = None
        for key, val in self.strategy_map.items():
            if val == display_name:
                strategy_key = key
                break
        
        if strategy_key:
            self.update_strategy_stats(strategy_key, won, profit)
        
        # Disable all cells
        for row in self.cells:
            for cell in row:
                cell.config(state=tk.DISABLED)
        
        # Show all mines
        self.reveal_all_mines()
        
        # Show result message with profit info
        strategy_name = self.strategy_var.get()
        if won:
            message = f"🎉 VICTORY ACHIEVED!\n\n" \
                     f"🏆 Game Results:\n" \
                     f"   • Strategy: {strategy_name}\n" \
                     f"   • Safe Clicks: {self.clicks_made}\n" \
                     f"   • Final Multiplier: {self.current_multiplier:.2f}x\n" \
                     f"   • Profit Earned: ${profit:.2f}\n\n" \
                     f"💰 Account Update:\n" \
                     f"   • Balance: ${self.current_balance:.2f}\n\n" \
                     f"🎯 Excellent performance! Ready for the next challenge?"
            # Only show message box if not in auto-run mode
            if not self.autorun:
                messagebox.showinfo("🎉 VICTORY!", message)
        else:
            message = f"💥 GAME OVER - MINE STRUCK!\n\n" \
                     f"⚠️  Final Results:\n" \
                     f"   • Strategy: {strategy_name}\n" \
                     f"   • Safe Clicks Made: {self.clicks_made}\n" \
                     f"   • Loss Incurred: ${abs(profit):.2f}\n\n" \
                     f"🏦 Account Update:\n" \
                     f"   • Balance: ${self.current_balance:.2f}\n\n" \
                     f"💡 Tip: Consider cashing out earlier next time!"
            # Only show message box if not in auto-run mode
            if not self.autorun:
                messagebox.showwarning("💥 Game Over", message)

        # Session logging
        try:
            if self.logging_enabled.get():
                import os, json, time
                os.makedirs('logs', exist_ok=True)
                fair_hash = self.compute_fair_hash()
                entry = {
                    'ts': int(time.time()),
                    'seed': self.game.seed,
                    'board_size': self.game.board_size,
                    'mine_count': self.game.mine_count,
                    'mines': sorted(list(self.game.mines)),
                    'clicks': self.clicks_made,
                    'multiplier': round(self.current_multiplier, 2),
                    'bet': self.bet_amount,
                    'won': bool(won),
                    'profit': round(self.current_profit, 2),
                    'balance': round(self.current_balance, 2),
                    'fair_hash': fair_hash,
                }
                with open('logs/sessions.jsonl', 'a', encoding='utf-8') as f:
                    f.write(json.dumps(entry) + "\n")
        except Exception as e:
            print(f"log error: {e}")

        # Record in rolling win history and schedule next game if autorun
        try:
            self.win_history.append((time.time(), 1 if won else 0))
            self.update_time_winrate()
        except Exception:
            pass

        if self.autorun:
            delay = max(200, int(self.autorun_interval.get())) if hasattr(self, 'autorun_interval') else 1200
            self.autorun_after_id = self.root.after(delay, self.autorun_start_one)
        
        # Re-enable start button, disable AI buttons
        self.start_button.config(state=tk.NORMAL)
        # Keep cash out button always enabled
        self.hint_button.config(state=tk.DISABLED)
        self.auto_play_button.config(state=tk.DISABLED, text="🤖 AI Auto-Play")
    
    def cash_out(self):
        """Cash out and end the game (counts as a win)."""
        if not self.game_active:
            # Only show message box if not in auto-run mode
            if not self.autorun:
                messagebox.showinfo("No Active Game", "Please start a new game first to use the cash out feature.")
            return

        # Calculate payout and profit (bet was already deducted at start)
        payout = self.bet_amount * self.current_multiplier
        profit = payout - self.bet_amount
        new_balance = self.current_balance + payout

        message = f"💰 CASH OUT SUCCESSFUL!\n\n" \
                 f"📊 Game Summary:\n" \
                 f"   • Clicks Made: {self.clicks_made}\n" \
                 f"   • Final Multiplier: {self.current_multiplier:.2f}x\n" \
                 f"   • Bet Amount: ${self.bet_amount:.2f}\n" \
                 f"   • Payout: ${payout:.2f}\n" \
                 f"   • Profit Earned: ${profit:.2f}\n\n" \
                 f"🏦 Account Update:\n" \
                 f"   • Previous Balance: ${self.current_balance:.2f}\n" \
                 f"   • New Balance: ${new_balance:.2f}\n\n" \
                 f"✅ Your profit has been secured!"

        # Only show message box if not in auto-run mode
        if not self.autorun:
            messagebox.showinfo("💰 Cash Out Complete", message)

        # Mark as won for statistics (game_over will handle balance update)
        self.game_over(won=True)
    
    def get_valid_actions(self):
        """Get list of valid (unrevealed) cells."""
        actions = []
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if not self.game.revealed[r][c]:
                    actions.append((r, c))
        return actions
    
    def get_hint(self):
        """Get a hint from the AI strategy."""
        if not self.game_active or not self.current_strategy:
            return
        
        valid_actions = self.get_valid_actions()
        if not valid_actions:
            return
        
        # Get strategy's recommended action
        state = {
            'revealed': self.game.revealed.tolist() if hasattr(self.game.revealed, 'tolist') else self.game.revealed,
            'board': self.game.board.tolist() if hasattr(self.game.board, 'tolist') else self.game.board,
            'board_size': self.game.board_size,
            'mine_count': self.game.mine_count,
            'clicks_made': self.clicks_made
        }
        
        try:
            action = self.current_strategy.select_action(state, valid_actions)
            if action:
                row, col = action
                # Highlight the recommended cell
                self.cells[row][col].config(bg="#ffd700")  # Gold color for hint
                self.root.after(2000, lambda: self.reset_cell_color(row, col))
        except Exception as e:
            messagebox.showerror("Hint Error", f"Strategy error: {str(e)}")
    
    def reset_cell_color(self, row, col):
        """Reset cell color after hint."""
        if not self.game.revealed[row][col] and self.game_active:
            self.cells[row][col].config(bg=self.CELL_COLOR)
    
    def toggle_ai_play(self):
        """Toggle AI auto-play mode."""
        if self.ai_playing:
            # Stop AI
            self.ai_playing = False
            self.auto_play_button.config(text="🤖 AI Auto-Play")
        else:
            # Start AI
            self.ai_playing = True
            self.auto_play_button.config(text="⏸️ Pause AI")
            self.ai_make_move()

    # ===== Auto Run (multi-game) =====
    def toggle_auto_run(self):
        if self.autorun:
            self.autorun = False
            self.autorun_button.config(text="▶ START AUTO RUN", bg="#ff6f00")
            if self.autorun_after_id:
                try:
                    self.root.after_cancel(self.autorun_after_id)
                except Exception:
                    pass
                self.autorun_after_id = None
        else:
            self.autorun = True
            self.autorun_button.config(text="⏸️ STOP AUTO RUN", bg="#d32f2f")
            # Force AI mode and start
            self.play_mode_var.set("ai")
            if not self.game_active:
                self.autorun_start_one()

    def autorun_start_one(self):
        # Start a new AI game immediately
        if self.game_active:
            return
        self.play_mode_var.set("ai")
        self.start_game()
    
    def start_ai_play(self):
        """Start AI playing automatically."""
        self.ai_playing = True
        self.auto_play_button.config(text="⏸️ Pause AI")
        self.ai_make_move()
    
    def ai_make_move(self):
        """Make one move using the AI strategy."""
        if not self.ai_playing or not self.game_active or not self.current_strategy:
            return
        
        valid_actions = self.get_valid_actions()
        if not valid_actions:
            return
        
        # Check if AI should cash out based on risk/reward
        if self.should_ai_cash_out():
            self.cash_out()
            return
        
        # Get strategy's recommended action
        state = {
            'revealed': self.game.revealed.tolist() if hasattr(self.game.revealed, 'tolist') else self.game.revealed,
            'board': self.game.board.tolist() if hasattr(self.game.board, 'tolist') else self.game.board,
            'board_size': self.game.board_size,
            'mine_count': self.game.mine_count,
            'clicks_made': self.clicks_made
        }
        
        try:
            action = self.current_strategy.select_action(state, valid_actions)
            if action:
                row, col = action
                # Highlight cell briefly before clicking
                self.cells[row][col].config(bg="#ffd700")
                self.root.after(300, lambda: self.ai_click_cell(row, col))
        except Exception as e:
            if not self.autorun:
                messagebox.showerror("AI Error", f"Strategy error: {str(e)}")
            self.ai_playing = False
            self.auto_play_button.config(text="🤖 AI Auto-Play")
    
    def start_ai_bet_auto_refresh(self):
        """Start automatic refresh of AI betting dashboard."""
        self.update_all_ai_bets()
        # Schedule next update in 3 seconds
        self.root.after(3000, self.start_ai_bet_auto_refresh)
    
    def update_all_ai_bets(self):
        """Update the AI Betting Dashboard with current bet amounts for all strategies."""
        # Calculate what each AI would bet with current game state
        ai_strategies = ['takeshi', 'lelouch', 'kazuya', 'senku', 'okabe', 'hybrid']
        
        for strategy_key in ai_strategies:
            try:
                # Calculate bet for this strategy
                bet_amount = self.calculate_ai_bet_size(strategy_key)
                bankroll_pct = (bet_amount / self.current_balance * 100) if self.current_balance > 0 else 0
                
                # Update label with bet amount and percentage
                bet_text = f"${bet_amount:.2f} ({bankroll_pct:.1f}%)"
                
                # Add trend indicator
                if strategy_key in self.ai_bet_labels:
                    self.ai_bet_labels[strategy_key].config(text=bet_text)
            except Exception as e:
                if strategy_key in self.ai_bet_labels:
                    self.ai_bet_labels[strategy_key].config(text="—")
                # Silently handle errors during auto-refresh
    
    def calculate_ai_bet_size(self, strategy_name):
        """Calculate dynamic bet size based on character strategy and game state."""
        import random
        
        # Base bet (percentage of bankroll)
        base_bet = self.current_balance * 0.02  # 2% of bankroll as base
        base_bet = max(1.0, min(base_bet, self.current_balance))
        
        # Get recent performance (convert deque to list for slicing)
        try:
            win_history_list = list(self.win_history)
        except:
            win_history_list = []
        
        # Extract just the win/loss values (second element of each tuple)
        # win_history stores (timestamp, 1 or 0)
        recent_results = [w for _, w in win_history_list[-10:]] if len(win_history_list) > 0 else []
        win_rate = sum(recent_results) / len(recent_results) if recent_results else 0.5
        win_streak = 0
        loss_streak = 0
        
        # Calculate current streak
        if len(win_history_list) > 0:
            # Iterate through last 5 games, extracting win/loss value from tuple
            for timestamp, result in reversed(win_history_list[-5:]):
                if result == 1:
                    win_streak += 1
                else:
                    if win_streak > 0:
                        break  # Stop counting win streak
                    loss_streak += 1
        
        # Character-specific bet sizing
        strategy_lower = strategy_name.lower()
        
        if 'takeshi' in strategy_lower or 'aggressive' in strategy_lower:
            # TAKESHI KOVACS - Aggressive Berserker
            # Increases bets aggressively after wins, maintains high base
            bet_multiplier = 1.5  # Start with 50% more than base
            
            if win_streak >= 3:
                bet_multiplier *= 2.0  # Double down on winning streaks
            elif win_streak >= 1:
                bet_multiplier *= 1.3  # Increase on any win
            
            if loss_streak >= 2:
                bet_multiplier *= 0.7  # Reduce slightly on losses, but stay aggressive
            
            # Perceived advantage boost
            if win_rate > 0.6:
                bet_multiplier *= 1.4  # High confidence = bigger bets
            
            bet_size = base_bet * bet_multiplier
            
        elif 'lelouch' in strategy_lower:
            # LELOUCH VI BRITANNIA - Strategic Mastermind
            # Calculated, strategic bet sizing based on long-term planning
            bet_multiplier = 1.2  # Moderate base
            
            # Strategic phase-based betting
            if win_rate > 0.65:
                bet_multiplier *= 1.5  # Execute the master plan
            elif win_rate < 0.4:
                bet_multiplier *= 0.6  # Retreat and regroup
            
            # Avoid predictable patterns
            bet_multiplier *= random.uniform(0.9, 1.1)  # Add 10% variance
            
            bet_size = base_bet * bet_multiplier
            
        elif 'kazuya' in strategy_lower or 'conservative' in strategy_lower:
            # KAZUYA KINOSHITA - Conservative Survivor
            # Extremely risk-averse, small consistent bets
            bet_multiplier = 0.5  # Start with half of base (1% of bankroll)
            
            if loss_streak >= 1:
                bet_multiplier *= 0.5  # Cut bet in half after ANY loss
            
            if win_streak >= 5:
                bet_multiplier *= 1.2  # Only increase after long win streak
            
            # Never bet more than 1.5% of bankroll
            bet_size = min(base_bet * bet_multiplier, self.current_balance * 0.015)
            
        elif 'senku' in strategy_lower:
            # SENKU ISHIGAMI - Analytical Scientist
            # Data-driven, optimal Kelly Criterion-based betting
            # Kelly Criterion: f* = (bp - q) / b
            # Simplified: bet = bankroll * (win_rate - loss_rate)
            
            if win_rate > 0.5:
                kelly_fraction = (win_rate - (1 - win_rate)) * 0.5  # Half Kelly for safety
                bet_multiplier = max(0.5, min(2.0, kelly_fraction * 10))  # Scale and cap
            else:
                bet_multiplier = 0.5  # Minimum bet when EV is negative
            
            # Adjust for statistical confidence
            if len(recent_games) < 5:
                bet_multiplier *= 0.7  # Less confidence with small sample
            
            bet_size = base_bet * bet_multiplier
            
        elif 'okabe' in strategy_lower or 'rintaro' in strategy_lower:
            # RINTARO OKABE - Mad Scientist
            # Worldline manipulation, unpredictable but calculated
            bet_multiplier = 1.0
            
            # "Reading Steiner" - intuitive adjustments
            if random.random() < 0.3:  # 30% chance of "worldline shift"
                bet_multiplier *= random.choice([0.5, 1.5, 2.0])  # Dramatic changes
            
            # Lab member consensus (simulated)
            if win_rate > 0.6:
                bet_multiplier *= 1.4  # Consensus: increase
            elif win_rate < 0.4:
                bet_multiplier *= 0.6  # Consensus: decrease
            
            # Meta-game: avoid detection patterns
            bet_multiplier *= random.uniform(0.8, 1.2)
            
            bet_size = base_bet * bet_multiplier
            
        elif 'hybrid' in strategy_lower:
            # HYBRID STRATEGY - Ultimate Fusion
            # Blends Senku's analytical approach with Lelouch's strategic timing
            
            # Senku component: Kelly-based
            if win_rate > 0.5:
                kelly_component = (win_rate - (1 - win_rate)) * 0.5
            else:
                kelly_component = 0.5
            
            # Lelouch component: Strategic timing
            strategic_component = 1.0
            if win_rate > 0.65:
                strategic_component = 1.5
            elif win_rate < 0.4:
                strategic_component = 0.6
            
            # Blend based on game phase
            if len(recent_games) < 5:
                # Early: favor Senku (analytical)
                bet_multiplier = kelly_component * 5 * 0.7 + strategic_component * 0.3
            else:
                # Later: favor Lelouch (strategic)
                bet_multiplier = kelly_component * 5 * 0.4 + strategic_component * 0.6
            
            bet_size = base_bet * bet_multiplier
            
        else:
            # Default: moderate betting
            bet_multiplier = 1.0
            if win_streak >= 2:
                bet_multiplier *= 1.2
            if loss_streak >= 2:
                bet_multiplier *= 0.8
            bet_size = base_bet * bet_multiplier
        
        # Final constraints
        bet_size = max(1.0, min(bet_size, self.current_balance * 0.1))  # Never more than 10% of bankroll
        bet_size = min(bet_size, self.current_balance)  # Never more than available balance
        
        return round(bet_size, 2)
    
    def should_ai_cash_out(self):
        """Determine if AI should cash out based on risk/reward analysis."""
        import random
        
        # Don't cash out too early
        if self.clicks_made < 2:
            return False
        
        # Calculate current risk
        total_cells = self.grid_size * self.grid_size
        revealed_cells = self.clicks_made
        remaining_cells = total_cells - revealed_cells
        remaining_mines = self.num_mines  # Assume all mines still hidden (worst case)
        
        if remaining_cells == 0:
            return False
        
        # Risk = probability of hitting a mine on next click
        risk = remaining_mines / remaining_cells
        
        # Get current multiplier and profit
        multiplier = self.current_multiplier
        profit_ratio = (multiplier - 1.0)  # How much profit vs bet
        
        # Cash out conditions (more aggressive = higher thresholds)
        aggression = self.aggr_var.get()  # 0.30 to 0.70
        
        # Conservative: cash out at lower multipliers
        # Aggressive: push for higher multipliers
        target_multiplier = 1.5 + (aggression * 3.0)  # Range: 1.5x to 3.6x
        
        # Cash out if:
        # 1. We've reached target multiplier
        if multiplier >= target_multiplier:
            return True
        
        # 2. Risk is too high for current profit
        max_acceptable_risk = 0.15 + (aggression * 0.35)  # Range: 15% to 50%
        if risk > max_acceptable_risk and multiplier > 1.3:
            return True
        
        # 3. Random cash out with probability based on profit (simulate smart play)
        if multiplier > 1.5:
            cash_out_chance = (multiplier - 1.5) * 0.15  # Increases with multiplier
            if random.random() < cash_out_chance:
                return True
        
        return False
    
    def ai_click_cell(self, row, col):
        """AI clicks a cell."""
        self.click_cell(row, col)
        
        # Continue playing if still active
        if self.ai_playing and self.game_active:
            self.root.after(600, self.ai_make_move)
    
    def load_stats(self):
        """Load strategy statistics from file."""
        import json
        import os
        
        stats_file = 'strategy_stats.json'
        if os.path.exists(stats_file):
            try:
                with open(stats_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        
        # Default stats structure
        return {}
    
    def save_stats(self):
        """Save strategy statistics to file."""
        import json
        
        try:
            with open('strategy_stats.json', 'w') as f:
                json.dump(self.strategy_stats, f, indent=2)
        except Exception as e:
            print(f"Error saving stats: {e}")
    
    def get_strategy_stats(self, strategy_key):
        """Get stats for a specific strategy."""
        if strategy_key not in self.strategy_stats:
            self.strategy_stats[strategy_key] = {
                'wins': 0,
                'losses': 0,
                'total_profit': 0.0,
                'games_played': 0,
                'learning_data': {
                    'successful_moves': [],
                    'failed_moves': [],
                    'risk_adjustments': 0.0
                }
            }
        return self.strategy_stats[strategy_key]
    
    def update_strategy_stats(self, strategy_key, won, profit):
        """Update statistics for a strategy."""
        stats = self.get_strategy_stats(strategy_key)
        
        stats['games_played'] += 1
        if won:
            stats['wins'] += 1
        else:
            stats['losses'] += 1
        stats['total_profit'] += profit
        
        # Learning: adjust risk based on performance
        if stats['games_played'] >= 10:
            win_rate = stats['wins'] / stats['games_played']
            if win_rate < 0.3:
                stats['learning_data']['risk_adjustments'] -= 0.05  # Be more conservative
            elif win_rate > 0.7:
                stats['learning_data']['risk_adjustments'] += 0.05  # Be more aggressive
        
        self.save_stats()
        self.update_stats_display(strategy_key)
    
    def update_stats_display(self, strategy_key):
        """Update the statistics display for current strategy."""
        stats = self.get_strategy_stats(strategy_key)
        
        wins = stats['wins']
        losses = stats['losses']
        total_games = stats['games_played']
        win_rate = (wins / total_games * 100) if total_games > 0 else 0.0
        total_profit = stats['total_profit']
        
        self.wins_label.config(text=f"Wins: {wins}")
        self.losses_label.config(text=f"Losses: {losses}")
        self.winrate_label.config(text=f"Win Rate: {win_rate:.1f}%")
        
        profit_color = self.SAFE_COLOR if total_profit >= 0 else self.MINE_COLOR
        self.total_profit_label.config(text=f"Total Profit: ${total_profit:.2f}", fg=profit_color)
    
    def update_profit_display(self):
        """Update the current profit and balance display."""
        profit_color = self.SAFE_COLOR if self.current_profit >= 0 else self.MINE_COLOR
        self.profit_label.config(text=f"${self.current_profit:.2f}", fg=profit_color)
        
        balance_color = "#ffc107" if self.current_balance >= self.starting_balance else self.MINE_COLOR
        self.balance_label.config(text=f"${self.current_balance:.2f}", fg=balance_color)
        # Update top bar balance
        if hasattr(self, 'top_balance_label'):
            self.top_balance_label.config(text=f"Balance: ${self.current_balance:.2f}")

    def update_time_winrate(self):
        """Compute rolling win-rate over the selected time window and update button."""
        try:
            window_s = int(self.winrate_window.get()) if hasattr(self, 'winrate_window') else 60
        except Exception:
            window_s = 60
        now = time.time()
        # Drop old entries
        while self.win_history and now - self.win_history[0][0] > window_s:
            self.win_history.popleft()
        if not self.win_history:
            rate = 0.0
        else:
            wins = sum(w for _, w in self.win_history)
            rate = (wins / len(self.win_history)) * 100.0
        if hasattr(self, 'time_winrate_button'):
            self.time_winrate_button.config(text=f"📊 Win Rate (last {window_s}s): {rate:.1f}%")
        
    # Quick bet helpers
    def _bet_half(self):
        try:
            amt = max(1.0, float(self.bet_var.get()) / 2.0)
            self.bet_var.set(f"{amt:.2f}")
        except Exception:
            self.bet_var.set("10.00")
    
    def _bet_double(self):
        try:
            amt = min(self.current_balance, float(self.bet_var.get()) * 2.0)
            self.bet_var.set(f"{amt:.2f}")
        except Exception:
            self.bet_var.set("10.00")
    
    def _bet_max(self):
        self.bet_var.set(f"{self.current_balance:.2f}")
    
    def apply_learning_to_strategy(self, strategy, strategy_key):
        """Apply learned adjustments to a strategy instance."""
        stats = self.get_strategy_stats(strategy_key)
        learning_data = stats['learning_data']
        
        # Apply risk adjustments if the strategy supports it
        if hasattr(strategy, 'max_tolerable_risk'):
            # Adjust risk tolerance based on learning
            base_risk = 0.20  # Default 20%
            rate = self.evo_rate.get() if hasattr(self, 'evo_rate') else 1.0
            adjusted_risk = max(0.05, min(0.35, base_risk + learning_data['risk_adjustments'] * rate))
            strategy.max_tolerable_risk = adjusted_risk
        
        if hasattr(strategy, 'aggression_threshold'):
            # Adjust aggression based on learning
            base_aggression = 0.5
            rate = self.evo_rate.get() if hasattr(self, 'evo_rate') else 1.0
            adjusted_aggression = max(0.3, min(0.7, base_aggression + learning_data['risk_adjustments'] * rate))
            strategy.aggression_threshold = adjusted_aggression
        
        if hasattr(strategy, 'aggression_factor'):
            base_factor = 1.5
            rate = self.evo_rate.get() if hasattr(self, 'evo_rate') else 1.0
            adjusted_factor = max(1.0, min(2.0, base_factor + learning_data['risk_adjustments'] * rate))
            strategy.aggression_factor = adjusted_factor

    def apply_user_tuning(self, strategy):
        """Apply user slider tuning to the strategy instance."""
        try:
            aggr = self.aggr_var.get()
            maxrisk = self.maxrisk_var.get()
        except Exception:
            aggr = 0.5
            maxrisk = 0.20
        if hasattr(strategy, 'aggression_threshold'):
            strategy.aggression_threshold = aggr
        if hasattr(strategy, 'aggression_factor'):
            # Map aggr 0.3-0.7 to factor 1.0-2.0 linearly
            strategy.aggression_factor = 1.0 + (aggr - 0.3) * (1.0 / 0.4)
        if hasattr(strategy, 'max_tolerable_risk'):
            strategy.max_tolerable_risk = maxrisk


def main():
    """Main entry point for the game GUI."""
    root = tk.Tk()
    app = MinesGameGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

