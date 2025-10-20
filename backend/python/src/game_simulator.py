class GameSimulator:
    def __init__(self):
        self.game_state = {}

    def initialize_game(self):
        self.game_state = {"players": [], "turn": 0}

    def run_simulation(self, strategies):
        print("Running simulation...")
        for i in range(10):
            print(f"Turn {i+1}")
            for strategy in strategies:
                strategy.apply()
        print("Simulation finished.")
