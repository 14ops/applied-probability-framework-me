class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.wins = 0
        self.visits = 0

class MonteCarloTreeSearch:
    def search(self, initial_state):
        print("Performing MCTS.")
        return None
