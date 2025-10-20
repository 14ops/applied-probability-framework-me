class MultiAgentCore:
    def __init__(self, num_agents):
        self.agents = [f"Agent_{i}" for i in range(num_agents)]

    def get_agents(self):
        return self.agents
