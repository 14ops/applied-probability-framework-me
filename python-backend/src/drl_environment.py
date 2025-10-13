class DRLEnvironment:
    def __init__(self):
        self.state = 0

    def reset(self):
        self.state = 0
        return self.state

    def step(self, action):
        # Dummy implementation
        reward = 1 if action == self.state % 2 else -1
        self.state += 1
        done = self.state > 10
        return self.state, reward, done, {}
