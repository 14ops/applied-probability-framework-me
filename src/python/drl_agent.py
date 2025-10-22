import numpy as np


class DRLAgent:
    """A simple Deep Reinforcement Learning (DRL) agent using Q-learning.

    This agent learns to choose actions based on states and rewards, aiming to maximize
    cumulative reward over time. It uses an epsilon-greedy policy for exploration.
    """

    def __init__(self, state_size, action_size):
        """Initializes the DRLAgent with given state and action space sizes.

        Args:
            state_size (int): The number of possible states in the environment.
            action_size (int): The number of possible actions the agent can take.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = np.zeros((state_size, action_size))
        self.learning_rate = 0.1
        self.discount_factor = 0.99
        self.epsilon = 1.0  # Exploration-exploitation trade-off parameter
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

    def choose_action(self, state):
        """Chooses an action based on the current state using an epsilon-greedy policy.

        Args:
            state (int): The current state of the environment.

        Returns:
            int: The chosen action.
        """
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state, done):
        """Updates the Q-table based on the agent's experience.

        Args:
            state (int): The previous state.
            action (int): The action taken in the previous state.
            reward (float): The reward received after taking the action.
            next_state (int): The state after taking the action.
            done (bool): True if the episode has ended, False otherwise.
        """
        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state])

        new_value = (1 - self.learning_rate) * old_value + self.learning_rate * (
            reward + self.discount_factor * next_max
        )
        self.q_table[state, action] = new_value

        if done:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay


