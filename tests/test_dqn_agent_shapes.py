import numpy as np

from src.python.drl.dqn_agent import DQNAgent, DQNConfig


def test_agent_shapes_and_train_step():
    obs_dim = 66
    n_actions = 3
    agent = DQNAgent(obs_dim, n_actions, DQNConfig(device="cpu", batch_size=8))
    # Fill buffer with random transitions
    for _ in range(16):
        s = np.random.randn(obs_dim).astype(np.float32)
        a = np.random.randint(n_actions)
        r = float(np.random.randn())
        s2 = np.random.randn(obs_dim).astype(np.float32)
        d = bool(np.random.rand() < 0.1)
        agent.remember(s, a, r, s2, d)
    loss = agent.train_step()
    assert isinstance(loss, float)
    # action selection
    act = agent.act(np.random.randn(obs_dim).astype(np.float32))
    assert 0 <= act < n_actions

import numpy as np
from src.python.drl.dqn_agent import DQNAgent, DQNConfig


def test_dqn_forward_and_optimize():
    input_dim = 60 * 6
    n_actions = 3
    agent = DQNAgent(input_dim, n_actions, DQNConfig())
    s = np.zeros((input_dim,), dtype=np.float32)
    a = agent.act(s)
    assert 0 <= a < n_actions
    # push a few transitions and optimize
    for _ in range(100):
        ns = np.ones_like(s)
        agent.remember(s, a, 0.1, ns, False)
    loss = agent.optimize()
    assert isinstance(loss, float)

