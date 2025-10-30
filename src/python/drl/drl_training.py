from typing import Callable
import numpy as np
from .drl_environment import MinesEnv
from .drl_agent import DQNAgent

def train_dqn(episodes=2000, n_cells=25, log_every=100) -> DQNAgent:
    env = MinesEnv(n_cells=n_cells)
    state_dim = env.encode_state().shape[0]
    action_dim = env.n_cells + 1
    agent = DQNAgent(state_dim, action_dim)
    
    print(f"[DQN] Starting training with {episodes} episodes...", flush=True)
    
    for ep in range(episodes):
        s = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            valid = [False]*action_dim
            for a in env.valid_actions(): valid[a] = True
            a = agent.act(s, valid_mask=valid)
            res = env.step(a)
            agent.buf.add(s, a, res.reward, res.next_state, float(res.done))
            agent.learn()
            s = res.next_state; done = res.done; ep_reward += res.reward
        if (ep+1) % log_every == 0 or ep == 0 or ep == episodes - 1:
            print(f"[DQN] ep={ep+1} reward={ep_reward:.3f}", flush=True)
    
    print(f"[DQN] Training completed!", flush=True)
    return agent
