import random
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class QNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim)
        )
    def forward(self, x): return self.net(x)

class ReplayBuffer:
    def __init__(self, cap=50000):
        self.s, self.a, self.r, self.ns, self.d = [], [], [], [], []
        self.cap = cap
    def add(self, s,a,r,ns,d):
        if len(self.s) >= self.cap:
            self.s.pop(0); self.a.pop(0); self.r.pop(0); self.ns.pop(0); self.d.pop(0)
        self.s.append(s); self.a.append(a); self.r.append(r); self.ns.append(ns); self.d.append(d)
    def sample(self, bs=256):
        idx = np.random.choice(len(self.s), size=min(bs, len(self.s)), replace=False)
        return (np.array([self.s[i] for i in idx], dtype=np.float32),
                np.array([self.a[i] for i in idx], dtype=np.int64),
                np.array([self.r[i] for i in idx], dtype=np.float32),
                np.array([self.ns[i] for i in idx], dtype=np.float32),
                np.array([self.d[i] for i in idx], dtype=np.float32))

class DQNAgent:
    def __init__(self, state_dim: int, action_dim: int, gamma=0.99, lr=1e-3, eps_start=1.0, eps_end=0.05, eps_decay=20000):
        self.q = QNet(state_dim, action_dim)
        self.t = QNet(state_dim, action_dim)
        self.t.load_state_dict(self.q.state_dict())
        self.opt = optim.Adam(self.q.parameters(), lr=lr)
        self.buf = ReplayBuffer()
        self.gamma = gamma
        self.step_i = 0
        self.eps_start, self.eps_end, self.eps_decay = eps_start, eps_end, eps_decay
        self.action_dim = action_dim

    def epsilon(self):
        return self.eps_end + (self.eps_start - self.eps_end) * np.exp(-self.step_i / self.eps_decay)

    def act(self, state, valid_mask=None):
        self.step_i += 1
        if random.random() < self.epsilon():
            if valid_mask is None: return random.randrange(self.action_dim)
            valid_idxs = [i for i,ok in enumerate(valid_mask) if ok]
            return random.choice(valid_idxs) if valid_idxs else 0
        with torch.no_grad():
            qv = self.q(torch.tensor(state, dtype=torch.float32).unsqueeze(0)).squeeze(0)
            if valid_mask is not None:
                vm = torch.tensor(valid_mask, dtype=torch.bool)
                qv[~vm] = -1e9
            return int(torch.argmax(qv).item())

    def learn(self, batch_size=256, target_update=500):
        if len(self.buf.s) < 1000: return
        s,a,r,ns,d = self.buf.sample(batch_size)
        s = torch.tensor(s); a = torch.tensor(a); r = torch.tensor(r); ns = torch.tensor(ns); d = torch.tensor(d)
        q_sa = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            max_next = self.t(ns).max(1)[0]
            tgt = r + self.gamma * (1.0 - d) * max_next
        loss = (q_sa - tgt).pow(2).mean()
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        if self.step_i % target_update == 0:
            self.t.load_state_dict(self.q.state_dict())
