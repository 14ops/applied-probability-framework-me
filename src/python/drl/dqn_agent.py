from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Deque, List
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class DQNConfig:
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    buffer_size: int = 50_000
    target_sync: int = 500
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 20_000
    device: str = "cpu"


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buf: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=capacity)

    def add(self, s: np.ndarray, a: int, r: float, s2: np.ndarray, d: bool) -> None:
        self.buf.append((s, a, r, s2, d))

    def __len__(self) -> int:
        return len(self.buf)

    def sample(self, batch_size: int):
        idx = np.random.choice(len(self.buf), size=batch_size, replace=False)
        s, a, r, s2, d = zip(*(self.buf[i] for i in idx))
        return (
            np.stack(s).astype(np.float32),
            np.array(a, dtype=np.int64),
            np.array(r, dtype=np.float32),
            np.stack(s2).astype(np.float32),
            np.array(d, dtype=np.float32),
        )


class DQNAgent:
    def __init__(self, obs_dim: int, n_actions: int, cfg: DQNConfig):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.q = QNetwork(obs_dim, n_actions).to(self.device)
        self.target = QNetwork(obs_dim, n_actions).to(self.device)
        self.target.load_state_dict(self.q.state_dict())
        self.opt = optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.criterion = nn.SmoothL1Loss()
        self.buffer = ReplayBuffer(cfg.buffer_size)
        self.epsilon = cfg.epsilon_start
        self.global_step = 0

    def act(self, obs: np.ndarray) -> int:
        self.global_step += 1
        eps = self._current_epsilon()
        if np.random.rand() < eps:
            return int(np.random.randint(self.n_actions))
        with torch.no_grad():
            x = torch.from_numpy(obs).float().to(self.device).unsqueeze(0)
            qvals = self.q(x)
            return int(torch.argmax(qvals, dim=1).item())

    def _current_epsilon(self) -> float:
        frac = min(1.0, self.global_step / max(1, self.cfg.epsilon_decay_steps))
        self.epsilon = self.cfg.epsilon_start + frac * (self.cfg.epsilon_end - self.cfg.epsilon_start)
        return self.epsilon

    def remember(self, s: np.ndarray, a: int, r: float, s2: np.ndarray, d: bool) -> None:
        self.buffer.add(s, a, r, s2, d)

    def train_step(self) -> float:
        if len(self.buffer) < self.cfg.batch_size:
            return 0.0
        s, a, r, s2, d = self.buffer.sample(self.cfg.batch_size)
        s_t = torch.from_numpy(s).float().to(self.device)
        a_t = torch.from_numpy(a).long().to(self.device)
        r_t = torch.from_numpy(r).float().to(self.device)
        s2_t = torch.from_numpy(s2).float().to(self.device)
        d_t = torch.from_numpy(d).float().to(self.device)
        # Q(s,a)
        q_sa = self.q(s_t).gather(1, a_t.view(-1, 1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target(s2_t).max(1)[0]
            target = r_t + (1.0 - d_t) * self.cfg.gamma * next_q
        loss = self.criterion(q_sa, target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), max_norm=5.0)
        self.opt.step()
        # Periodic target sync
        if self.global_step % self.cfg.target_sync == 0:
            self.target.load_state_dict(self.q.state_dict())
        return float(loss.item())

    def save(self, path: str) -> None:
        torch.save({"model": self.q.state_dict(), "cfg": self.cfg.__dict__}, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.q.load_state_dict(ckpt["model"])
        self.target.load_state_dict(self.q.state_dict())

from __future__ import annotations

from dataclasses import dataclass
from typing import Deque, Tuple
from collections import deque
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    def __init__(self, input_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class DQNConfig:
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    replay_size: int = 50_000
    target_sync: int = 500
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 10_000
    device: str = "cpu"


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=capacity)

    def push(self, s: np.ndarray, a: int, r: float, ns: np.ndarray, d: bool) -> None:
        self.buf.append((s, a, r, ns, d))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s, a, r, ns, d = map(np.array, zip(*batch))
        return s, a, r, ns, d

    def __len__(self) -> int:
        return len(self.buf)


class DQNAgent:
    def __init__(self, input_dim: int, n_actions: int, cfg: DQNConfig):
        self.n_actions = n_actions
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.q = QNetwork(input_dim, n_actions).to(self.device)
        self.target = QNetwork(input_dim, n_actions).to(self.device)
        self.target.load_state_dict(self.q.state_dict())
        self.optimizer = optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.criterion = nn.SmoothL1Loss()
        self.replay = ReplayBuffer(cfg.replay_size)
        self.total_steps = 0

    def epsilon(self) -> float:
        # linear decay
        eps = max(
            self.cfg.epsilon_end,
            self.cfg.epsilon_start
            - (self.cfg.epsilon_start - self.cfg.epsilon_end)
            * (self.total_steps / max(1, self.cfg.epsilon_decay_steps)),
        )
        return float(eps)

    def act(self, state: np.ndarray) -> int:
        self.total_steps += 1
        if random.random() < self.epsilon():
            return random.randrange(self.n_actions)
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            qv = self.q(s)
            return int(qv.argmax(dim=1).item())

    def remember(self, s, a, r, ns, d) -> None:
        self.replay.push(s, a, r, ns, d)

    def optimize(self) -> float:
        if len(self.replay) < self.cfg.batch_size:
            return 0.0
        s, a, r, ns, d = self.replay.sample(self.cfg.batch_size)
        s = torch.tensor(s, dtype=torch.float32, device=self.device)
        a = torch.tensor(a, dtype=torch.int64, device=self.device).unsqueeze(1)
        r = torch.tensor(r, dtype=torch.float32, device=self.device).unsqueeze(1)
        ns = torch.tensor(ns, dtype=torch.float32, device=self.device)
        d = torch.tensor(d, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_pred = self.q(s).gather(1, a)
        with torch.no_grad():
            q_next = self.target(ns).max(dim=1, keepdim=True)[0]
            q_target = r + (1.0 - d) * self.cfg.gamma * q_next
        loss = self.criterion(q_pred, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 1.0)
        self.optimizer.step()
        if self.total_steps % self.cfg.target_sync == 0:
            self.target.load_state_dict(self.q.state_dict())
        return float(loss.item())

    def save(self, path: str) -> None:
        torch.save({"model": self.q.state_dict(), "steps": self.total_steps}, path)

    def load(self, path: str, map_location: str | None = None) -> None:
        ckpt = torch.load(path, map_location=map_location or self.device)
        self.q.load_state_dict(ckpt["model"])
        self.target.load_state_dict(self.q.state_dict())
        self.total_steps = int(ckpt.get("steps", 0))


