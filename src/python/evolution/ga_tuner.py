from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, Dict, List
import random

import pandas as pd

from src.python.validation.walkforward import evaluate_sma_strategy


@dataclass
class GAConfig:
    population: int = 20
    generations: int = 8
    crossover_rate: float = 0.6
    mutation_rate: float = 0.3
    fast_range: Tuple[int, int] = (5, 20)
    slow_range: Tuple[int, int] = (25, 100)


def random_individual(cfg: GAConfig) -> Tuple[int, int]:
    f = random.randint(*cfg.fast_range)
    s = random.randint(*cfg.slow_range)
    if f >= s:
        f = max(cfg.fast_range[0], s - 1)
    return f, s


def mutate(ind: Tuple[int, int], cfg: GAConfig) -> Tuple[int, int]:
    f, s = ind
    if random.random() < 0.5:
        f += random.choice([-2, -1, 1, 2])
        f = min(max(f, cfg.fast_range[0]), cfg.fast_range[1])
    else:
        s += random.choice([-4, -2, 2, 4])
        s = min(max(s, cfg.slow_range[0]), cfg.slow_range[1])
    if f >= s:
        f = max(cfg.fast_range[0], s - 1)
    return f, s


def crossover(a: Tuple[int, int], b: Tuple[int, int]) -> Tuple[int, int]:
    return (a[0], b[1]) if random.random() < 0.5 else (b[0], a[1])


def fitness(df: pd.DataFrame, ind: Tuple[int, int]) -> float:
    m = evaluate_sma_strategy(df, ind[0], ind[1])
    sharpe = m.get("sharpe", 0.0) or 0.0
    return float(sharpe)


def run_ga(df: pd.DataFrame, cfg: GAConfig = GAConfig()) -> Dict[str, float]:
    pop: List[Tuple[int, int]] = [random_individual(cfg) for _ in range(cfg.population)]
    best: Tuple[int, int] = pop[0]
    best_fit = -1e18
    for gen in range(cfg.generations):
        scored = [(ind, fitness(df, ind)) for ind in pop]
        scored.sort(key=lambda x: x[1], reverse=True)
        if scored[0][1] > best_fit:
            best, best_fit = scored[0]
        # selection (top 1/3)
        k = max(2, cfg.population // 3)
        parents = [ind for ind, _ in scored[:k]]
        # create next generation
        next_pop: List[Tuple[int, int]] = parents[:]
        while len(next_pop) < cfg.population:
            if random.random() < cfg.crossover_rate and len(parents) >= 2:
                a, b = random.sample(parents, 2)
                child = crossover(a, b)
            else:
                child = random.choice(parents)
            if random.random() < cfg.mutation_rate:
                child = mutate(child, cfg)
            next_pop.append(child)
        pop = next_pop
    return {"fast": best[0], "slow": best[1], "fitness_sharpe": best_fit}


