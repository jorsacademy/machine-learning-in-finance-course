"""Shared toy reinforcement-learning environment for portfolio examples."""

from dataclasses import dataclass
import numpy as np


@dataclass
class StepResult:
    state: int
    reward: float
    done: bool


class TwoAssetRegimeEnv:
    """Toy environment with three observable regimes and fixed allocation actions."""

    def __init__(self, n_steps: int = 500, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.n_steps = n_steps
        self.regimes = self.rng.choice([0, 1, 2], size=n_steps, p=[0.3, 0.4, 0.3])
        self.risky_returns = np.where(
            self.regimes == 0,
            self.rng.normal(0.0010, 0.010, n_steps),
            np.where(
                self.regimes == 1,
                self.rng.normal(0.0002, 0.008, n_steps),
                self.rng.normal(-0.0008, 0.012, n_steps),
            ),
        )
        self.cash_return = 0.00008
        self.allocations = np.array([0.0, 0.5, 1.0])
        self.t = 0

    def reset(self) -> int:
        self.t = 0
        return int(self.regimes[0])

    def step(self, action: int) -> StepResult:
        if action < 0 or action >= len(self.allocations):
            raise ValueError("Invalid action index.")
        risky_weight = float(self.allocations[action])
        reward = risky_weight * self.risky_returns[self.t] + (1.0 - risky_weight) * self.cash_return
        self.t += 1
        done = self.t >= self.n_steps
        next_state = int(self.regimes[self.t]) if not done else 0
        return StepResult(next_state, float(reward), done)
