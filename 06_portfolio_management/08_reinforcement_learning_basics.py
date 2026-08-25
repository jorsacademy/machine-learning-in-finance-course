"""Minimal reinforcement-learning environment for portfolio allocation concepts."""

from dataclasses import dataclass
import numpy as np


@dataclass
class StepResult:
    state: int
    reward: float
    done: bool


class TwoAssetRegimeEnv:
    """Toy environment with bull, neutral, and bear regimes.

    Actions represent three fixed allocations between a risky asset and cash.
    The environment is intentionally small so students can inspect every state,
    action, reward, and transition.
    """

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
        return int(self.regimes[self.t])

    def step(self, action: int) -> StepResult:
        risky_weight = self.allocations[action]
        reward = risky_weight * self.risky_returns[self.t] + (1 - risky_weight) * self.cash_return
        self.t += 1
        done = self.t >= self.n_steps
        next_state = int(self.regimes[self.t]) if not done else 0
        return StepResult(next_state, float(reward), done)


def main() -> None:
    env = TwoAssetRegimeEnv(n_steps=20)
    state = env.reset()
    total_reward = 0.0

    while True:
        action = state  # Demonstration policy: cash in state 0, balanced in 1, risky in 2 is intentionally naive.
        action = min(action, len(env.allocations) - 1)
        result = env.step(action)
        total_reward += result.reward
        if result.done:
            break
        state = result.state

    print(f"Cumulative simple reward over the toy episode: {total_reward:.6f}")
    print("This environment is pedagogical and not a realistic trading simulator.")


if __name__ == "__main__":
    main()
