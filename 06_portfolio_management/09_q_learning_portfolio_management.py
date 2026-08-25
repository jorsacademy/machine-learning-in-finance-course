"""Tabular Q-learning for a small portfolio allocation environment."""

import numpy as np

from portfolio_rl_env import TwoAssetRegimeEnv


def train_q_learning(
    episodes: int = 500,
    alpha: float = 0.1,
    gamma: float = 0.95,
    epsilon: float = 1.0,
    epsilon_decay: float = 0.99,
    min_epsilon: float = 0.05,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)
    env = TwoAssetRegimeEnv(n_steps=250, seed=seed)
    q_table = np.zeros((3, len(env.allocations)), dtype=float)

    for _ in range(episodes):
        state = env.reset()
        while True:
            if rng.random() < epsilon:
                action = int(rng.integers(len(env.allocations)))
            else:
                action = int(np.argmax(q_table[state]))

            result = env.step(action)
            td_target = result.reward
            if not result.done:
                td_target += gamma * np.max(q_table[result.state])
            q_table[state, action] += alpha * (td_target - q_table[state, action])

            if result.done:
                break
            state = result.state

        epsilon = max(min_epsilon, epsilon * epsilon_decay)

    return q_table, env.allocations


def evaluate_policy(q_table: np.ndarray, seed: int = 123) -> float:
    env = TwoAssetRegimeEnv(n_steps=1000, seed=seed)
    state = env.reset()
    wealth = 1.0

    while True:
        action = int(np.argmax(q_table[state]))
        result = env.step(action)
        wealth *= 1.0 + result.reward
        if result.done:
            break
        state = result.state

    return wealth


def main() -> None:
    q_table, allocations = train_q_learning()
    print("Q-table:\n", np.round(q_table, 6))
    print("\nLearned risky-asset allocation by state:")
    for state in range(q_table.shape[0]):
        action = int(np.argmax(q_table[state]))
        print(f"State {state}: {allocations[action]:.0%}")

    terminal_wealth = evaluate_policy(q_table)
    print(f"\nOut-of-sample terminal wealth in toy environment: {terminal_wealth:.4f}")


if __name__ == "__main__":
    main()
