"""Minimal reinforcement-learning walkthrough for portfolio allocation concepts."""

from portfolio_rl_env import TwoAssetRegimeEnv


def main() -> None:
    env = TwoAssetRegimeEnv(n_steps=20)
    state = env.reset()
    total_reward = 0.0

    while True:
        # Simple hand-written policy for demonstration only.
        # State 0 is favorable, state 1 is neutral, state 2 is unfavorable.
        action_by_state = {0: 2, 1: 1, 2: 0}
        action = action_by_state[state]
        result = env.step(action)
        total_reward += result.reward
        if result.done:
            break
        state = result.state

    print(f"Cumulative simple reward over the toy episode: {total_reward:.6f}")
    print("This environment is pedagogical and not a realistic trading simulator.")


if __name__ == "__main__":
    main()
