"""Small Deep Q-Network example for portfolio allocation.

This is intentionally compact and pedagogical. It demonstrates replay memory,
epsilon-greedy exploration, a neural Q-function, and a target network.
"""

from collections import deque
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras

from portfolio_rl_env import TwoAssetRegimeEnv


def one_hot(state: int, n_states: int = 3) -> np.ndarray:
    x = np.zeros(n_states, dtype=np.float32)
    x[state] = 1.0
    return x


def build_q_network(n_states: int = 3, n_actions: int = 3) -> keras.Model:
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(n_states,)),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(n_actions),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return model


def train_dqn(
    episodes: int = 150,
    gamma: float = 0.95,
    batch_size: int = 64,
    replay_size: int = 5000,
    seed: int = 42,
):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

    env = TwoAssetRegimeEnv(n_steps=250, seed=seed)
    online = build_q_network()
    target = build_q_network()
    target.set_weights(online.get_weights())

    memory = deque(maxlen=replay_size)
    epsilon = 1.0
    min_epsilon = 0.05
    epsilon_decay = 0.97

    for episode in range(episodes):
        state = env.reset()
        while True:
            state_vec = one_hot(state)
            if np.random.random() < epsilon:
                action = np.random.randint(3)
            else:
                q_values = online.predict(state_vec[None, :], verbose=0)[0]
                action = int(np.argmax(q_values))

            result = env.step(action)
            memory.append((state, action, result.reward, result.state, result.done))

            if len(memory) >= batch_size:
                batch = random.sample(memory, batch_size)
                states = np.stack([one_hot(b[0]) for b in batch])
                next_states = np.stack([one_hot(b[3]) for b in batch])
                actions = np.array([b[1] for b in batch], dtype=int)
                rewards = np.array([b[2] for b in batch], dtype=np.float32)
                dones = np.array([b[4] for b in batch], dtype=np.float32)

                q_current = online.predict(states, verbose=0)
                q_next = target.predict(next_states, verbose=0)
                targets = q_current.copy()
                targets[np.arange(batch_size), actions] = rewards + (1.0 - dones) * gamma * np.max(q_next, axis=1)
                online.train_on_batch(states, targets)

            if result.done:
                break
            state = result.state

        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        if (episode + 1) % 10 == 0:
            target.set_weights(online.get_weights())

    return online, env.allocations


def evaluate(model: keras.Model, seed: int = 123) -> float:
    env = TwoAssetRegimeEnv(n_steps=1000, seed=seed)
    state = env.reset()
    wealth = 1.0

    while True:
        q_values = model.predict(one_hot(state)[None, :], verbose=0)[0]
        action = int(np.argmax(q_values))
        result = env.step(action)
        wealth *= 1.0 + result.reward
        if result.done:
            break
        state = result.state

    return wealth


def main() -> None:
    model, allocations = train_dqn()
    print("Learned risky-asset allocation by state:")
    for state in range(3):
        q_values = model.predict(one_hot(state)[None, :], verbose=0)[0]
        action = int(np.argmax(q_values))
        print(f"State {state}: {allocations[action]:.0%}")

    print(f"Out-of-sample terminal wealth in toy environment: {evaluate(model):.4f}")
    print("Do not interpret this toy result as evidence of a tradable strategy.")


if __name__ == "__main__":
    main()
