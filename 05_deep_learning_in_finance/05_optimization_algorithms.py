"""Compare basic gradient-based optimizers on a small regression problem."""

import numpy as np
import tensorflow as tf
from tensorflow import keras


def make_data(n=1000, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4)).astype("float32")
    weights = np.array([0.8, -1.2, 0.5, 0.3], dtype="float32")
    y = (X @ weights + rng.normal(scale=0.5, size=n)).astype("float32")
    return X, y


def build_model():
    return keras.Sequential(
        [
            keras.layers.Input(shape=(4,)),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(1),
        ]
    )


def train_with_optimizer(name, optimizer, X, y):
    tf.keras.utils.set_random_seed(42)
    model = build_model()
    model.compile(optimizer=optimizer, loss="mse")
    history = model.fit(X, y, epochs=20, batch_size=32, verbose=0, shuffle=False)
    final_loss = history.history["loss"][-1]
    print(f"{name:10s} final loss: {final_loss:.4f}")


def main():
    X, y = make_data()

    optimizers = {
        "SGD": keras.optimizers.SGD(learning_rate=0.01),
        "Momentum": keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
        "RMSprop": keras.optimizers.RMSprop(learning_rate=0.001),
        "Adam": keras.optimizers.Adam(learning_rate=0.001),
    }

    for name, optimizer in optimizers.items():
        train_with_optimizer(name, optimizer, X, y)

    print("\nLearning rate is usually as important as optimizer choice.")
    print("Validation performance, not training loss alone, should drive selection.")


if __name__ == "__main__":
    main()
