"""Forward and backward propagation from first principles with NumPy."""

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    eps = 1e-12
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))


def main() -> None:
    rng = np.random.default_rng(42)

    X = rng.normal(size=(200, 3))
    true_weights = np.array([[1.2], [-0.8], [0.5]])
    logits = X @ true_weights + 0.2
    probabilities = sigmoid(logits)
    y = (rng.random((200, 1)) < probabilities).astype(float)

    W1 = rng.normal(scale=0.2, size=(3, 5))
    b1 = np.zeros((1, 5))
    W2 = rng.normal(scale=0.2, size=(5, 1))
    b2 = np.zeros((1, 1))

    learning_rate = 0.05

    for epoch in range(1000):
        # Forward propagation
        z1 = X @ W1 + b1
        a1 = np.maximum(0.0, z1)
        z2 = a1 @ W2 + b2
        y_hat = sigmoid(z2)

        # Backward propagation for sigmoid + binary cross-entropy
        m = X.shape[0]
        dz2 = (y_hat - y) / m
        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = dz2 @ W2.T
        dz1 = da1 * (z1 > 0)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1

        if epoch % 200 == 0:
            print(f"Epoch {epoch:4d} | Loss: {binary_cross_entropy(y, y_hat):.4f}")

    final_predictions = (y_hat >= 0.5).astype(int)
    accuracy = np.mean(final_predictions == y)
    print(f"Final accuracy: {accuracy:.3f}")


if __name__ == "__main__":
    main()
