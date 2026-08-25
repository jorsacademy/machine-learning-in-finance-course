"""Common loss functions for financial machine-learning tasks."""

import numpy as np


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def binary_cross_entropy(y_true, y_pred):
    eps = 1e-12
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def huber(y_true, y_pred, delta=1.0):
    error = y_true - y_pred
    absolute_error = np.abs(error)
    quadratic = np.minimum(absolute_error, delta)
    linear = absolute_error - quadratic
    return np.mean(0.5 * quadratic**2 + delta * linear)


def main() -> None:
    actual_returns = np.array([0.010, -0.015, 0.004, 0.080, -0.006])
    predicted_returns = np.array([0.008, -0.010, 0.006, 0.020, -0.004])

    print(f"MSE:   {mse(actual_returns, predicted_returns):.6f}")
    print(f"MAE:   {mae(actual_returns, predicted_returns):.6f}")
    print(f"Huber: {huber(actual_returns, predicted_returns, delta=0.02):.6f}")

    actual_direction = np.array([1, 0, 1, 1, 0])
    predicted_probability = np.array([0.72, 0.31, 0.64, 0.55, 0.18])
    print(f"Binary cross-entropy: {binary_cross_entropy(actual_direction, predicted_probability):.6f}")

    print("\nInterpretation:")
    print("- MSE penalizes large errors strongly and is sensitive to outliers.")
    print("- MAE is more robust to extreme observations.")
    print("- Huber loss combines quadratic and linear behavior.")
    print("- Binary cross-entropy is appropriate for probabilistic binary targets.")


if __name__ == "__main__":
    main()
