"""Activation functions used in neural networks."""

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


def relu(x):
    return np.maximum(0.0, x)


def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)


def main() -> None:
    x = np.linspace(-6, 6, 400)

    for name, function in [
        ("Sigmoid", sigmoid),
        ("Tanh", tanh),
        ("ReLU", relu),
        ("Leaky ReLU", leaky_relu),
    ]:
        plt.figure(figsize=(8, 4))
        plt.plot(x, function(x))
        plt.axhline(0, linewidth=0.8)
        plt.axvline(0, linewidth=0.8)
        plt.title(name)
        plt.xlabel("Input")
        plt.ylabel("Activation")
        plt.grid(True)
        plt.show()

    print("Practical guidance:")
    print("- ReLU is a common default for hidden dense layers.")
    print("- Sigmoid is useful for binary-classification outputs.")
    print("- Tanh is zero-centered and historically common in recurrent models.")
    print("- Leaky ReLU reduces the risk of permanently inactive ReLU units.")


if __name__ == "__main__":
    main()
