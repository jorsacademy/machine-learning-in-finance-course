"""Simple recurrent neural network for financial sequence classification."""

import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow import keras


def make_sequence_data(n_samples=1200, timesteps=15, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, timesteps, 2)).astype("float32")
    signal = X[:, -3:, 0].sum(axis=1) + 0.5 * X[:, -1, 1]
    y = (signal > 0).astype(int)
    return X, y


def main():
    X, y = make_sequence_data()
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = keras.Sequential(
        [
            keras.layers.Input(shape=X_train.shape[1:]),
            keras.layers.SimpleRNN(16, activation="tanh"),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=12, batch_size=32, validation_split=0.2, shuffle=False, verbose=0)

    probabilities = model.predict(X_test, verbose=0).ravel()
    predictions = (probabilities >= 0.5).astype(int)
    print(f"Test accuracy: {accuracy_score(y_test, predictions):.3f}")
    print("Simple RNNs are useful pedagogically but can struggle with long-range dependencies.")


if __name__ == "__main__":
    main()
