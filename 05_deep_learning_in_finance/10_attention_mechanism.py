"""Minimal self-attention example for financial sequences using Keras MultiHeadAttention."""

import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow import keras


def make_data(n_samples=1400, timesteps=20, n_features=4, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, timesteps, n_features)).astype("float32")

    # Construct a target that depends on selected recent positions.
    score = (
        0.8 * X[:, -1, 0]
        + 0.6 * X[:, -5, 1]
        - 0.5 * X[:, -10, 2]
        + 0.3 * X[:, -3:, 3].mean(axis=1)
    )
    y = (score > 0).astype(int)
    return X, y


def build_model(timesteps, n_features):
    inputs = keras.layers.Input(shape=(timesteps, n_features))

    x = keras.layers.Dense(16)(inputs)
    attention_output = keras.layers.MultiHeadAttention(
        num_heads=2,
        key_dim=8,
        dropout=0.1,
    )(x, x)

    x = keras.layers.Add()([x, attention_output])
    x = keras.layers.LayerNormalization()(x)
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dense(16, activation="relu")(x)
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)

    return keras.Model(inputs=inputs, outputs=outputs)


def main():
    X, y = make_data()
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = build_model(X.shape[1], X.shape[2])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(
        X_train,
        y_train,
        epochs=15,
        batch_size=32,
        validation_split=0.2,
        shuffle=False,
        verbose=0,
    )

    probabilities = model.predict(X_test, verbose=0).ravel()
    predictions = (probabilities >= 0.5).astype(int)
    print(f"Test accuracy: {accuracy_score(y_test, predictions):.3f}")
    print("Attention allows the model to weight information from different time steps dynamically.")


if __name__ == "__main__":
    main()
