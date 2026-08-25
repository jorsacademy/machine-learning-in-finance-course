"""Use a 1D convolutional neural network for sequence-based return classification."""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from tensorflow import keras


def make_sequences(n=2500, lookback=20, seed=42):
    rng = np.random.default_rng(seed)
    returns = pd.Series(rng.normal(0.0002, 0.012, n))
    volume_change = pd.Series(rng.normal(0.0, 0.15, n))
    volatility = returns.rolling(10).std()

    frame = pd.DataFrame(
        {
            "return_1d": returns,
            "volume_change": volume_change,
            "volatility_10d": volatility,
            "next_return": returns.shift(-1),
        }
    ).dropna()

    feature_columns = ["return_1d", "volume_change", "volatility_10d"]
    values = frame[feature_columns].to_numpy()
    targets = (frame["next_return"].to_numpy() > 0).astype(int)

    X, y = [], []
    for end in range(lookback, len(frame)):
        X.append(values[end - lookback : end])
        y.append(targets[end])

    return np.asarray(X), np.asarray(y)


def main():
    X, y = make_sequences()
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler = StandardScaler()
    n_features = X_train.shape[-1]
    X_train_2d = X_train.reshape(-1, n_features)
    X_test_2d = X_test.reshape(-1, n_features)
    scaler.fit(X_train_2d)
    X_train = scaler.transform(X_train_2d).reshape(X_train.shape)
    X_test = scaler.transform(X_test_2d).reshape(X_test.shape)

    model = keras.Sequential(
        [
            keras.layers.Input(shape=X_train.shape[1:]),
            keras.layers.Conv1D(32, kernel_size=3, activation="relu"),
            keras.layers.GlobalAveragePooling1D(),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy")
    model.fit(X_train, y_train, epochs=15, batch_size=32, validation_split=0.2, shuffle=False, verbose=0)

    probabilities = model.predict(X_test, verbose=0).ravel()
    print(f"Test ROC AUC: {roc_auc_score(y_test, probabilities):.3f}")


if __name__ == "__main__":
    main()
