"""LSTM example for next-period return direction classification."""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from tensorflow import keras


def build_dataset(n=3000, lookback=30, seed=42):
    rng = np.random.default_rng(seed)
    returns = pd.Series(rng.normal(0.0002, 0.012, n))
    volume_change = pd.Series(rng.normal(0.0, 0.2, n))

    frame = pd.DataFrame(
        {
            "return_1d": returns,
            "momentum_5d": returns.rolling(5).sum(),
            "volatility_20d": returns.rolling(20).std(),
            "volume_change": volume_change,
            "next_return": returns.shift(-1),
        }
    ).dropna()

    features = ["return_1d", "momentum_5d", "volatility_20d", "volume_change"]
    values = frame[features].to_numpy()
    target = (frame["next_return"].to_numpy() > 0).astype(int)

    X, y = [], []
    for end in range(lookback, len(frame)):
        X.append(values[end - lookback : end])
        y.append(target[end])
    return np.asarray(X), np.asarray(y)


def scale_sequences(X_train, X_test):
    scaler = StandardScaler()
    n_features = X_train.shape[-1]
    train_2d = X_train.reshape(-1, n_features)
    test_2d = X_test.reshape(-1, n_features)
    scaler.fit(train_2d)
    return (
        scaler.transform(train_2d).reshape(X_train.shape),
        scaler.transform(test_2d).reshape(X_test.shape),
    )


def main():
    X, y = build_dataset()
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    X_train, X_test = scale_sequences(X_train, X_test)

    model = keras.Sequential(
        [
            keras.layers.Input(shape=X_train.shape[1:]),
            keras.layers.LSTM(32, dropout=0.1),
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
