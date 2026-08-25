"""Learn long-only portfolio weights from market features with a small neural network."""

import numpy as np
import tensorflow as tf
from tensorflow import keras


def make_windows(n_samples: int = 1800, n_assets: int = 4, lookback: int = 20, seed: int = 42):
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.00025, 0.012, size=(n_samples, n_assets))
    X, y = [], []
    for t in range(lookback, n_samples - 1):
        window = returns[t - lookback:t]
        features = np.concatenate([window.mean(axis=0), window.std(axis=0, ddof=1)])
        X.append(features)
        y.append(returns[t + 1])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


def build_model(n_features: int, n_assets: int) -> keras.Model:
    inputs = keras.Input(shape=(n_features,))
    x = keras.layers.Dense(32, activation="relu")(inputs)
    x = keras.layers.Dense(16, activation="relu")(x)
    logits = keras.layers.Dense(n_assets)(x)
    weights = keras.layers.Softmax(name="portfolio_weights")(logits)
    return keras.Model(inputs, weights)


def portfolio_loss(realized_returns: tf.Tensor, weights: tf.Tensor) -> tf.Tensor:
    portfolio_return = tf.reduce_sum(weights * realized_returns, axis=1)
    mean_return = tf.reduce_mean(portfolio_return)
    volatility = tf.math.reduce_std(portfolio_return) + 1e-6
    return -(mean_return / volatility)


def main() -> None:
    tf.keras.utils.set_random_seed(42)
    X, y = make_windows()
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + 1e-8
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    model = build_model(X_train.shape[1], y_train.shape[1])
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=portfolio_loss)
    model.fit(X_train, y_train, validation_split=0.2, epochs=20, batch_size=64, verbose=0)

    weights = model.predict(X_test, verbose=0)
    realized = np.sum(weights * y_test, axis=1)
    annual_return = realized.mean() * 252
    annual_vol = realized.std(ddof=1) * np.sqrt(252)

    print("Average learned weights:", np.round(weights.mean(axis=0), 4))
    print(f"Annualized return: {annual_return:.4f}")
    print(f"Annualized volatility: {annual_vol:.4f}")
    print(f"Ex-post Sharpe-like ratio: {annual_return / annual_vol:.4f}")


if __name__ == "__main__":
    main()
