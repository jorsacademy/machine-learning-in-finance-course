"""Introduction to neural networks using a small financial classification example."""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from tensorflow import keras


def make_dataset(n: int = 1500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    returns = pd.Series(rng.normal(0.0002, 0.012, n), name="return_1d")
    volume_change = rng.normal(0.0, 0.15, n)

    df = pd.DataFrame(
        {
            "return_1d": returns,
            "volume_change": volume_change,
            "volatility_20d": returns.rolling(20).std(),
            "momentum_5d": returns.rolling(5).sum(),
            "next_return": returns.shift(-1),
        }
    )
    df = df.dropna().copy()
    df["target"] = (df["next_return"] > 0).astype(int)
    return df.drop(columns="next_return").reset_index(drop=True)


def chronological_split(df: pd.DataFrame, train_fraction: float = 0.8):
    split = int(len(df) * train_fraction)
    return df.iloc[:split].copy(), df.iloc[split:].copy()


def main() -> None:
    df = make_dataset()
    train, test = chronological_split(df)

    features = ["return_1d", "volume_change", "volatility_20d", "momentum_5d"]
    X_train = train[features]
    y_train = train["target"]
    X_test = test[features]
    y_test = test["target"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(len(features),)),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(8, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )
    model.fit(
        X_train_scaled,
        y_train,
        validation_split=0.2,
        epochs=20,
        batch_size=32,
        shuffle=False,
        verbose=0,
    )

    probabilities = model.predict(X_test_scaled, verbose=0).ravel()
    predictions = (probabilities >= 0.5).astype(int)

    print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
    print(f"ROC AUC:  {roc_auc_score(y_test, probabilities):.3f}")
    print("\nModel architecture:")
    model.summary()


if __name__ == "__main__":
    main()
