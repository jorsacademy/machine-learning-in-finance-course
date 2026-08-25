"""Neural-network classification example for financial risk prediction."""

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from tensorflow import keras


def make_dataset(n: int = 4000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    income = rng.lognormal(mean=10.7, sigma=0.45, size=n)
    debt = rng.lognormal(mean=9.8, sigma=0.65, size=n)
    utilization = np.clip(rng.beta(2.2, 4.5, size=n), 0, 1)
    delinquencies = rng.poisson(0.45, size=n)
    age = rng.integers(21, 75, size=n)

    debt_to_income = debt / income
    logit = -3.4 + 2.3 * debt_to_income + 2.0 * utilization + 0.55 * delinquencies - 0.012 * age
    probability = 1.0 / (1.0 + np.exp(-logit))
    default = rng.binomial(1, np.clip(probability, 0.001, 0.999))

    return pd.DataFrame(
        {
            "income": income,
            "debt": debt,
            "utilization": utilization,
            "delinquencies": delinquencies,
            "age": age,
            "debt_to_income": debt_to_income,
            "default": default,
        }
    )


def chronological_split(df: pd.DataFrame, train_fraction: float = 0.8):
    split = int(len(df) * train_fraction)
    return df.iloc[:split].copy(), df.iloc[split:].copy()


def main() -> None:
    df = make_dataset()
    train, test = chronological_split(df)
    features = ["income", "debt", "utilization", "delinquencies", "age", "debt_to_income"]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[features])
    X_test = scaler.transform(test[features])
    y_train = train["default"].to_numpy()
    y_test = test["default"].to_numpy()

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(len(features),)),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dropout(0.20),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[keras.metrics.AUC(name="auc")])
    model.fit(X_train, y_train, validation_split=0.2, epochs=20, batch_size=64, verbose=0)

    probability = model.predict(X_test, verbose=0).ravel()
    prediction = (probability >= 0.5).astype(int)

    print(f"ROC AUC: {roc_auc_score(y_test, probability):.3f}")
    print(classification_report(y_test, prediction, digits=3))
    print("Probability calibration and fairness analysis are required before any real credit decision use.")


if __name__ == "__main__":
    main()
