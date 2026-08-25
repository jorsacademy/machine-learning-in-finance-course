"""Logistic regression for next-day market direction prediction.

This example uses a chronological split and constructs the target from the
next-period return to avoid look-ahead leakage.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_dataset(n: int = 800) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    returns = pd.Series(rng.normal(0, 0.01, n), name="Return")
    df = pd.DataFrame({"Return": returns})
    df["Lag1"] = df["Return"].shift(1)
    df["Lag2"] = df["Return"].shift(2)
    df["Momentum5"] = df["Return"].rolling(5).sum()
    df["Volatility10"] = df["Return"].rolling(10).std()
    df["Target"] = (df["Return"].shift(-1) > 0).astype("Int64")
    df = df.dropna()
    return df[["Lag1", "Lag2", "Momentum5", "Volatility10"]], df["Target"].astype(int)


def main() -> None:
    X, y = build_dataset()
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(classification_report(y_test, predictions, digits=4))


if __name__ == "__main__":
    main()
