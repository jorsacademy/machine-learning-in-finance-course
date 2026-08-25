"""Ridge and Lasso regression for next-period return prediction."""

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_dataset(n: int = 1000) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    returns = pd.Series(rng.normal(0, 0.01, n), name="Return")
    df = pd.DataFrame({"Return": returns})
    for lag in range(1, 6):
        df[f"Lag{lag}"] = df["Return"].shift(lag)
    df["Momentum5"] = df["Return"].rolling(5).sum()
    df["Volatility10"] = df["Return"].rolling(10).std()
    df["Target"] = df["Return"].shift(-1)
    df = df.dropna()
    X = df.drop(columns=["Return", "Target"])
    return X, df["Target"]


def evaluate(model: Pipeline, X_train, X_test, y_train, y_test, name: str) -> None:
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, pred) ** 0.5
    print(f"{name} RMSE: {rmse:.6f}")


def main() -> None:
    X, y = build_dataset()
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    ridge = Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=1.0))])
    lasso = Pipeline([("scale", StandardScaler()), ("model", Lasso(alpha=0.0001, max_iter=10000))])

    evaluate(ridge, X_train, X_test, y_train, y_test, "Ridge")
    evaluate(lasso, X_train, X_test, y_train, y_test, "Lasso")


if __name__ == "__main__":
    main()
