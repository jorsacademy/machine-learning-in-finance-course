"""Support Vector Regression for next-period return prediction."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def main() -> None:
    rng = np.random.default_rng(42)
    returns = pd.Series(rng.normal(0, 0.01, 800), name="Return")
    df = pd.DataFrame({"Return": returns})
    df["Lag1"] = df["Return"].shift(1)
    df["Lag2"] = df["Return"].shift(2)
    df["Momentum5"] = df["Return"].rolling(5).sum()
    df["Volatility10"] = df["Return"].rolling(10).std()
    df["Target"] = df["Return"].shift(-1)
    df = df.dropna()

    X = df[["Lag1", "Lag2", "Momentum5", "Volatility10"]]
    y = df["Target"]

    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", SVR(kernel="rbf", C=1.0, epsilon=0.001, gamma="scale")),
        ]
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    rmse = mean_squared_error(y_test, predictions) ** 0.5
    print(f"RMSE: {rmse:.6f}")


if __name__ == "__main__":
    main()
