"""Polynomial regression for non-linear financial relationships."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def main() -> None:
    rng = np.random.default_rng(42)
    n = 500
    x = np.linspace(-2, 2, n)
    y = 0.002 + 0.004 * x - 0.003 * x**2 + rng.normal(0, 0.002, n)

    X = pd.DataFrame({"Signal": x})
    target = pd.Series(y, name="Next_Return")

    split = int(n * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = target.iloc[:split], target.iloc[split:]

    model = Pipeline(
        [
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("scale", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    rmse = mean_squared_error(y_test, predictions) ** 0.5
    print(f"RMSE: {rmse:.6f}")


if __name__ == "__main__":
    main()
