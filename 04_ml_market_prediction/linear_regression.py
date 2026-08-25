"""Linear regression for next-period price-change prediction.

Predicting raw price levels can create deceptively strong metrics because price
levels are highly persistent. This example predicts the next-period return
instead, which is a more defensible teaching target for financial ML.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def main() -> None:
    rng = np.random.default_rng(42)
    returns = pd.Series(rng.normal(0, 0.01, 1000), name="Return")
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

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(f"MAE:  {mean_absolute_error(y_test, predictions):.6f}")
    print(f"RMSE: {mean_squared_error(y_test, predictions) ** 0.5:.6f}")
    print(f"R2:   {r2_score(y_test, predictions):.4f}")

    coefficients = pd.Series(model.coef_, index=X.columns, name="Coefficient")
    print("\nCoefficients:\n", coefficients)


if __name__ == "__main__":
    main()
