"""Decision Tree regression for next-period return prediction."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor


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

    model = DecisionTreeRegressor(
        max_depth=4,
        min_samples_leaf=20,
        random_state=42,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    rmse = mean_squared_error(y_test, predictions) ** 0.5
    print(f"RMSE: {rmse:.6f}")

    importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nFeature importance:\n", importance)


if __name__ == "__main__":
    main()
