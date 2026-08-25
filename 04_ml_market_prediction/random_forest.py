"""Random Forest for next-day market direction prediction."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def build_dataset(n: int = 1000) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(7)
    returns = pd.Series(rng.normal(0, 0.01, n), name="Return")
    volume = pd.Series(rng.lognormal(mean=14, sigma=0.25, size=n), name="Volume")

    df = pd.DataFrame({"Return": returns, "Volume": volume})
    df["Lag1"] = df["Return"].shift(1)
    df["Lag2"] = df["Return"].shift(2)
    df["Momentum5"] = df["Return"].rolling(5).sum()
    df["Volatility10"] = df["Return"].rolling(10).std()
    df["VolumeChange"] = df["Volume"].pct_change()
    df["Target"] = (df["Return"].shift(-1) > 0).astype("Int64")
    df = df.dropna()

    features = ["Lag1", "Lag2", "Momentum5", "Volatility10", "VolumeChange"]
    return df[features], df["Target"].astype(int)


def main() -> None:
    X, y = build_dataset()
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(classification_report(y_test, predictions, digits=4))

    importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nFeature importance:\n", importance)


if __name__ == "__main__":
    main()
