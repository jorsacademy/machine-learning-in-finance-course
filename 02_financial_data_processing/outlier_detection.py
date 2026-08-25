"""Outlier detection and treatment for financial return data."""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def iqr_bounds(series: pd.Series, multiplier: float = 1.5) -> tuple[float, float]:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return q1 - multiplier * iqr, q3 + multiplier * iqr


def main() -> None:
    rng = np.random.default_rng(42)
    returns = pd.Series(rng.normal(0, 0.01, 250), name="Return")
    returns.iloc[[40, 120, 200]] = [0.12, -0.10, 0.15]

    z_scores = (returns - returns.mean()) / returns.std(ddof=1)
    z_outliers = returns[z_scores.abs() > 3]

    lower, upper = iqr_bounds(returns)
    iqr_outliers = returns[(returns < lower) | (returns > upper)]

    winsorized = returns.clip(lower=lower, upper=upper)

    isolation_forest = IsolationForest(contamination=0.02, random_state=42)
    flags = isolation_forest.fit_predict(returns.to_frame())
    model_outliers = returns[flags == -1]

    print("Z-score outliers:\n", z_outliers, "\n")
    print("IQR outliers:\n", iqr_outliers, "\n")
    print("Isolation Forest outliers:\n", model_outliers, "\n")
    print("Original vs winsorized extremes:")
    print(pd.DataFrame({"Original": returns, "Winsorized": winsorized}).iloc[[40, 120, 200]])


if __name__ == "__main__":
    main()
