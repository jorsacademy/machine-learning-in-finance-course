"""Anomaly detection for financial risk monitoring using Isolation Forest."""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def make_transactions(n: int = 2500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    amount = rng.lognormal(mean=4.0, sigma=0.8, size=n)
    frequency_7d = rng.poisson(5, size=n)
    night_share = rng.beta(1.5, 8.0, size=n)
    cross_border_share = rng.beta(1.2, 10.0, size=n)

    anomaly_idx = rng.choice(n, size=35, replace=False)
    amount[anomaly_idx] *= rng.uniform(6, 15, size=len(anomaly_idx))
    frequency_7d[anomaly_idx] += rng.integers(15, 35, size=len(anomaly_idx))
    night_share[anomaly_idx] = np.clip(night_share[anomaly_idx] + 0.55, 0, 1)

    return pd.DataFrame(
        {
            "amount": amount,
            "frequency_7d": frequency_7d,
            "night_share": night_share,
            "cross_border_share": cross_border_share,
        }
    )


def main() -> None:
    df = make_transactions()
    features = df.columns.tolist()

    scaler = StandardScaler()
    X = scaler.fit_transform(df[features])

    model = IsolationForest(
        n_estimators=300,
        contamination=0.02,
        random_state=42,
    )
    df["anomaly_flag"] = model.fit_predict(X)
    df["anomaly_score"] = -model.score_samples(X)

    anomalies = df[df["anomaly_flag"] == -1].sort_values("anomaly_score", ascending=False)
    print(f"Detected anomalies: {len(anomalies)}")
    print(anomalies.head(10))
    print("\nAnomaly detection identifies unusual observations; it does not prove fraud or financial misconduct.")


if __name__ == "__main__":
    main()
