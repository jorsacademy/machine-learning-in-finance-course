"""Ensemble methods for financial risk scoring on tabular data."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def make_dataset(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    income = rng.lognormal(10.6, 0.5, n)
    balance = rng.lognormal(9.5, 0.8, n)
    utilization = rng.beta(2.0, 4.0, n)
    late_payments = rng.poisson(0.6, n)
    account_age = rng.gamma(4.0, 2.0, n)

    score = -3.6 + 2.0 * (balance / income) + 2.2 * utilization + 0.5 * late_payments - 0.05 * account_age
    probability = 1.0 / (1.0 + np.exp(-score))
    default = rng.binomial(1, np.clip(probability, 0.001, 0.999))

    return pd.DataFrame(
        {
            "income": income,
            "balance": balance,
            "utilization": utilization,
            "late_payments": late_payments,
            "account_age": account_age,
            "default": default,
        }
    )


def main() -> None:
    df = make_dataset()
    split = int(len(df) * 0.8)
    train, test = df.iloc[:split], df.iloc[split:]
    features = ["income", "balance", "utilization", "late_payments", "account_age"]

    logistic = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    random_forest = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=8,
        class_weight="balanced",
        random_state=42,
    )
    gradient_boosting = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, random_state=42)

    ensemble = VotingClassifier(
        estimators=[("lr", logistic), ("rf", random_forest), ("hgb", gradient_boosting)],
        voting="soft",
    )
    ensemble.fit(train[features], train["default"])
    probability = ensemble.predict_proba(test[features])[:, 1]

    print(f"Soft-voting ensemble ROC AUC: {roc_auc_score(test['default'], probability):.3f}")
    print("Ensembling can improve robustness, but it does not remove the need for calibration, validation, and governance.")


if __name__ == "__main__":
    main()
