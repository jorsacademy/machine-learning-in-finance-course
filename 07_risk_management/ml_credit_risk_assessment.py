"""Machine learning for credit risk assessment.

This educational example emphasizes several practices that matter in credit risk:
- out-of-time validation rather than a purely random split;
- preprocessing learned only from training data;
- class weighting for imbalanced default labels;
- probability-based evaluation, not accuracy alone;
- probability calibration diagnostics;
- separation of model development from policy thresholds.

The data is synthetic so the example is fully reproducible and contains no
personal or protected-attribute information.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

RANDOM_STATE = 42


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def simulate_credit_data(n: int = 12000) -> pd.DataFrame:
    """Generate a reproducible synthetic credit-risk dataset."""
    rng = np.random.default_rng(RANDOM_STATE)

    application_date = pd.date_range("2019-01-01", periods=n, freq="6h")
    age = rng.integers(21, 70, n)
    income = rng.lognormal(mean=np.log(55000), sigma=0.45, size=n)
    total_debt = np.maximum(0, income * rng.uniform(0.05, 1.25, n))
    loan_amount = rng.lognormal(mean=np.log(14000), sigma=0.55, size=n)
    credit_score = np.clip(rng.normal(680, 75, n), 300, 850)
    employment_years = np.clip(rng.normal(7, 5, n), 0, 40)
    late_payments_12m = rng.poisson(0.6, n)
    utilization = np.clip(rng.beta(2.0, 3.2, n), 0, 1)
    loan_purpose = rng.choice(
        ["debt_consolidation", "home_improvement", "auto", "education", "other"],
        size=n,
        p=[0.38, 0.18, 0.16, 0.10, 0.18],
    )

    debt_to_income = total_debt / income
    loan_to_income = loan_amount / income

    # Later applications are slightly riskier to create a mild temporal shift.
    time_effect = np.linspace(-0.15, 0.20, n)
    logit = (
        -3.3
        + 3.0 * debt_to_income
        + 1.6 * loan_to_income
        + 2.0 * utilization
        + 0.38 * late_payments_12m
        - 0.009 * (credit_score - 650)
        - 0.025 * employment_years
        + time_effect
    )
    probability_default = sigmoid(logit)
    default = rng.binomial(1, probability_default)

    data = pd.DataFrame(
        {
            "application_date": application_date,
            "age": age,
            "income": income,
            "total_debt": total_debt,
            "loan_amount": loan_amount,
            "credit_score": credit_score,
            "employment_years": employment_years,
            "late_payments_12m": late_payments_12m,
            "utilization": utilization,
            "loan_purpose": loan_purpose,
            "default": default,
        }
    )

    # Insert missing values to demonstrate train-only imputation.
    for column in ["income", "employment_years", "credit_score"]:
        mask = rng.random(n) < 0.02
        data.loc[mask, column] = np.nan

    return data


def add_engineered_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create economically interpretable credit-risk features."""
    result = data.copy()
    safe_income = result["income"].replace(0, np.nan)
    result["debt_to_income"] = result["total_debt"] / safe_income
    result["loan_to_income"] = result["loan_amount"] / safe_income
    return result


def out_of_time_split(data: pd.DataFrame, train_fraction: float = 0.80):
    """Split chronologically to approximate real deployment evaluation."""
    data = data.sort_values("application_date").reset_index(drop=True)
    split = int(len(data) * train_fraction)
    train = data.iloc[:split].copy()
    test = data.iloc[split:].copy()
    return train, test


def build_model(numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    """Build a leakage-safe preprocessing and Random Forest pipeline."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )

    classifier = RandomForestClassifier(
        n_estimators=400,
        max_depth=8,
        min_samples_leaf=20,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", classifier),
        ]
    )


def evaluate_credit_model(y_true: pd.Series, probabilities: np.ndarray, threshold: float = 0.50) -> None:
    """Evaluate ranking, calibration, and threshold-based classification."""
    predictions = (probabilities >= threshold).astype(int)

    print(f"Default rate: {y_true.mean():.3f}")
    print(f"ROC AUC: {roc_auc_score(y_true, probabilities):.3f}")
    print(f"PR AUC: {average_precision_score(y_true, probabilities):.3f}")
    print(f"Brier score: {brier_score_loss(y_true, probabilities):.4f}")
    print(f"Decision threshold: {threshold:.2f}")
    print("\nConfusion matrix:\n", confusion_matrix(y_true, predictions))
    print("\nClassification report:\n", classification_report(y_true, predictions, digits=3))


def main() -> None:
    data = add_engineered_features(simulate_credit_data())
    train, test = out_of_time_split(data)

    numeric_features = [
        "age",
        "income",
        "total_debt",
        "loan_amount",
        "credit_score",
        "employment_years",
        "late_payments_12m",
        "utilization",
        "debt_to_income",
        "loan_to_income",
    ]
    categorical_features = ["loan_purpose"]
    feature_columns = numeric_features + categorical_features

    model = build_model(numeric_features, categorical_features)
    model.fit(train[feature_columns], train["default"])

    probabilities = model.predict_proba(test[feature_columns])[:, 1]
    evaluate_credit_model(test["default"], probabilities, threshold=0.50)

    # Threshold selection is a business-policy decision. A lender may choose a
    # lower threshold when missing a likely default is more costly than a false
    # positive, but that choice should be validated against explicit costs,
    # capacity constraints, regulation, and fairness requirements.


if __name__ == "__main__":
    main()
