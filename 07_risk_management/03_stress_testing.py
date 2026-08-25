"""Simple portfolio stress testing with deterministic market shocks."""

import numpy as np
import pandas as pd


def build_portfolio() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "asset": ["Equity", "Government Bonds", "Credit", "Gold"],
            "weight": [0.50, 0.25, 0.15, 0.10],
        }
    ).set_index("asset")


def scenarios() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Equity": [-0.30, -0.15, 0.10],
            "Government Bonds": [0.08, -0.10, -0.04],
            "Credit": [-0.12, -0.08, 0.04],
            "Gold": [0.12, 0.05, -0.08],
        },
        index=["Equity Crash", "Rates Shock", "Risk-On Rally"],
    )


def portfolio_scenario_return(weights: pd.Series, shock: pd.Series) -> float:
    aligned = shock.reindex(weights.index)
    return float(np.dot(weights.to_numpy(), aligned.to_numpy()))


def main() -> None:
    portfolio = build_portfolio()
    stress = scenarios()
    weights = portfolio["weight"]

    results = {}
    for scenario_name, shocks in stress.iterrows():
        results[scenario_name] = portfolio_scenario_return(weights, shocks)

    result_series = pd.Series(results, name="portfolio_return")
    print("Portfolio weights:\n", portfolio)
    print("\nScenario results:\n", result_series.apply(lambda x: f"{x:.2%}"))
    print("\nStress tests are conditional scenarios, not probabilistic forecasts.")


if __name__ == "__main__":
    main()
