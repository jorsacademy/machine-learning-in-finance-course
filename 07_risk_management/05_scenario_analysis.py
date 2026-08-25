"""Scenario analysis using macro-factor sensitivities."""

import numpy as np
import pandas as pd


def build_exposures() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "equity_growth_beta": [1.2, 0.2, 0.5, 0.1],
            "rate_beta": [-0.1, -4.0, -1.5, 0.2],
            "inflation_beta": [-0.2, -0.3, -0.4, 1.3],
            "weight": [0.45, 0.30, 0.15, 0.10],
        },
        index=["Equity", "Government Bonds", "Credit", "Commodity"],
    )


def scenario_shocks() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "growth": [-0.08, -0.03, 0.05],
            "rates": [-0.01, 0.02, 0.005],
            "inflation": [-0.01, 0.04, 0.015],
        },
        index=["Recession", "Inflation Shock", "Soft Landing"],
    )


def estimate_asset_returns(exposures: pd.DataFrame, shock: pd.Series) -> pd.Series:
    return (
        exposures["equity_growth_beta"] * shock["growth"]
        + exposures["rate_beta"] * shock["rates"]
        + exposures["inflation_beta"] * shock["inflation"]
    )


def main() -> None:
    exposures = build_exposures()
    shocks = scenario_shocks()

    rows = []
    for name, shock in shocks.iterrows():
        asset_returns = estimate_asset_returns(exposures, shock)
        portfolio_return = float(np.dot(exposures["weight"], asset_returns))
        rows.append({"scenario": name, "portfolio_return": portfolio_return})

    results = pd.DataFrame(rows).set_index("scenario")
    formatted = results.map(lambda x: f"{x:.2%}")
    print(formatted)
    print("\nThis is a linear sensitivity framework; real scenario models can include nonlinear and path-dependent effects.")


if __name__ == "__main__":
    main()
