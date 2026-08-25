"""Estimate simple linear factor exposures for multiple assets."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def simulate_factor_data(n_days: int = 1000, n_assets: int = 4, seed: int = 42):
    rng = np.random.default_rng(seed)
    market = rng.normal(0.0003, 0.01, n_days)
    size = rng.normal(0.0, 0.006, n_days)
    value = rng.normal(0.0, 0.005, n_days)
    factors = pd.DataFrame({"Market": market, "Size": size, "Value": value})

    assets = {}
    for i in range(n_assets):
        beta_m = 0.7 + 0.2 * i
        beta_s = -0.2 + 0.15 * i
        beta_v = 0.3 - 0.1 * i
        noise = rng.normal(0.0, 0.007, n_days)
        assets[f"Asset_{i+1}"] = 0.00005 + beta_m * market + beta_s * size + beta_v * value + noise
    return factors, pd.DataFrame(assets)


def estimate_exposures(factors: pd.DataFrame, asset_returns: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for asset in asset_returns.columns:
        model = LinearRegression().fit(factors, asset_returns[asset])
        rows.append(
            {
                "Asset": asset,
                "Alpha_daily": model.intercept_,
                "Market_beta": model.coef_[0],
                "Size_beta": model.coef_[1],
                "Value_beta": model.coef_[2],
                "R2": model.score(factors, asset_returns[asset]),
            }
        )
    return pd.DataFrame(rows).set_index("Asset")


def main() -> None:
    factors, asset_returns = simulate_factor_data()
    exposures = estimate_exposures(factors, asset_returns)
    print(exposures.round(4))
    print("\nFactor models explain systematic return exposures; residual risk remains asset-specific.")


if __name__ == "__main__":
    main()
