"""A simple residual-based statistical arbitrage example."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def simulate_panel(n: int = 1200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    market = rng.normal(0.0002, 0.01, n)
    sector = rng.normal(0.0001, 0.006, n)
    data = {}
    for i in range(6):
        beta_m = rng.uniform(0.7, 1.3)
        beta_s = rng.uniform(0.2, 0.9)
        idio = rng.normal(0.0, 0.005, n)
        data[f"Asset_{i+1}"] = beta_m * market + beta_s * sector + idio
    data["Market"] = market
    data["Sector"] = sector
    return pd.DataFrame(data, index=pd.date_range("2020-01-01", periods=n, freq="B"))


def residual_signals(df: pd.DataFrame, train_fraction: float = 0.6) -> tuple[pd.DataFrame, pd.DataFrame]:
    split = int(len(df) * train_fraction)
    assets = [c for c in df.columns if c.startswith("Asset_")]
    factors = ["Market", "Sector"]

    residuals = pd.DataFrame(index=df.index, columns=assets, dtype=float)
    for asset in assets:
        model = LinearRegression().fit(df.iloc[:split][factors], df.iloc[:split][asset])
        residuals[asset] = df[asset] - model.predict(df[factors])

    mean = residuals.iloc[:split].mean()
    std = residuals.iloc[:split].std(ddof=1)
    z = (residuals - mean) / std
    return residuals, z


def main() -> None:
    df = simulate_panel()
    _, z = residual_signals(df)
    asset_cols = [c for c in df.columns if c.startswith("Asset_")]

    target = pd.DataFrame(0.0, index=df.index, columns=asset_cols)
    target[z < -2.0] = 1.0
    target[z > 2.0] = -1.0

    # Normalize gross exposure to 1 when active.
    gross = target.abs().sum(axis=1).replace(0, np.nan)
    target = target.div(gross, axis=0).fillna(0.0)
    executed = target.shift(1).fillna(0.0)
    turnover = executed.diff().abs().sum(axis=1).fillna(0.0)
    strategy = (executed * df[asset_cols]).sum(axis=1) - turnover * 3 / 10_000

    print(f"Residual stat-arb total return: {(1 + strategy).prod() - 1:.2%}")
    print("Factor coefficients are estimated only on the training window.")


if __name__ == "__main__":
    main()
