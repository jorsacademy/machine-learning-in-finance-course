"""Cross-sectional momentum strategy on simulated assets."""

import numpy as np
import pandas as pd


def simulate_asset_returns(n_assets: int = 8, n_days: int = 1000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    market = rng.normal(0.0002, 0.008, n_days)
    data = {}
    for i in range(n_assets):
        beta = rng.uniform(0.7, 1.3)
        alpha = rng.normal(0.0, 0.0001)
        idio = rng.normal(0.0, 0.006, n_days)
        data[f"Asset_{i+1}"] = alpha + beta * market + idio
    idx = pd.date_range("2020-01-01", periods=n_days, freq="B")
    return pd.DataFrame(data, index=idx)


def momentum_weights(returns: pd.DataFrame, lookback: int = 126, top_k: int = 2) -> pd.DataFrame:
    scores = (1 + returns).rolling(lookback).apply(np.prod, raw=True) - 1
    ranks = scores.rank(axis=1, ascending=False, method="first")
    weights = (ranks <= top_k).astype(float) / top_k
    return weights


def backtest(returns: pd.DataFrame, weights: pd.DataFrame, cost_bps: float = 3.0) -> pd.Series:
    executed = weights.shift(1).fillna(0.0)
    gross = (executed * returns).sum(axis=1)
    turnover = executed.diff().abs().sum(axis=1).fillna(executed.abs().sum(axis=1))
    return gross - turnover * cost_bps / 10_000


def main() -> None:
    returns = simulate_asset_returns()
    weights = momentum_weights(returns)
    strategy = backtest(returns, weights)
    total = (1 + strategy.dropna()).prod() - 1
    print(f"Cross-sectional momentum total return: {total:.2%}")
    print("The example uses lagged weights and explicit turnover costs.")


if __name__ == "__main__":
    main()
