"""Pairs trading with hedge-ratio estimation and spread z-scores."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def simulate_cointegrated_pair(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    common = np.cumsum(rng.normal(0.02, 1.0, n)) + 100
    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = 0.92 * spread[t - 1] + rng.normal(0, 0.8)
    a = common + 0.5 * spread
    b = 0.8 * common - 0.4 * spread + 20
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame({"A": a, "B": b}, index=idx)


def estimate_spread(prices: pd.DataFrame, train_fraction: float = 0.6) -> tuple[pd.Series, float, float]:
    split = int(len(prices) * train_fraction)
    model = LinearRegression().fit(prices[["B"]].iloc[:split], prices["A"].iloc[:split])
    hedge_ratio = float(model.coef_[0])
    intercept = float(model.intercept_)
    spread = prices["A"] - (intercept + hedge_ratio * prices["B"])
    return spread, hedge_ratio, intercept


def trading_positions(spread: pd.Series, train_fraction: float = 0.6) -> pd.Series:
    split = int(len(spread) * train_fraction)
    mean = float(spread.iloc[:split].mean())
    std = float(spread.iloc[:split].std(ddof=1))
    z = (spread - mean) / std

    target = pd.Series(0.0, index=spread.index)
    target[z > 2.0] = -1.0
    target[z < -2.0] = 1.0
    target[(z.abs() < 0.5)] = 0.0
    return target.replace(0.0, np.nan).ffill().fillna(0.0)


def main() -> None:
    prices = simulate_cointegrated_pair()
    spread, hedge_ratio, intercept = estimate_spread(prices)
    spread_position = trading_positions(spread)

    ret_a = prices["A"].pct_change().fillna(0.0)
    ret_b = prices["B"].pct_change().fillna(0.0)
    executed = spread_position.shift(1).fillna(0.0)

    # Dollar-normalized approximation for educational purposes.
    gross = executed * ret_a - executed * hedge_ratio * ret_b
    turnover = executed.diff().abs().fillna(executed.abs()) * (1 + abs(hedge_ratio))
    net = gross - turnover * 3 / 10_000

    print(f"Estimated hedge ratio: {hedge_ratio:.4f}")
    print(f"Estimated intercept:   {intercept:.4f}")
    print(f"Pairs strategy return: {(1 + net).prod() - 1:.2%}")
    print("Cointegration should be tested explicitly in a full research workflow.")


if __name__ == "__main__":
    main()
