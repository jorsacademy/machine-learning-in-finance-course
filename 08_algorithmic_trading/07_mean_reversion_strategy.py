"""Mean-reversion strategy using a rolling z-score."""

import numpy as np
import pandas as pd


def simulate_prices(n: int = 1000, seed: int = 123) -> pd.Series:
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.96 * x[t - 1] + rng.normal(0, 1)
    price = 100 + x
    return pd.Series(price, index=pd.date_range("2020-01-01", periods=n, freq="B"), name="price")


def build_positions(prices: pd.Series, window: int = 40, entry_z: float = 1.5, exit_z: float = 0.25) -> pd.Series:
    mean = prices.rolling(window).mean()
    std = prices.rolling(window).std(ddof=1)
    z = (prices - mean) / std

    position = pd.Series(0.0, index=prices.index)
    current = 0.0
    for i in range(len(prices)):
        zi = z.iloc[i]
        if np.isnan(zi):
            position.iloc[i] = 0.0
            continue
        if current == 0.0:
            if zi > entry_z:
                current = -1.0
            elif zi < -entry_z:
                current = 1.0
        elif current > 0 and zi >= -exit_z:
            current = 0.0
        elif current < 0 and zi <= exit_z:
            current = 0.0
        position.iloc[i] = current
    return position


def main() -> None:
    prices = simulate_prices()
    returns = prices.pct_change().fillna(0.0)
    target = build_positions(prices)
    executed = target.shift(1).fillna(0.0)
    turnover = executed.diff().abs().fillna(executed.abs())
    strategy = executed * returns - turnover * 3 / 10_000
    print(f"Mean-reversion total return: {(1 + strategy).prod() - 1:.2%}")
    print(f"Total turnover: {turnover.sum():.2f}")


if __name__ == "__main__":
    main()
