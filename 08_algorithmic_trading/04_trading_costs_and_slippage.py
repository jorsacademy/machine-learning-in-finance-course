"""Transaction-cost and slippage modeling for educational backtests."""

import numpy as np
import pandas as pd


def apply_costs(
    returns: pd.Series,
    positions: pd.Series,
    commission_bps: float = 1.0,
    slippage_bps: float = 2.0,
) -> pd.DataFrame:
    turnover = positions.diff().abs().fillna(positions.abs())
    one_way_cost = (commission_bps + slippage_bps) / 10_000
    gross = positions.shift(1).fillna(0.0) * returns
    costs = turnover.shift(1).fillna(0.0) * one_way_cost
    net = gross - costs
    return pd.DataFrame({"gross_return": gross, "turnover": turnover, "cost": costs, "net_return": net})


def main() -> None:
    rng = np.random.default_rng(42)
    idx = pd.date_range("2024-01-01", periods=250, freq="B")
    returns = pd.Series(rng.normal(0.0003, 0.01, len(idx)), index=idx)
    signal = pd.Series(np.where(returns.rolling(20).mean() > 0, 1.0, 0.0), index=idx)

    result = apply_costs(returns, signal)
    gross = (1 + result["gross_return"]).prod() - 1
    net = (1 + result["net_return"]).prod() - 1
    print(f"Gross return: {gross:.2%}")
    print(f"Net return:   {net:.2%}")
    print(f"Total turnover: {result['turnover'].sum():.2f}")


if __name__ == "__main__":
    main()
