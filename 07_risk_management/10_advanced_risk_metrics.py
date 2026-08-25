"""Advanced risk metrics: drawdown, downside deviation, Sortino ratio, and Omega ratio."""

import numpy as np
import pandas as pd


def simulate_returns(n: int = 1500, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    returns = 0.0004 + 0.011 * rng.standard_t(df=6, size=n)
    return pd.Series(returns, name="return")


def max_drawdown(returns: pd.Series) -> float:
    wealth = (1.0 + returns).cumprod()
    peak = wealth.cummax()
    drawdown = wealth / peak - 1.0
    return float(drawdown.min())


def downside_deviation(returns: pd.Series, minimum_acceptable_return: float = 0.0, periods_per_year: int = 252) -> float:
    downside = np.minimum(returns - minimum_acceptable_return, 0.0)
    return float(np.sqrt(np.mean(np.square(downside))) * np.sqrt(periods_per_year))


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    annual_return = returns.mean() * periods_per_year
    downside = downside_deviation(returns, 0.0, periods_per_year)
    return float((annual_return - risk_free_rate) / downside) if downside > 0 else np.nan


def omega_ratio(returns: pd.Series, threshold: float = 0.0) -> float:
    gains = np.maximum(returns - threshold, 0.0).sum()
    losses = np.maximum(threshold - returns, 0.0).sum()
    return float(gains / losses) if losses > 0 else np.inf


def main() -> None:
    returns = simulate_returns()
    print(f"Maximum drawdown:    {max_drawdown(returns):.2%}")
    print(f"Downside deviation: {downside_deviation(returns):.2%}")
    print(f"Sortino ratio:      {sortino_ratio(returns):.3f}")
    print(f"Omega ratio:        {omega_ratio(returns):.3f}")
    print("\nNo single risk metric is sufficient. Metrics should be interpreted together and in context.")


if __name__ == "__main__":
    main()
