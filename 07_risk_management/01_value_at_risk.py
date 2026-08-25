"""Value at Risk (VaR) examples for a simple return series."""

import numpy as np
import pandas as pd
from scipy.stats import norm


def simulate_returns(n: int = 1500, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0004, 0.012, n)
    return pd.Series(returns, name="return")


def historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
    alpha = 1.0 - confidence
    return float(-returns.quantile(alpha))


def parametric_var(returns: pd.Series, confidence: float = 0.95) -> float:
    mu = returns.mean()
    sigma = returns.std(ddof=1)
    z = norm.ppf(1.0 - confidence)
    return float(-(mu + z * sigma))


def portfolio_var(var_return: float, portfolio_value: float) -> float:
    return float(var_return * portfolio_value)


def main() -> None:
    returns = simulate_returns()
    portfolio_value = 1_000_000

    for confidence in (0.95, 0.99):
        hist = historical_var(returns, confidence)
        param = parametric_var(returns, confidence)
        print(f"{confidence:.0%} Historical VaR: {hist:.4%} | ${portfolio_var(hist, portfolio_value):,.0f}")
        print(f"{confidence:.0%} Parametric VaR: {param:.4%} | ${portfolio_var(param, portfolio_value):,.0f}")

    print("\nVaR is a quantile loss estimate, not a worst-case loss estimate.")


if __name__ == "__main__":
    main()
