"""Maximize a long-only portfolio's ex-ante Sharpe ratio."""

import numpy as np
from scipy.optimize import minimize

from portfolio_utils import annualized_statistics, portfolio_statistics, simulate_returns


def maximum_sharpe_weights(expected_returns: np.ndarray, cov: np.ndarray, risk_free_rate: float = 0.02) -> np.ndarray:
    n_assets = len(expected_returns)
    x0 = np.repeat(1 / n_assets, n_assets)
    bounds = tuple((0.0, 1.0) for _ in range(n_assets))
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)

    def negative_sharpe(weights: np.ndarray) -> float:
        port_return = float(weights @ expected_returns)
        port_vol = np.sqrt(float(weights @ cov @ weights))
        if port_vol <= 0:
            return np.inf
        return -((port_return - risk_free_rate) / port_vol)

    result = minimize(negative_sharpe, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    if not result.success:
        raise RuntimeError(result.message)
    return result.x


def main() -> None:
    returns = simulate_returns(n_assets=5)
    annual_return, _, annual_cov = annualized_statistics(returns)
    weights = maximum_sharpe_weights(annual_return.to_numpy(), annual_cov.to_numpy())
    exp_ret, vol = portfolio_statistics(weights, annual_return, annual_cov)
    sharpe = (exp_ret - 0.02) / vol

    print("Maximum-Sharpe portfolio")
    print("Weights:", np.round(weights, 4))
    print(f"Expected return: {exp_ret:.4f}")
    print(f"Volatility:      {vol:.4f}")
    print(f"Sharpe ratio:    {sharpe:.4f}")


if __name__ == "__main__":
    main()
