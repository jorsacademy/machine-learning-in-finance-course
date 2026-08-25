"""Construct a long-only efficient frontier with constrained optimization."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from portfolio_utils import annualized_statistics, portfolio_statistics, simulate_returns


def efficient_frontier(expected_returns: np.ndarray, cov: np.ndarray, n_points: int = 30):
    n_assets = len(expected_returns)
    bounds = tuple((0.0, 1.0) for _ in range(n_assets))
    x0 = np.repeat(1 / n_assets, n_assets)
    targets = np.linspace(expected_returns.min(), expected_returns.max(), n_points)
    frontier = []

    for target in targets:
        constraints = (
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w, t=target: float(w @ expected_returns) - t},
        )
        result = minimize(
            lambda w: float(w @ cov @ w),
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        if result.success:
            volatility = np.sqrt(float(result.x @ cov @ result.x))
            frontier.append((target, volatility, result.x.copy()))

    return frontier


def main() -> None:
    returns = simulate_returns(n_assets=5)
    annual_return, _, annual_cov = annualized_statistics(returns)
    frontier = efficient_frontier(annual_return.to_numpy(), annual_cov.to_numpy())

    frontier_returns = [item[0] for item in frontier]
    frontier_vols = [item[1] for item in frontier]

    equal_weights = np.repeat(1 / returns.shape[1], returns.shape[1])
    eq_ret, eq_vol = portfolio_statistics(equal_weights, annual_return, annual_cov)

    plt.figure(figsize=(9, 6))
    plt.plot(frontier_vols, frontier_returns, marker="o", label="Efficient frontier")
    plt.scatter([eq_vol], [eq_ret], s=80, label="Equal-weight portfolio")
    plt.xlabel("Annualized volatility")
    plt.ylabel("Annualized expected return")
    plt.title("Long-Only Efficient Frontier")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
