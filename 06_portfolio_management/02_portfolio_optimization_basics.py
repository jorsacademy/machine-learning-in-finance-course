"""Long-only mean-variance portfolio optimization with scipy."""

import numpy as np
from scipy.optimize import minimize

from portfolio_utils import annualized_statistics, portfolio_statistics, simulate_returns


def minimum_variance_weights(annual_cov: np.ndarray) -> np.ndarray:
    n_assets = annual_cov.shape[0]
    x0 = np.repeat(1 / n_assets, n_assets)

    def objective(weights: np.ndarray) -> float:
        return float(weights @ annual_cov @ weights)

    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bounds = tuple((0.0, 1.0) for _ in range(n_assets))
    result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    if not result.success:
        raise RuntimeError(result.message)
    return result.x


def target_return_weights(expected_returns: np.ndarray, annual_cov: np.ndarray, target_return: float) -> np.ndarray:
    n_assets = len(expected_returns)
    x0 = np.repeat(1 / n_assets, n_assets)

    def objective(weights: np.ndarray) -> float:
        return float(weights @ annual_cov @ weights)

    constraints = (
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        {"type": "eq", "fun": lambda w: float(w @ expected_returns) - target_return},
    )
    bounds = tuple((0.0, 1.0) for _ in range(n_assets))
    result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    if not result.success:
        raise RuntimeError(result.message)
    return result.x


def main() -> None:
    returns = simulate_returns()
    annual_return, _, annual_cov = annualized_statistics(returns)

    min_var = minimum_variance_weights(annual_cov.to_numpy())
    target = float(annual_return.median())
    target_w = target_return_weights(annual_return.to_numpy(), annual_cov.to_numpy(), target)

    for name, weights in [("Minimum variance", min_var), ("Target return", target_w)]:
        exp_ret, vol = portfolio_statistics(weights, annual_return, annual_cov)
        print(f"\n{name} portfolio")
        print("Weights:", np.round(weights, 4))
        print(f"Expected return: {exp_ret:.4f}")
        print(f"Volatility:      {vol:.4f}")


if __name__ == "__main__":
    main()
