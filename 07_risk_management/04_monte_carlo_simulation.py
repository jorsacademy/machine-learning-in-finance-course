"""Monte Carlo simulation of portfolio wealth under correlated asset returns."""

import numpy as np


def simulate_terminal_wealth(
    initial_wealth: float = 1_000_000,
    n_paths: int = 20_000,
    n_days: int = 252,
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)

    annual_mu = np.array([0.08, 0.04, 0.06])
    annual_vol = np.array([0.18, 0.08, 0.12])
    corr = np.array(
        [
            [1.00, 0.15, 0.45],
            [0.15, 1.00, 0.25],
            [0.45, 0.25, 1.00],
        ]
    )
    annual_cov = np.outer(annual_vol, annual_vol) * corr
    weights = np.array([0.50, 0.30, 0.20])

    daily_mu = annual_mu / 252
    daily_cov = annual_cov / 252

    draws = rng.multivariate_normal(daily_mu, daily_cov, size=(n_paths, n_days))
    portfolio_returns = draws @ weights
    terminal_wealth = initial_wealth * np.prod(1.0 + portfolio_returns, axis=1)
    return terminal_wealth


def main() -> None:
    terminal = simulate_terminal_wealth()
    initial = 1_000_000

    p05, median, p95 = np.quantile(terminal, [0.05, 0.50, 0.95])
    probability_of_loss = np.mean(terminal < initial)

    print(f"5th percentile terminal wealth:  ${p05:,.0f}")
    print(f"Median terminal wealth:          ${median:,.0f}")
    print(f"95th percentile terminal wealth: ${p95:,.0f}")
    print(f"Probability of ending below initial wealth: {probability_of_loss:.2%}")
    print("\nMonte Carlo results depend on the assumed return distribution and parameter estimates.")


if __name__ == "__main__":
    main()
