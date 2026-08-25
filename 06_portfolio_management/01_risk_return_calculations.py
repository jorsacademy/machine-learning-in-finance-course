"""Core portfolio risk and return calculations for educational use."""

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def simulate_returns(n_days: int = 1000, n_assets: int = 4, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    mean = np.linspace(0.0002, 0.0005, n_assets)
    base_corr = 0.25
    corr = np.full((n_assets, n_assets), base_corr)
    np.fill_diagonal(corr, 1.0)
    vol = np.linspace(0.008, 0.016, n_assets)
    cov = corr * np.outer(vol, vol)
    data = rng.multivariate_normal(mean, cov, size=n_days)
    columns = [f"Asset_{i+1}" for i in range(n_assets)]
    return pd.DataFrame(data, columns=columns)


def annualized_statistics(returns: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    annual_return = returns.mean() * TRADING_DAYS
    annual_volatility = returns.std(ddof=1) * np.sqrt(TRADING_DAYS)
    annual_covariance = returns.cov() * TRADING_DAYS
    return annual_return, annual_volatility, annual_covariance


def portfolio_statistics(weights: np.ndarray, annual_return: pd.Series, annual_cov: pd.DataFrame) -> tuple[float, float]:
    weights = np.asarray(weights, dtype=float)
    expected_return = float(weights @ annual_return.to_numpy())
    variance = float(weights @ annual_cov.to_numpy() @ weights)
    volatility = np.sqrt(variance)
    return expected_return, volatility


def sharpe_ratio(expected_return: float, volatility: float, risk_free_rate: float = 0.02) -> float:
    if volatility <= 0:
        raise ValueError("Volatility must be positive.")
    return (expected_return - risk_free_rate) / volatility


def main() -> None:
    returns = simulate_returns()
    annual_return, annual_volatility, annual_cov = annualized_statistics(returns)
    weights = np.repeat(1 / returns.shape[1], returns.shape[1])

    port_return, port_vol = portfolio_statistics(weights, annual_return, annual_cov)
    port_sharpe = sharpe_ratio(port_return, port_vol)

    print("Annualized asset returns:\n", annual_return.round(4))
    print("\nAnnualized asset volatilities:\n", annual_volatility.round(4))
    print("\nEqual-weight portfolio")
    print(f"Expected return: {port_return:.4f}")
    print(f"Volatility:      {port_vol:.4f}")
    print(f"Sharpe ratio:    {port_sharpe:.4f}")


if __name__ == "__main__":
    main()
