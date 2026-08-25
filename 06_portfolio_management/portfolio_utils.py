"""Shared helpers for portfolio management examples."""

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def simulate_returns(n_days: int = 1000, n_assets: int = 4, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    mean = np.linspace(0.0002, 0.0005, n_assets)
    corr = np.full((n_assets, n_assets), 0.25)
    np.fill_diagonal(corr, 1.0)
    vol = np.linspace(0.008, 0.016, n_assets)
    cov = corr * np.outer(vol, vol)
    data = rng.multivariate_normal(mean, cov, size=n_days)
    return pd.DataFrame(data, columns=[f"Asset_{i+1}" for i in range(n_assets)])


def annualized_statistics(returns: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    annual_return = returns.mean() * TRADING_DAYS
    annual_volatility = returns.std(ddof=1) * np.sqrt(TRADING_DAYS)
    annual_covariance = returns.cov() * TRADING_DAYS
    return annual_return, annual_volatility, annual_covariance


def portfolio_statistics(weights: np.ndarray, annual_return: pd.Series, annual_cov: pd.DataFrame) -> tuple[float, float]:
    w = np.asarray(weights, dtype=float)
    expected_return = float(w @ annual_return.to_numpy())
    variance = float(w @ annual_cov.to_numpy() @ w)
    return expected_return, float(np.sqrt(variance))
