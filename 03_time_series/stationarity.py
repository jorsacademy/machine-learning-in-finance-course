"""Stationarity diagnostics with ADF and KPSS tests."""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss


def run_tests(series: pd.Series) -> None:
    clean = series.dropna()

    adf = adfuller(clean, autolag="AIC")
    print(f"ADF statistic: {adf[0]:.4f}")
    print(f"ADF p-value:   {adf[1]:.4f}")

    kpss_result = kpss(clean, regression="c", nlags="auto")
    print(f"KPSS statistic: {kpss_result[0]:.4f}")
    print(f"KPSS p-value:   {kpss_result[1]:.4f}")


def main() -> None:
    rng = np.random.default_rng(42)
    prices = pd.Series(100 + rng.normal(0, 1, 500).cumsum(), name="Price")
    returns = prices.pct_change()

    print("Price level tests")
    run_tests(prices)
    print("\nReturn tests")
    run_tests(returns)


if __name__ == "__main__":
    main()
