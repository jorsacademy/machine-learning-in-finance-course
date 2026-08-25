"""Autocorrelation and partial autocorrelation diagnostics."""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf, pacf


def main() -> None:
    rng = np.random.default_rng(42)
    innovations = rng.normal(0, 1, 500)
    series = np.zeros(500)

    for t in range(2, len(series)):
        series[t] = 0.6 * series[t - 1] - 0.25 * series[t - 2] + innovations[t]

    s = pd.Series(series, name="AR2")
    acf_values = acf(s, nlags=10, fft=True)
    pacf_values = pacf(s, nlags=10, method="ywm")

    diagnostics = pd.DataFrame(
        {
            "Lag": range(11),
            "ACF": acf_values,
            "PACF": pacf_values,
        }
    )
    print(diagnostics)


if __name__ == "__main__":
    main()
