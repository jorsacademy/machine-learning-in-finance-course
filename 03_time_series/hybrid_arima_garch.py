"""Hybrid ARIMA-GARCH example.

ARIMA models the conditional mean. GARCH is then fitted to ARIMA residuals to
model conditional variance. The models are not combined by adding volatility
to the mean forecast; they answer different questions.
"""

import numpy as np
import pandas as pd
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA


def main() -> None:
    rng = np.random.default_rng(42)
    n = 800

    variance = np.empty(n)
    innovations = np.empty(n)
    returns = np.empty(n)
    variance[0] = 0.0001
    innovations[0] = rng.normal(0, np.sqrt(variance[0]))
    returns[0] = innovations[0]

    for t in range(1, n):
        variance[t] = 0.000002 + 0.08 * innovations[t - 1] ** 2 + 0.90 * variance[t - 1]
        innovations[t] = rng.normal(0, np.sqrt(variance[t]))
        returns[t] = 0.15 * returns[t - 1] + innovations[t]

    series = pd.Series(returns * 100, name="ReturnPct")

    mean_model = ARIMA(series, order=(1, 0, 0)).fit()
    residuals = mean_model.resid.dropna()

    variance_model = arch_model(
        residuals,
        mean="Zero",
        vol="GARCH",
        p=1,
        q=1,
        dist="normal",
        rescale=False,
    ).fit(disp="off")

    mean_forecast = mean_model.forecast(steps=5)
    variance_forecast = variance_model.forecast(horizon=5, reindex=False).variance.iloc[-1]
    volatility_forecast = np.sqrt(variance_forecast)

    output = pd.DataFrame(
        {
            "Mean_Return_Forecast_Pct": mean_forecast.to_numpy(),
            "Volatility_Forecast_Pct": volatility_forecast.to_numpy(),
        },
        index=[f"h+{i}" for i in range(1, 6)],
    )

    print(output)


if __name__ == "__main__":
    main()
