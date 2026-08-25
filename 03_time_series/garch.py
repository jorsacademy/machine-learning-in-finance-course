"""GARCH(1,1) implementation for financial return volatility."""

import numpy as np
import pandas as pd
from arch import arch_model


def main() -> None:
    rng = np.random.default_rng(42)
    n = 1000
    omega = 0.000002
    alpha = 0.08
    beta = 0.90

    variance = np.empty(n)
    returns = np.empty(n)
    variance[0] = omega / (1 - alpha - beta)
    returns[0] = rng.normal(0, np.sqrt(variance[0]))

    for t in range(1, n):
        variance[t] = omega + alpha * returns[t - 1] ** 2 + beta * variance[t - 1]
        returns[t] = rng.normal(0, np.sqrt(variance[t]))

    # arch is numerically more stable when daily returns are expressed in percent.
    returns_pct = pd.Series(returns * 100, name="ReturnPct")

    model = arch_model(
        returns_pct,
        mean="Constant",
        vol="GARCH",
        p=1,
        q=1,
        dist="normal",
        rescale=False,
    )
    fitted = model.fit(disp="off")

    forecast = fitted.forecast(horizon=10, reindex=False)
    variance_forecast_pct2 = forecast.variance.iloc[-1]
    volatility_forecast_pct = np.sqrt(variance_forecast_pct2)

    print(fitted.summary())
    print("\nForecasted variance (%^2):")
    print(variance_forecast_pct2)
    print("\nForecasted volatility (%):")
    print(volatility_forecast_pct)


if __name__ == "__main__":
    main()
