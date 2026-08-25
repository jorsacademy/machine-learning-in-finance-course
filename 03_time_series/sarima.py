"""SARIMA implementation for seasonal time-series data."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX


def main() -> None:
    rng = np.random.default_rng(42)
    periods = 180
    index = pd.date_range("2010-01-31", periods=periods, freq="ME")
    trend = np.linspace(100, 130, periods)
    seasonality = 8 * np.sin(2 * np.pi * np.arange(periods) / 12)
    noise = rng.normal(0, 2, periods)
    series = pd.Series(trend + seasonality + noise, index=index, name="Value")

    split = int(len(series) * 0.8)
    train = series.iloc[:split]
    test = series.iloc[split:]

    model = SARIMAX(
        train,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False)
    forecast = fitted.get_forecast(steps=len(test)).predicted_mean
    forecast.index = test.index

    mae = mean_absolute_error(test, forecast)
    print(fitted.summary())
    print(f"\nMAE: {mae:.4f}")
    print(pd.DataFrame({"Actual": test, "Forecast": forecast}).head())


if __name__ == "__main__":
    main()
