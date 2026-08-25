"""ARIMA implementation with chronological validation."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA


def main() -> None:
    rng = np.random.default_rng(42)
    innovations = rng.normal(0, 1, 300)
    series = pd.Series(100 + innovations.cumsum(), name="Price")

    split = int(len(series) * 0.8)
    train = series.iloc[:split]
    test = series.iloc[split:]

    model = ARIMA(train, order=(1, 1, 1))
    fitted = model.fit()
    forecast = fitted.forecast(steps=len(test))
    forecast.index = test.index

    mae = mean_absolute_error(test, forecast)
    rmse = mean_squared_error(test, forecast) ** 0.5

    print(fitted.summary())
    print(f"\nMAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print("\nForecast sample:")
    print(pd.DataFrame({"Actual": test, "Forecast": forecast}).head())


if __name__ == "__main__":
    main()
