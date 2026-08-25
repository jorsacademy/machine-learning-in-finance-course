"""Time-series manipulation examples with pandas.

Covers slicing, resampling, missing values, lags, returns, rolling windows,
and timezone conversion using deterministic data.
"""

import numpy as np
import pandas as pd


def main() -> None:
    rng = np.random.default_rng(42)
    index = pd.date_range("2024-01-01", periods=20, freq="D")
    series = pd.Series(100 + rng.normal(0, 1, len(index)).cumsum(), index=index, name="Price")

    print("Original series:\n", series.head(), "\n")
    print("Date slice:\n", series.loc["2024-01-05":"2024-01-10"], "\n")

    weekly = series.resample("W").agg(["mean", "min", "max"])
    print("Weekly aggregation:\n", weekly, "\n")

    missing = series.copy()
    missing.iloc[5:8] = np.nan
    forward_filled = missing.ffill()
    time_interpolated = missing.interpolate(method="time")

    print("Forward-filled values:\n", forward_filled.iloc[4:9], "\n")
    print("Time-interpolated values:\n", time_interpolated.iloc[4:9], "\n")

    frame = pd.DataFrame({"Price": series})
    frame["Lag_1"] = frame["Price"].shift(1)
    frame["Simple_Return"] = frame["Price"].pct_change()
    frame["Log_Return"] = np.log(frame["Price"] / frame["Price"].shift(1))
    frame["Rolling_Mean_5"] = frame["Price"].rolling(5).mean()
    frame["Rolling_Volatility_5"] = frame["Log_Return"].rolling(5).std()
    frame["Expanding_Mean"] = frame["Price"].expanding().mean()

    print("Derived features:\n", frame.tail(), "\n")

    utc_series = series.tz_localize("UTC")
    new_york_series = utc_series.tz_convert("America/New_York")
    print("Timezone conversion example:\n", new_york_series.head())


if __name__ == "__main__":
    main()
