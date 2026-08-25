"""Handling missing financial data without introducing look-ahead bias."""

import numpy as np
import pandas as pd


def main() -> None:
    index = pd.date_range("2024-01-02", periods=8, freq="B")
    frame = pd.DataFrame(
        {
            "Close": [100.0, 101.0, np.nan, 102.5, np.nan, np.nan, 103.0, 104.0],
            "Volume": [1000, 1200, 1100, np.nan, 1500, 1400, 1600, 1700],
        },
        index=index,
    )

    print("Original data:\n", frame, "\n")
    print("Missing values per column:\n", frame.isna().sum(), "\n")

    # Forward fill is causal: it uses only information available up to time t.
    forward_filled = frame.ffill()
    print("Forward-filled data:\n", forward_filled, "\n")

    # Interpolation can be useful for descriptive analysis, but ordinary interpolation
    # may use future observations. Avoid it in production forecasting pipelines unless
    # the timing assumptions make that information available.
    interpolated = frame.interpolate(method="time")
    print("Time-interpolated data (descriptive use):\n", interpolated, "\n")

    missing_flags = frame.isna().add_suffix("_Missing")
    with_flags = pd.concat([frame, missing_flags], axis=1)
    print("Missingness indicators:\n", with_flags)


if __name__ == "__main__":
    main()
