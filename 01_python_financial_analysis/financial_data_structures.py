"""Financial data structures in Python.

Educational examples covering pandas Series, DataFrame, DatetimeIndex,
MultiIndex, returns, filtering, and CSV export.
"""

from pathlib import Path

import pandas as pd


def main() -> None:
    prices = pd.Series(
        [100.0, 102.5, 101.2, 104.8],
        index=pd.Index(["AAPL", "MSFT", "GOOG", "AMZN"], name="Ticker"),
        name="Price",
    )
    print("Series example:\n", prices, "\n")

    market = pd.DataFrame(
        {
            "Ticker": ["AAPL", "MSFT", "GOOG", "AMZN"],
            "Price": [175.64, 325.12, 139.69, 145.24],
            "Volume": [1_200_000, 450_000, 800_000, 300_000],
        }
    ).set_index("Ticker")
    print("DataFrame example:\n", market, "\n")

    dates = pd.date_range("2024-01-02", periods=5, freq="B")
    ohlc = pd.DataFrame(
        {
            "Open": [100, 102, 105, 108, 110],
            "Close": [101, 103, 107, 109, 111],
        },
        index=dates,
    )
    ohlc.index.name = "Date"
    ohlc["Return"] = ohlc["Close"].pct_change()
    print("Time-indexed data:\n", ohlc, "\n")

    filtered = ohlc.loc[ohlc["Close"] > 105]
    print("Filtered observations:\n", filtered, "\n")

    multi = pd.concat(
        {
            "AAPL": pd.DataFrame({"Price": [175, 176], "Volume": [1000, 1100]}, index=dates[:2]),
            "MSFT": pd.DataFrame({"Price": [325, 327], "Volume": [900, 950]}, index=dates[:2]),
        },
        names=["Ticker", "Date"],
    )
    print("MultiIndex example:\n", multi, "\n")

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "financial_data_structures.csv"
    ohlc.to_csv(output_path)
    print(f"Saved example data to {output_path}")


if __name__ == "__main__":
    main()
