"""Download and inspect market data with yfinance.

Network access is required. Data availability and column structure may change
with upstream provider behavior.
"""

import yfinance as yf


def main() -> None:
    symbol = "AAPL"
    data = yf.download(
        symbol,
        start="2024-01-01",
        end="2025-01-01",
        auto_adjust=False,
        progress=False,
    )

    if data.empty:
        raise RuntimeError("No data returned by Yahoo Finance.")

    print(data.head())
    print("\nColumns:", list(data.columns))

    ticker = yf.Ticker(symbol)
    print("\nRecent dividends:\n", ticker.dividends.tail())
    print("\nRecent splits:\n", ticker.splits.tail())


if __name__ == "__main__":
    main()
