"""Alpha Vantage integration example.

Set the environment variable ALPHA_VANTAGE_API_KEY before running.
"""

import os

from alpha_vantage.timeseries import TimeSeries


def main() -> None:
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing ALPHA_VANTAGE_API_KEY. Set it in your environment before running this script."
        )

    ts = TimeSeries(key=api_key, output_format="pandas")
    data, metadata = ts.get_daily(symbol="MSFT", outputsize="compact")

    print("Metadata:")
    print(metadata)
    print("\nDaily data:")
    print(data.head())


if __name__ == "__main__":
    main()
