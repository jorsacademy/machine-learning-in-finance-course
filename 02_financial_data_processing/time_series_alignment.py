"""Align multiple financial time series by timestamp."""

import pandas as pd


def main() -> None:
    aapl = pd.Series(
        [180.0, 181.5, 182.2],
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-05"]),
        name="AAPL",
    )
    msft = pd.Series(
        [370.0, 372.0, 371.5],
        index=pd.to_datetime(["2024-01-03", "2024-01-04", "2024-01-05"]),
        name="MSFT",
    )

    union = pd.concat([aapl, msft], axis=1, join="outer")
    intersection = pd.concat([aapl, msft], axis=1, join="inner")

    print("Outer alignment (union):\n", union, "\n")
    print("Inner alignment (intersection):\n", intersection, "\n")

    business_days = pd.date_range("2024-01-02", "2024-01-05", freq="B")
    reindexed = union.reindex(business_days)
    print("Reindexed to business days:\n", reindexed, "\n")

    # Forward filling is causal for price series when the previous observed price is
    # considered the latest known value. It should not be used blindly across long gaps.
    print("Forward-filled alignment:\n", reindexed.ffill())


if __name__ == "__main__":
    main()
