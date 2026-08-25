"""Basic market microstructure calculations from top-of-book quotes."""

import pandas as pd


def quote_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["mid"] = (out["bid"] + out["ask"]) / 2
    out["spread"] = out["ask"] - out["bid"]
    out["relative_spread_bps"] = out["spread"] / out["mid"] * 10_000
    out["microprice"] = (
        out["ask"] * out["bid_size"] + out["bid"] * out["ask_size"]
    ) / (out["bid_size"] + out["ask_size"])
    out["order_book_imbalance"] = (
        out["bid_size"] - out["ask_size"]
    ) / (out["bid_size"] + out["ask_size"])
    return out


def main() -> None:
    quotes = pd.DataFrame(
        {
            "bid": [99.98, 100.00, 100.01],
            "ask": [100.02, 100.04, 100.05],
            "bid_size": [800, 1200, 600],
            "ask_size": [500, 700, 1000],
        }
    )
    print(quote_metrics(quotes).round(4))
    print("\nMicroprice and imbalance are descriptive signals, not guaranteed predictors of future price moves.")


if __name__ == "__main__":
    main()
