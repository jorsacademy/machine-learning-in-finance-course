"""High-frequency trading concepts using a toy limit-order-book stream.

This example is deliberately simplified. It demonstrates descriptive order-book
features and latency concepts, not a production HFT strategy.
"""

import numpy as np
import pandas as pd


def simulate_lob(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    mid = 100 + np.cumsum(rng.normal(0, 0.002, n))
    half_spread = rng.choice([0.005, 0.01, 0.015], size=n, p=[0.5, 0.4, 0.1])
    bid = mid - half_spread
    ask = mid + half_spread
    bid_size = rng.integers(100, 2000, n)
    ask_size = rng.integers(100, 2000, n)
    timestamps = pd.date_range("2025-01-02 09:30:00", periods=n, freq="100ms")
    return pd.DataFrame(
        {"bid": bid, "ask": ask, "bid_size": bid_size, "ask_size": ask_size},
        index=timestamps,
    )


def features(book: pd.DataFrame) -> pd.DataFrame:
    out = book.copy()
    out["mid"] = (out["bid"] + out["ask"]) / 2
    out["spread"] = out["ask"] - out["bid"]
    out["imbalance"] = (out["bid_size"] - out["ask_size"]) / (out["bid_size"] + out["ask_size"])
    out["microprice"] = (
        out["ask"] * out["bid_size"] + out["bid"] * out["ask_size"]
    ) / (out["bid_size"] + out["ask_size"])
    out["future_mid_change"] = out["mid"].shift(-10) - out["mid"]
    return out


def main() -> None:
    df = features(simulate_lob()).dropna()
    corr = df[["imbalance", "future_mid_change"]].corr().iloc[0, 1]
    avg_spread_bps = (df["spread"] / df["mid"] * 10_000).mean()
    print(f"Average spread: {avg_spread_bps:.3f} bps")
    print(f"Imbalance / future mid-change correlation: {corr:.4f}")
    print("Real HFT research requires exchange-grade data, queue modeling, latency measurement, and realistic execution simulation.")


if __name__ == "__main__":
    main()
