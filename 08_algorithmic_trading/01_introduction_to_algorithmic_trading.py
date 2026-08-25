"""A minimal vectorized strategy example with realistic signal timing."""

import numpy as np
import pandas as pd


def simulate_prices(n: int = 1000, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0002, 0.01, n)
    prices = 100 * np.exp(np.cumsum(returns))
    return pd.Series(prices, index=pd.date_range("2020-01-01", periods=n, freq="B"), name="price")


def build_strategy(prices: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"price": prices})
    df["return"] = df["price"].pct_change()
    df["fast_ma"] = df["price"].rolling(20).mean()
    df["slow_ma"] = df["price"].rolling(100).mean()
    df["signal"] = (df["fast_ma"] > df["slow_ma"]).astype(float)

    # The position is lagged by one period to avoid using today's close
    # to earn today's return.
    df["position"] = df["signal"].shift(1).fillna(0.0)
    df["strategy_return"] = df["position"] * df["return"]
    return df.dropna()


def main() -> None:
    df = build_strategy(simulate_prices())
    buy_hold = (1 + df["return"]).prod() - 1
    strategy = (1 + df["strategy_return"]).prod() - 1
    print(f"Buy-and-hold return: {buy_hold:.2%}")
    print(f"Strategy return:     {strategy:.2%}")
    print("Signals are lagged to preserve causal timing.")


if __name__ == "__main__":
    main()
