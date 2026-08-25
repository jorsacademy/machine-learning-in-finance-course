"""News-based trading signal example with strict timestamp alignment.

The key lesson is that sentiment must be available before the return being
predicted. The example therefore shifts the signal forward one period before
computing strategy returns.
"""

import numpy as np
import pandas as pd


def make_data(n: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2025-01-01", periods=n, freq="D")
    sentiment = rng.normal(0.0, 1.0, n)
    noise = rng.normal(0.0, 0.012, n)

    # Synthetic next-period relationship for demonstration only.
    returns = 0.0015 * np.roll(sentiment, 1) + noise
    returns[0] = noise[0]

    return pd.DataFrame({"sentiment": sentiment, "return": returns}, index=index)


def backtest(df: pd.DataFrame, cost_bps: float = 5.0) -> pd.DataFrame:
    out = df.copy()
    out["signal"] = np.where(out["sentiment"] > 0.5, 1.0, np.where(out["sentiment"] < -0.5, -1.0, 0.0))

    # Position at t uses information from t-1, avoiding same-period look-ahead.
    out["position"] = out["signal"].shift(1).fillna(0.0)
    out["turnover"] = out["position"].diff().abs().fillna(out["position"].abs())
    out["gross_strategy_return"] = out["position"] * out["return"]
    out["cost"] = out["turnover"] * cost_bps / 10_000
    out["net_strategy_return"] = out["gross_strategy_return"] - out["cost"]
    out["equity"] = (1.0 + out["net_strategy_return"]).cumprod()
    return out


def main() -> None:
    result = backtest(make_data())
    print(result[["sentiment", "signal", "position", "return", "net_strategy_return", "equity"]].tail(10))
    print(f"\nTerminal equity: {result['equity'].iloc[-1]:.4f}")
    print("Synthetic predictability is embedded for teaching and is not evidence of a real trading edge.")


if __name__ == "__main__":
    main()
