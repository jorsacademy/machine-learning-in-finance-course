"""Feature engineering for financial time-series data."""

import numpy as np
import pandas as pd


def relative_strength_index(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = losses.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def main() -> None:
    rng = np.random.default_rng(42)
    index = pd.date_range("2024-01-01", periods=120, freq="B")
    close = pd.Series(100 + rng.normal(0, 1, len(index)).cumsum(), index=index)
    volume = pd.Series(rng.integers(800_000, 2_000_000, len(index)), index=index)

    df = pd.DataFrame({"Close": close, "Volume": volume})
    df["Return_1d"] = df["Close"].pct_change()
    df["Log_Return_1d"] = np.log(df["Close"] / df["Close"].shift(1))
    df["Lag_Return_1"] = df["Return_1d"].shift(1)
    df["Lag_Return_5"] = df["Return_1d"].shift(5)
    df["SMA_10"] = df["Close"].rolling(10).mean()
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["SMA_Ratio"] = df["SMA_10"] / df["SMA_20"]
    df["Rolling_Volatility_20"] = df["Log_Return_1d"].rolling(20).std() * np.sqrt(252)
    df["RSI_14"] = relative_strength_index(df["Close"], 14)
    df["Volume_Change"] = df["Volume"].pct_change()

    # Example next-period classification target. Features at time t must be used
    # to predict the return from t to t+1, not the already-known return at time t.
    df["Target_Up_Next_Day"] = (df["Return_1d"].shift(-1) > 0).astype("Int64")

    model_data = df.dropna().copy()
    print(model_data.tail())


if __name__ == "__main__":
    main()
