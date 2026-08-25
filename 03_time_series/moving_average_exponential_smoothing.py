"""Moving averages and exponential smoothing."""

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import SimpleExpSmoothing


def main() -> None:
    rng = np.random.default_rng(42)
    index = pd.date_range("2024-01-01", periods=60, freq="D")
    values = 100 + np.linspace(0, 8, len(index)) + rng.normal(0, 2, len(index))
    series = pd.Series(values, index=index, name="Value")

    result = pd.DataFrame({"Value": series})
    result["SMA_5"] = series.rolling(5).mean()
    result["EWMA_alpha_0_2"] = series.ewm(alpha=0.2, adjust=False).mean()

    ses = SimpleExpSmoothing(series, initialization_method="estimated").fit(optimized=True)
    result["SES"] = ses.fittedvalues

    print(result.tail(10))
    print(f"\nOptimized smoothing level: {ses.params['smoothing_level']:.4f}")


if __name__ == "__main__":
    main()
