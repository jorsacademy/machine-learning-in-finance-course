"""Expected Shortfall (Conditional VaR) for financial return distributions."""

import numpy as np
import pandas as pd


def simulate_fat_tailed_returns(n: int = 3000, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    returns = 0.0003 + 0.01 * rng.standard_t(df=5, size=n)
    return pd.Series(returns, name="return")


def historical_var(returns: pd.Series, confidence: float = 0.975) -> float:
    threshold = returns.quantile(1.0 - confidence)
    return float(-threshold)


def expected_shortfall(returns: pd.Series, confidence: float = 0.975) -> float:
    threshold = returns.quantile(1.0 - confidence)
    tail = returns[returns <= threshold]
    if tail.empty:
        raise ValueError("No tail observations available.")
    return float(-tail.mean())


def main() -> None:
    returns = simulate_fat_tailed_returns()
    confidence = 0.975
    var = historical_var(returns, confidence)
    es = expected_shortfall(returns, confidence)

    print(f"{confidence:.1%} VaR: {var:.4%}")
    print(f"{confidence:.1%} Expected Shortfall: {es:.4%}")
    print(f"Tail severity ratio (ES / VaR): {es / var:.3f}")
    print("\nExpected Shortfall measures the average loss conditional on being beyond the VaR threshold.")


if __name__ == "__main__":
    main()
