"""A compact vectorized backtesting framework with basic risk metrics."""

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    returns: pd.Series
    equity: pd.Series
    turnover: pd.Series

    def annualized_return(self, periods_per_year: int = 252) -> float:
        n = len(self.returns)
        return float((1 + self.returns).prod() ** (periods_per_year / n) - 1)

    def annualized_volatility(self, periods_per_year: int = 252) -> float:
        return float(self.returns.std(ddof=1) * np.sqrt(periods_per_year))

    def sharpe(self, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
        vol = self.annualized_volatility(periods_per_year)
        return np.nan if vol == 0 else (self.annualized_return(periods_per_year) - risk_free_rate) / vol

    def max_drawdown(self) -> float:
        peak = self.equity.cummax()
        drawdown = self.equity / peak - 1
        return float(drawdown.min())


def run_backtest(
    asset_returns: pd.Series,
    target_positions: pd.Series,
    cost_bps: float = 2.0,
) -> BacktestResult:
    asset_returns, target_positions = asset_returns.align(target_positions, join="inner")
    executed_position = target_positions.shift(1).fillna(0.0)
    turnover = executed_position.diff().abs().fillna(executed_position.abs())
    gross = executed_position * asset_returns
    costs = turnover * cost_bps / 10_000
    net = gross - costs
    equity = (1 + net).cumprod()
    return BacktestResult(net, equity, turnover)


def main() -> None:
    rng = np.random.default_rng(7)
    idx = pd.date_range("2021-01-01", periods=750, freq="B")
    returns = pd.Series(rng.normal(0.00025, 0.012, len(idx)), index=idx)
    momentum = returns.rolling(60).sum()
    positions = (momentum > 0).astype(float)

    result = run_backtest(returns, positions, cost_bps=3.0)
    print(f"Annualized return:     {result.annualized_return():.2%}")
    print(f"Annualized volatility: {result.annualized_volatility():.2%}")
    print(f"Sharpe ratio:          {result.sharpe():.3f}")
    print(f"Maximum drawdown:      {result.max_drawdown():.2%}")


if __name__ == "__main__":
    main()
