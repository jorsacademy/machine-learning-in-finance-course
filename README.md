# Machine Learning in Finance Course

Companion code and Jupyter notebooks for the Jors Academy course **Machine Learning in Finance: A Comprehensive Course**.

This repository focuses on practical, reproducible implementations of financial data processing, time-series analysis, machine learning, deep learning, portfolio management, risk management, algorithmic trading, NLP, and advanced financial applications.

## Repository Structure

- `01_python_financial_analysis/` — financial data structures and time-series manipulation
- `02_financial_data_processing/` — data sources, cleaning, alignment, normalization, and feature engineering
- `03_time_series/` — stationarity, ACF/PACF, smoothing, ARIMA, SARIMA, GARCH, and hybrid models
- `04_ml_market_prediction/` — classification and regression models for market prediction
- `05_deep_learning_in_finance/` — neural networks, CNNs, RNNs, LSTMs, GRUs, and attention
- `06_portfolio_management/` — portfolio analytics and ML-assisted diversification, including clustering
- `07_risk_management/` — risk analytics and machine-learning-based credit risk assessment
- `data/` — notes about data sources and local datasets

## Important Methodology Notes

Financial machine learning is highly sensitive to data leakage and invalid validation procedures. The examples in this repository use chronological train/test splits where appropriate and avoid random shuffling for time-series forecasting tasks.

Whenever a target depends on future prices or returns, predictors are constructed only from information that would have been available at the prediction timestamp.

For cross-sectional credit-risk examples, preprocessing is fitted only on training data and out-of-time validation is preferred when the objective is to approximate future deployment conditions.

Portfolio clustering examples standardize heterogeneous K-Means features and use a proper correlation-distance representation for hierarchical clustering rather than treating correlation-matrix rows as ordinary Euclidean observations.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

## Educational Use

The material is intended for educational and non-commercial use. See `LICENSE` for details.

## Disclaimer

This repository is for educational purposes only. It does not constitute investment advice, financial advice, credit advice, or a recommendation to trade, lend, borrow, or purchase any financial instrument. Historical performance, model estimates, and backtests do not guarantee future results.
