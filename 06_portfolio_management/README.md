# Portfolio Management

This section covers classical portfolio theory and machine-learning approaches to portfolio construction. The examples are designed to be reproducible, transparent, and suitable for teaching.

## Included lessons

### 1. Risk and Return Calculations

`01_risk_return_calculations.py`

Covers:

- simple asset returns;
- annualized expected return;
- annualized volatility;
- covariance matrices;
- portfolio expected return;
- portfolio variance and volatility;
- Sharpe ratio.

Core formulas:

\[
E[R_p] = w^T \mu
\]

and

\[
\sigma_p^2 = w^T \Sigma w.
\]

### 2. Portfolio Optimization Basics

`02_portfolio_optimization_basics.py`

Demonstrates constrained long-only mean-variance optimization using SLSQP. It includes minimum-variance optimization and minimum-variance optimization subject to a target expected return.

The examples enforce

\[
\sum_i w_i = 1
\]

with long-only bounds

\[
0 \leq w_i \leq 1.
\]

### 3. Efficient Frontier Construction

`03_efficient_frontier.py`

Constructs a numerical efficient frontier by repeatedly solving a constrained minimum-variance problem for different target returns. The visualization also compares the frontier with an equal-weight portfolio.

### 4. Sharpe Ratio Optimization

`04_sharpe_ratio_optimization.py`

Finds the long-only portfolio that maximizes the ex-ante Sharpe ratio:

\[
S = \frac{E[R_p] - R_f}{\sigma_p}.
\]

Expected returns and covariance estimates are treated as inputs rather than known truths. In real portfolio management, estimation error is often more important than optimization precision.

### 5. Factor Models

`05_factor_models.py`

Demonstrates linear factor exposure estimation for multiple assets. The example estimates market, size, and value exposures using ordinary least squares.

A generic factor representation is

\[
R_i = \alpha_i + \beta_{i1}F_1 + \cdots + \beta_{ik}F_k + \epsilon_i.
\]

Factor models can help explain systematic return drivers and portfolio concentration that is not visible from asset labels alone.

### 6. Clustering for Portfolio Diversification

`clustering_portfolio_diversification.py`

Clustering is used as a structural tool rather than an automatic portfolio optimizer. The implementation includes:

- feature standardization before K-Means;
- silhouette diagnostics;
- clustering by return and risk characteristics;
- correlation-distance hierarchical clustering;
- representative asset selection;
- cluster visualization.

For correlation-based hierarchical clustering, the implementation uses

\[
d_{ij} = \sqrt{\frac{1 - \rho_{ij}}{2}}
\]

and converts the resulting matrix to a valid condensed distance representation before linkage.

### 7. Neural Networks for Portfolio Weights

`07_neural_network_portfolio_weights.py`

Uses a small neural network to map recent return statistics to long-only portfolio weights. A softmax output layer guarantees that weights are non-negative and sum to one.

The model is optimized using a differentiable Sharpe-like training objective. The example uses chronological train/test separation and estimates preprocessing statistics only on the training period.

This is a teaching example, not a claim that neural networks should replace classical portfolio optimization.

### 8. Reinforcement Learning Basics

`08_reinforcement_learning_basics.py`

Introduces the portfolio-management interpretation of:

- state;
- action;
- reward;
- transition;
- policy;
- episode.

`portfolio_rl_env.py` contains a deliberately small two-asset regime environment used by the reinforcement-learning examples.

### 9. Q-Learning for Portfolio Management

`09_q_learning_portfolio_management.py`

Provides a complete tabular Q-learning example rather than pseudocode placeholders. The implementation includes:

- epsilon-greedy exploration;
- temporal-difference updates;
- discounting;
- epsilon decay;
- out-of-sample evaluation in a separately seeded toy environment.

The Q-learning update is

\[
Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \alpha\left[r_{t+1} + \gamma\max_a Q(s_{t+1},a) - Q(s_t,a_t)\right].
\]

### 10. Deep Reinforcement Learning

`10_deep_reinforcement_learning.py`

Extends the same environment with a small Deep Q-Network. It demonstrates:

- neural approximation of the Q-function;
- experience replay;
- epsilon-greedy exploration;
- a target network;
- batched temporal-difference targets.

The environment is intentionally simple so the mechanics of DQN remain understandable. A realistic portfolio RL system would additionally require transaction costs, turnover constraints, non-stationary observations, partial observability, risk constraints, careful reward design, walk-forward validation, and robust benchmark comparisons.

## Shared utilities

`portfolio_utils.py` contains reusable return, covariance, and portfolio-statistics helpers.

`portfolio_rl_env.py` contains the toy environment shared by the reinforcement-learning lessons.

## Important methodological cautions

Portfolio optimization is extremely sensitive to expected-return and covariance estimates. A mathematically optimal portfolio can be economically poor if the inputs are unstable.

Machine-learning portfolio methods create additional risks:

- look-ahead bias;
- overfitting;
- unstable allocation weights;
- excessive turnover;
- ignored transaction costs;
- regime dependence;
- unrealistic reward functions;
- weak benchmark selection.

For serious work, use chronological or walk-forward evaluation and compare against simple baselines such as equal weight, minimum variance, and maximum diversification.

The material in this repository is educational and does not constitute investment advice.
