# Risk Management

This section covers both traditional financial risk measurement and machine-learning-based risk assessment. The goal is to show how statistical risk models, scenario analysis, simulation, and predictive models complement each other.

## Traditional risk methods

### Value at Risk (VaR)

VaR estimates a loss quantile over a chosen horizon and confidence level. For a return distribution \(R\), a one-period historical VaR at confidence level \(c\) can be written as

\[
VaR_c = -Q_{1-c}(R),
\]

where \(Q\) is the empirical quantile function.

The repository includes both historical and Gaussian parametric VaR examples. VaR is not a worst-case loss estimate and provides no information about the average severity of losses beyond the threshold.

### Expected Shortfall

Expected Shortfall (ES), also called Conditional VaR, measures the average loss in the tail beyond the VaR threshold:

\[
ES_c = -E[R \mid R \leq Q_{1-c}(R)].
\]

ES is useful because it explicitly measures tail severity rather than only the location of a tail quantile.

### Stress testing

Stress testing applies explicit shocks to portfolio positions or risk factors. Unlike VaR, a stress test does not need to assign a probability to the scenario. The objective is to answer questions such as:

- What happens if equities fall 30%?
- What happens if interest rates rise sharply?
- What happens if credit spreads widen while liquidity deteriorates?

Stress testing is particularly useful for events that may be too rare or structurally different from the historical sample.

### Monte Carlo simulation

Monte Carlo simulation generates many possible future paths from an assumed statistical model. It can be used to estimate:

- terminal wealth distributions;
- probabilities of loss;
- tail risk;
- nonlinear payoff distributions;
- scenario-dependent portfolio outcomes.

Simulation quality depends on the assumptions used for means, volatilities, correlations, distributions, and dynamics. A precise simulation of a poor model is still a poor risk estimate.

### Scenario analysis

Scenario analysis maps macroeconomic or market shocks into portfolio outcomes. The included example uses simple linear factor sensitivities for growth, interest rates, and inflation. Real systems may include nonlinear sensitivities, optionality, path dependence, liquidity effects, and cross-factor interactions.

## Machine learning for credit risk

Credit risk assessment estimates the probability that a borrower will fail to meet contractual obligations. A common binary target is

\[
Y = \begin{cases}
1, & \text{default within the defined performance window},\\
0, & \text{otherwise}.
\end{cases}
\]

A model may estimate

\[
PD_i = P(Y_i = 1 \mid X_i),
\]

where \(PD_i\) is probability of default conditional on information available at the decision timestamp.

Only information available at that timestamp should be used. Features produced after approval or during the performance window can create target leakage.

A simplified expected-loss relationship is

\[
EL = PD \times LGD \times EAD,
\]

where PD is probability of default, LGD is loss given default, and EAD is exposure at default.

The included credit-risk example uses an out-of-time holdout, preprocessing inside a scikit-learn pipeline, class weighting, ROC AUC, precision-recall AUC, Brier score, and threshold-based evaluation.

## Neural networks for risk prediction

Neural networks can model nonlinear interactions in large datasets, but additional model complexity does not automatically improve risk decisions. In high-impact domains such as lending, discrimination, calibration, interpretability, stability, fairness, and governance matter at least as much as raw predictive accuracy.

The neural-network example uses standardized training features, a chronological holdout, dropout regularization, binary cross-entropy, and ROC AUC evaluation.

## Anomaly detection

Unsupervised anomaly detection can identify observations that are unusual relative to the reference population. The included Isolation Forest example demonstrates how unusual transaction profiles can be ranked by anomaly score.

An anomaly flag is not evidence of fraud or misconduct. In practice, anomaly detection is normally one component of a broader monitoring and investigation workflow.

## Ensemble risk scoring

Ensemble models combine multiple learners to reduce dependence on a single modeling assumption. The included example uses soft voting across logistic regression, Random Forest, and histogram gradient boosting.

Ensembling can improve robustness, but ensemble predictions still require out-of-sample validation, calibration, drift monitoring, and governance.

## Advanced risk metrics

The section also includes several complementary metrics:

- maximum drawdown;
- downside deviation;
- Sortino ratio;
- Omega ratio.

Maximum drawdown measures peak-to-trough wealth loss. Downside deviation focuses only on returns below a chosen threshold. The Sortino ratio replaces total volatility with downside risk, while the Omega ratio compares probability-weighted gains and losses around a threshold.

No single risk metric should be treated as sufficient. Tail risk, path dependence, concentration, liquidity, leverage, model risk, and scenario sensitivity should be analyzed together.

## Validation and governance principles

Across the section, the examples follow several general principles:

1. Fit preprocessing only on training data.
2. Use chronological or out-of-time validation when time ordering matters.
3. Separate ranking performance from probability calibration.
4. Do not interpret anomaly scores as proof of wrongdoing.
5. Treat stress tests as conditional scenarios rather than forecasts.
6. Document assumptions behind simulations and parametric risk models.
7. Monitor model drift and performance across time and relevant subpopulations.
8. Treat protected attributes, fairness, privacy, and regulatory requirements as first-class design constraints in credit applications.

## Included files

- `01_value_at_risk.py` — historical and parametric VaR.
- `02_expected_shortfall.py` — historical Expected Shortfall.
- `03_stress_testing.py` — deterministic portfolio stress scenarios.
- `04_monte_carlo_simulation.py` — correlated multi-asset Monte Carlo simulation.
- `05_scenario_analysis.py` — macro-factor scenario analysis.
- `ml_credit_risk_assessment.py` — machine-learning credit risk workflow.
- `07_neural_network_risk_prediction.py` — neural-network risk classification.
- `08_anomaly_detection.py` — Isolation Forest anomaly detection.
- `09_ensemble_risk_scoring.py` — soft-voting ensemble risk scoring.
- `10_advanced_risk_metrics.py` — drawdown, downside deviation, Sortino, and Omega metrics.

All examples are educational. They are not production risk systems, underwriting systems, fraud-detection systems, or investment advice.
