# Machine Learning for Credit Risk Assessment

Credit risk assessment estimates the probability that a borrower will fail to meet contractual obligations. In a machine-learning workflow, the goal is usually not merely to classify borrowers as "good" or "bad". The more useful output is often a calibrated probability of default that can be combined with exposure, recovery assumptions, pricing, capital requirements, and lending policy.

## Typical use cases

- application credit scoring;
- probability-of-default estimation;
- behavioral risk monitoring;
- early-warning systems;
- portfolio segmentation;
- collections prioritization;
- fraud-screening support.

Credit risk and fraud risk are related but distinct. A borrower can be genuinely unable to repay without committing fraud, and a fraudulent application can exist even if its short-term payment behavior appears normal.

## Core modeling target

For binary default prediction, a common target is

\[
Y = \begin{cases}
1, & \text{default within the defined performance window},\\
0, & \text{otherwise}.
\end{cases}
\]

The performance window must be defined before modeling. Examples include default within 12 months after origination or becoming 90+ days past due within a specified horizon.

A model may estimate

\[
PD_i = P(Y_i = 1 \mid X_i),
\]

where \(PD_i\) is the probability of default for borrower \(i\) conditional on information available at the decision time.

## Feature engineering

Common features include:

- debt-to-income ratio;
- loan-to-income ratio;
- utilization ratio;
- repayment history;
- number of recent delinquencies;
- account age;
- employment stability;
- recent credit inquiries;
- existing debt burden;
- product and loan-purpose variables.

Only information available at the decision timestamp should be used. Features generated after loan approval or after the beginning of the performance window create target leakage.

## Validation strategy

A random train/test split can overstate performance when borrower populations, underwriting policy, or macroeconomic conditions change through time. For production-oriented credit modeling, an out-of-time holdout is often more informative:

1. train on older applications;
2. validate or tune on a later period;
3. test on the most recent untouched period.

The included example follows this principle.

## Class imbalance

Defaults are often less common than non-defaults. Accuracy can therefore be misleading. A model predicting "non-default" for almost everyone may achieve high accuracy while being operationally useless.

Useful metrics include:

- ROC AUC for ranking ability;
- precision-recall AUC when defaults are relatively rare;
- recall and precision at an operational threshold;
- confusion matrix;
- Brier score for probability calibration;
- calibration curves;
- business-cost or expected-loss metrics.

## Probability calibration

Credit decisions frequently depend on predicted probabilities rather than class labels. A model can rank borrowers well while producing poorly calibrated probabilities. If a group of loans receives an average predicted PD of 10%, approximately 10% should default over the relevant horizon for the probability estimate to be well calibrated.

Calibration should therefore be evaluated separately from discrimination.

## From probability of default to expected loss

A simplified expected credit loss relationship is

\[
EL = PD \times LGD \times EAD,
\]

where:

- \(PD\) is probability of default;
- \(LGD\) is loss given default;
- \(EAD\) is exposure at default.

A classification model typically estimates only one part of this framework.

## Interpretability, fairness, and governance

Credit models operate in a regulated, high-impact domain. Model development should therefore include:

- documented feature definitions;
- leakage controls;
- reason-code or explainability procedures where required;
- performance monitoring by time period and relevant subpopulation;
- fairness and disparate-impact analysis appropriate to the jurisdiction;
- drift monitoring;
- model-version control and reproducibility;
- human review and policy controls where appropriate.

Protected attributes should not be casually included as predictive inputs. Simply removing a protected attribute is also not sufficient to guarantee fairness because proxy variables may remain.

## Why the included Random Forest is not scaled

Tree-based models do not require standardization in the same way that distance-based or gradient-based models often do. The example therefore imputes numerical values but does not apply `StandardScaler` before the Random Forest.

Categorical variables are encoded inside a `ColumnTransformer`, and all preprocessing is fitted only on the training period through a scikit-learn `Pipeline`.

## Included code

`ml_credit_risk_assessment.py` demonstrates:

- synthetic credit application data;
- interpretable engineered ratios;
- missing-value handling;
- categorical encoding;
- out-of-time validation;
- class-weighted Random Forest modeling;
- ROC AUC and precision-recall AUC;
- Brier score;
- threshold-based evaluation.

The example intentionally excludes real customer data and protected personal attributes. It is educational material and is not a production underwriting system.
