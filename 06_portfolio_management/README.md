# Clustering for Portfolio Diversification

Clustering can support portfolio construction by grouping assets that exhibit similar behavior. The objective is not to let a clustering algorithm choose a portfolio automatically. Instead, clustering is used as a structural tool to reduce redundancy and reveal groups of assets that may share common return drivers.

## Why clustering can help

A portfolio containing many highly similar assets can still be poorly diversified. Clustering can identify this hidden concentration by grouping assets according to return behavior, risk characteristics, factor exposures, or correlation structure.

Common applications include:

- correlation-based clustering of asset returns;
- clustering by expected return and volatility;
- grouping assets by factor exposures;
- identifying unusual or isolated assets;
- comparing how asset relationships change across market regimes.

## K-Means approach

K-Means is useful when each asset can be represented by a fixed feature vector such as annualized return, volatility, downside volatility, beta, momentum, or factor exposures.

Because K-Means uses Euclidean distance, features should generally be standardized before fitting the model. Otherwise a feature with a larger numerical scale can dominate the clustering result.

A simplified objective is

\[
\min_{C_1,\ldots,C_K} \sum_{k=1}^{K} \sum_{x_i \in C_k} \lVert x_i - \mu_k \rVert^2,
\]

where \(\mu_k\) is the centroid of cluster \(C_k\).

The included example uses the silhouette score as a simple diagnostic for choosing a demonstration value of \(K\). This is not a guarantee that the selected clustering is economically meaningful.

## Correlation-based hierarchical clustering

For diversification, correlation is often more meaningful than Euclidean distance between raw price series. A common correlation distance is

\[
d_{ij} = \sqrt{\frac{1 - \rho_{ij}}{2}},
\]

where \(\rho_{ij}\) is the return correlation between assets \(i\) and \(j\).

The implementation in this repository converts the correlation matrix to a valid condensed distance matrix before applying hierarchical clustering. This is preferable to passing rows of a correlation matrix directly to Ward linkage, which does not represent the intended correlation-distance problem.

## Interpreting clusters

A cluster is not automatically an economic sector or a stable risk factor. After clustering, inspect:

- cluster-level correlations;
- sector and industry composition;
- factor exposures;
- liquidity and trading costs;
- cluster stability across rolling windows;
- behavior during stressed market periods.

Selecting one asset from each cluster is only a heuristic. Real portfolio construction should still consider expected returns, covariance, concentration limits, liquidity, turnover, transaction costs, and risk constraints.

## Dynamic relationships

Asset relationships are non-stationary. A clustering learned from one period may not remain valid in another. In practice, clustering should be re-estimated periodically and its stability should be measured rather than assumed.

## Advanced extensions

Useful extensions include PCA before clustering, rolling-window clustering, factor-based clustering, regime-conditioned clustering, spectral clustering, and clustering based on dynamic time warping for shape similarity. PCA can be useful when the feature set is highly correlated, but it changes interpretability because clusters are then formed in component space rather than the original feature space.

Reinforcement learning should not be described as a clustering method. It can use cluster-derived state variables or portfolio constraints, but clustering and reinforcement learning solve different problems.

## Included code

`clustering_portfolio_diversification.py` demonstrates:

- reproducible simulated asset prices;
- annualized return and risk features;
- feature standardization;
- K-Means clustering;
- silhouette analysis;
- correlation-distance hierarchical clustering;
- a simple cluster-representative heuristic;
- cluster visualizations.

The material is educational and does not constitute investment advice.
