"""Clustering for portfolio diversification.

Educational example showing two defensible approaches:
1. K-Means on standardized asset-level risk/return features.
2. Hierarchical clustering using correlation distance.

The script uses simulated prices so it runs without external data.
Replace the simulated price DataFrame with your own adjusted close data
when applying the workflow to real assets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

TRADING_DAYS = 252
RANDOM_STATE = 42


def simulate_prices(n_assets: int = 12, n_days: int = 756) -> pd.DataFrame:
    """Create reproducible synthetic adjusted-close style prices."""
    rng = np.random.default_rng(RANDOM_STATE)

    # Three latent return drivers create economically meaningful groups.
    factors = rng.normal(0.0003, 0.009, size=(n_days, 3))
    asset_returns = np.empty((n_days, n_assets))

    for i in range(n_assets):
        group = i % 3
        idiosyncratic = rng.normal(0.0, 0.006, size=n_days)
        asset_returns[:, i] = 0.75 * factors[:, group] + 0.25 * idiosyncratic

    prices = 100 * np.exp(np.cumsum(asset_returns, axis=0))
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    columns = [f"Asset_{i + 1:02d}" for i in range(n_assets)]
    return pd.DataFrame(prices, index=dates, columns=columns)


def build_asset_features(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return annualized asset features and daily simple returns."""
    daily_returns = prices.pct_change().dropna()

    annual_return = daily_returns.mean() * TRADING_DAYS
    annual_volatility = daily_returns.std() * np.sqrt(TRADING_DAYS)
    downside = daily_returns.clip(upper=0).std() * np.sqrt(TRADING_DAYS)

    features = pd.DataFrame(
        {
            "annual_return": annual_return,
            "annual_volatility": annual_volatility,
            "downside_volatility": downside,
        }
    )
    return features, daily_returns


def choose_k_with_silhouette(features: pd.DataFrame, k_min: int = 2, k_max: int = 6) -> int:
    """Choose a demonstration value of k using silhouette score."""
    scaled = StandardScaler().fit_transform(features)
    candidates = range(k_min, min(k_max, len(features) - 1) + 1)

    scores: dict[int, float] = {}
    for k in candidates:
        model = KMeans(n_clusters=k, n_init="auto", random_state=RANDOM_STATE)
        labels = model.fit_predict(scaled)
        scores[k] = silhouette_score(scaled, labels)

    best_k = max(scores, key=scores.get)
    print("Silhouette scores:", {k: round(v, 3) for k, v in scores.items()})
    print("Selected k:", best_k)
    return best_k


def kmeans_clustering(features: pd.DataFrame, n_clusters: int) -> pd.DataFrame:
    """Cluster assets after standardizing heterogeneous financial features."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    model = KMeans(n_clusters=n_clusters, n_init="auto", random_state=RANDOM_STATE)
    labels = model.fit_predict(scaled)

    result = features.copy()
    result["kmeans_cluster"] = labels
    return result


def correlation_hierarchical_clustering(
    daily_returns: pd.DataFrame,
    distance_threshold: float = 0.65,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Cluster assets from correlation distance using average linkage.

    Correlation distance is defined as sqrt(0.5 * (1 - rho)).
    This avoids the common mistake of passing correlation-matrix rows
    directly to Ward linkage as though they were Euclidean observations.
    """
    corr = daily_returns.corr().clip(-1, 1)
    distance = np.sqrt(0.5 * (1 - corr))
    np.fill_diagonal(distance.values, 0.0)

    condensed = squareform(distance.values, checks=False)
    linked = linkage(condensed, method="average")
    labels = fcluster(linked, t=distance_threshold, criterion="distance")

    clusters = pd.DataFrame(
        {"hierarchical_cluster": labels},
        index=daily_returns.columns,
    )
    return clusters, linked


def representative_assets(
    feature_clusters: pd.DataFrame,
    daily_returns: pd.DataFrame,
) -> list[str]:
    """Select one simple representative per K-Means cluster.

    The rule chooses the asset with the highest annual-return-to-volatility
    ratio inside each cluster. This is an educational heuristic, not an
    optimization objective or investment recommendation.
    """
    table = feature_clusters.copy()
    table["return_to_volatility"] = (
        table["annual_return"] / table["annual_volatility"].replace(0, np.nan)
    )

    selected = (
        table.sort_values("return_to_volatility", ascending=False)
        .groupby("kmeans_cluster", sort=True)
        .head(1)
        .index.tolist()
    )

    if len(selected) > 1:
        print("Average pairwise correlation of selected assets:")
        corr = daily_returns[selected].corr()
        mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
        print(round(corr.where(mask).stack().mean(), 3))

    return selected


def main() -> None:
    prices = simulate_prices()
    features, daily_returns = build_asset_features(prices)

    best_k = choose_k_with_silhouette(features)
    kmeans_result = kmeans_clustering(features, best_k)

    hierarchical_result, linked = correlation_hierarchical_clustering(daily_returns)
    combined = kmeans_result.join(hierarchical_result)

    print("\nAsset clusters:\n", combined.round(4))

    selected = representative_assets(kmeans_result, daily_returns)
    print("\nIllustrative representatives:", selected)

    plt.figure(figsize=(9, 6))
    plt.scatter(
        combined["annual_volatility"],
        combined["annual_return"],
        c=combined["kmeans_cluster"],
    )
    for asset, row in combined.iterrows():
        plt.annotate(asset, (row["annual_volatility"], row["annual_return"]), fontsize=8)
    plt.xlabel("Annualized volatility")
    plt.ylabel("Annualized return")
    plt.title("K-Means Asset Clusters")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(11, 6))
    dendrogram(linked, labels=daily_returns.columns.tolist())
    plt.ylabel("Correlation distance")
    plt.title("Hierarchical Clustering of Assets")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
