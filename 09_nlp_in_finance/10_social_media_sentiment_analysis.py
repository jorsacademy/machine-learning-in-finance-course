"""Social-media sentiment baseline for financial text.

This example emphasizes source quality, timestamp integrity, manipulation risk,
and aggregation rather than treating every post as an equally reliable signal.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def labeled_posts() -> pd.DataFrame:
    data = [
        ("Strong demand and raised guidance look positive", 1),
        ("Margins expanding and cash flow improving", 1),
        ("Debt pressure and weak orders are concerning", 0),
        ("Guidance cut again after another earnings miss", 0),
        ("New product launch is gaining traction", 1),
        ("Liquidity concerns are getting worse", 0),
        ("Buyback increased after record free cash flow", 1),
        ("Customer churn accelerated this quarter", 0),
    ]
    return pd.DataFrame(data, columns=["text", "positive"])


def score_posts() -> pd.DataFrame:
    train = labeled_posts()
    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )
    model.fit(train["text"], train["positive"])

    posts = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-01-02 13:00", "2026-01-02 13:04", "2026-01-02 13:08", "2026-01-02 13:11"],
                utc=True,
            ),
            "account": ["analyst_a", "new_account_1", "analyst_b", "new_account_2"],
            "text": [
                "Demand appears stronger and guidance may improve",
                "BUY BUY BUY guaranteed moon tomorrow",
                "Debt levels remain elevated despite better sales",
                "This stock will triple instantly trust me",
            ],
            "reliability_weight": [1.0, 0.15, 0.9, 0.10],
        }
    )

    posts["p_positive"] = model.predict_proba(posts["text"])[:, 1]
    posts["sentiment_score"] = 2.0 * posts["p_positive"] - 1.0
    posts["weighted_sentiment"] = posts["sentiment_score"] * posts["reliability_weight"]
    return posts


def main() -> None:
    posts = score_posts()
    weighted_average = np.average(posts["sentiment_score"], weights=posts["reliability_weight"])
    print(posts)
    print(f"\nReliability-weighted sentiment: {weighted_average:.3f}")
    print(
        "Social-media data is vulnerable to bots, coordinated manipulation, duplicate content, "
        "selection bias, and changing platform populations. These risks must be modeled explicitly."
    )


if __name__ == "__main__":
    main()
