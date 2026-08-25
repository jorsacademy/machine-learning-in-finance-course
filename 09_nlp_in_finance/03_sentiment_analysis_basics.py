"""Baseline financial sentiment classification with TF-IDF and logistic regression."""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline


def dataset() -> pd.DataFrame:
    rows = [
        ("Company beats earnings expectations and raises guidance", 1),
        ("Margins improve as demand remains strong", 1),
        ("Revenue growth accelerates after product launch", 1),
        ("Company misses earnings and cuts full-year outlook", 0),
        ("Regulator opens investigation into accounting practices", 0),
        ("Weak demand forces management to lower guidance", 0),
        ("Free cash flow reaches a record high", 1),
        ("Unexpected loss and rising debt pressure shares", 0),
        ("Management announces larger dividend and buyback", 1),
        ("Factory shutdown creates material supply disruption", 0),
    ]
    return pd.DataFrame(rows, columns=["text", "positive"])


def main() -> None:
    df = dataset()
    train = df.iloc[:8]
    test = df.iloc[8:]

    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )
    model.fit(train["text"], train["positive"])
    predictions = model.predict(test["text"])
    probabilities = model.predict_proba(test["text"])[:, 1]

    print(pd.DataFrame({"text": test["text"], "p_positive": probabilities, "prediction": predictions}))
    print("\n", classification_report(test["positive"], predictions, zero_division=0))
    print("This tiny dataset is pedagogical. Real financial sentiment requires larger, time-aware labeled data.")


if __name__ == "__main__":
    main()
