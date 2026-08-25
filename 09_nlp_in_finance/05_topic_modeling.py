"""Topic modeling of financial documents with Latent Dirichlet Allocation."""

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


def corpus() -> list[str]:
    return [
        "central bank interest rates inflation monetary policy bond yields",
        "rate cuts inflation expectations treasury yields monetary policy",
        "company earnings revenue margins guidance operating profit",
        "quarterly earnings sales growth margins management guidance",
        "oil prices crude production refinery energy demand commodity",
        "energy producers oil supply demand natural gas commodity prices",
        "bank lending credit losses deposits capital ratios loan growth",
        "consumer credit delinquency lending standards bank deposits",
    ]


def show_topics(model: LatentDirichletAllocation, feature_names, top_n: int = 6) -> None:
    for topic_id, weights in enumerate(model.components_):
        top_indices = weights.argsort()[-top_n:][::-1]
        terms = [feature_names[i] for i in top_indices]
        print(f"Topic {topic_id}: {', '.join(terms)}")


def main() -> None:
    vectorizer = CountVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(corpus())

    lda = LatentDirichletAllocation(n_components=4, random_state=42, learning_method="batch")
    lda.fit(matrix)
    show_topics(lda, vectorizer.get_feature_names_out())

    print("\nTopic labels require human interpretation; topic IDs have no intrinsic economic meaning.")


if __name__ == "__main__":
    main()
