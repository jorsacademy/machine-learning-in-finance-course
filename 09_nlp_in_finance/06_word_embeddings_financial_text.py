"""Educational word embeddings from a term-document representation.

This example uses TF-IDF followed by TruncatedSVD to obtain dense document and
term representations without requiring an external pretrained embedding model.
"""

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def corpus() -> list[str]:
    return [
        "earnings revenue profit margins guidance",
        "quarterly profit sales earnings outlook",
        "interest rates inflation monetary policy bonds",
        "central bank rates yields inflation",
        "credit losses defaults lending bank capital",
        "loan delinquencies credit risk bank provisions",
        "oil energy demand crude commodity prices",
        "natural gas energy supply commodity market",
    ]


def main() -> None:
    vectorizer = TfidfVectorizer()
    x = vectorizer.fit_transform(corpus())

    svd = TruncatedSVD(n_components=4, random_state=42)
    document_embeddings = svd.fit_transform(x)
    term_embeddings = svd.components_.T
    terms = vectorizer.get_feature_names_out()

    query = "earnings"
    query_idx = int(np.where(terms == query)[0][0])
    similarities = cosine_similarity(term_embeddings[[query_idx]], term_embeddings).ravel()
    nearest = similarities.argsort()[::-1][1:6]

    print("Document embedding shape:", document_embeddings.shape)
    print(f"Terms most similar to '{query}' in this toy corpus:")
    for idx in nearest:
        print(f"  {terms[idx]:15s} {similarities[idx]:.3f}")

    print("\nDense vectors depend heavily on the training corpus and should be validated for the intended domain.")


if __name__ == "__main__":
    main()
