"""Text preprocessing for financial NLP.

Avoid aggressive preprocessing that removes negation, numbers, ticker symbols,
or domain-specific terms that can carry financial meaning.
"""

import re
import unicodedata


FINANCE_STOPWORDS = {"the", "a", "an", "and", "or", "of", "to", "in", "for", "on"}


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"https?://\S+", " <URL> ", text)
    text = re.sub(r"\$([A-Za-z]{1,5})\b", r" TICKER_\1 ", text)
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    return re.findall(r"TICKER_[A-Z]+|<URL>|[A-Za-z]+(?:'[A-Za-z]+)?|[-+]?\d+(?:\.\d+)?%?", text)


def lightly_filter(tokens: list[str]) -> list[str]:
    result = []
    for token in tokens:
        lower = token.lower()
        # Keep negations because "not profitable" differs from "profitable".
        if lower in FINANCE_STOPWORDS and lower not in {"not", "no"}:
            continue
        result.append(token)
    return result


def main() -> None:
    headline = "$AAPL revenue rose 12.5%, but management did not raise guidance. https://example.com"
    normalized = normalize_text(headline)
    tokens = lightly_filter(tokenize(normalized))
    print("Original:  ", headline)
    print("Normalized:", normalized)
    print("Tokens:    ", tokens)


if __name__ == "__main__":
    main()
