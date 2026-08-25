"""Optional transformer-based sentiment inference for financial text.

This lesson uses a pretrained finance-domain classifier when the optional
`transformers` and `torch` packages are installed. Model downloads require
internet access and are subject to the model author's license.
"""


def main() -> None:
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise SystemExit(
            "Install optional NLP dependencies with: pip install transformers torch"
        ) from exc

    classifier = pipeline(
        "text-classification",
        model="ProsusAI/finbert",
        tokenizer="ProsusAI/finbert",
    )

    texts = [
        "The company raised full-year earnings guidance after strong demand.",
        "Management warned that margins will fall because of weak orders.",
        "The board declared an unchanged quarterly dividend.",
    ]

    for text, result in zip(texts, classifier(texts), strict=True):
        print(text)
        print(result)
        print()

    print("Do not assume pretrained sentiment labels directly imply tradable returns.")


if __name__ == "__main__":
    main()
