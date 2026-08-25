"""Simple financial named-entity extraction using transparent rules.

For production work, use a trained NER model and evaluate it on finance-specific
annotations. This example is deliberately interpretable for teaching.
"""

import re
import pandas as pd


COMPANY_SUFFIXES = r"Inc\.?|Corp\.?|Corporation|Ltd\.?|PLC|Bank|Holdings"


def extract_entities(text: str) -> list[dict[str, str]]:
    entities: list[dict[str, str]] = []

    for ticker in re.findall(r"\$([A-Z]{1,5})\b", text):
        entities.append({"entity": ticker, "label": "TICKER"})

    company_pattern = rf"\b([A-Z][A-Za-z&.-]*(?:\s+[A-Z][A-Za-z&.-]*)*\s+(?:{COMPANY_SUFFIXES}))\b"
    for company in re.findall(company_pattern, text):
        entities.append({"entity": company, "label": "ORGANIZATION"})

    for pct in re.findall(r"[-+]?\d+(?:\.\d+)?%", text):
        entities.append({"entity": pct, "label": "PERCENT"})

    for amount in re.findall(r"\$\d+(?:\.\d+)?\s?(?:million|billion|trillion|m|bn)?", text, flags=re.I):
        entities.append({"entity": amount, "label": "MONEY"})

    return entities


def main() -> None:
    text = "Acme Holdings raised revenue guidance by 8% after $ACME reported $2.4 billion in quarterly sales."
    print(pd.DataFrame(extract_entities(text)))
    print("\nRule-based extraction is a baseline, not a substitute for validated domain NER.")


if __name__ == "__main__":
    main()
