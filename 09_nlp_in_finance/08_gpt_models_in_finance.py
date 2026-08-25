"""Prompt construction patterns for GPT-style models in finance.

The example does not call an external API. It focuses on reproducible prompt
structure, explicit task constraints, and separation of source facts from model
output. Never send confidential client or regulated data to an external model
without approved governance and data-handling controls.
"""

from dataclasses import dataclass


@dataclass
class FinancialDocument:
    source: str
    published_at: str
    text: str


def build_extraction_prompt(document: FinancialDocument) -> str:
    return f"""You are extracting facts from a financial document.

Rules:
- Use only the supplied document.
- Do not infer missing numbers.
- Distinguish reported facts from management guidance.
- Return concise JSON with keys: revenue_comment, margin_comment, guidance_comment, risks.
- If information is absent, return null.

Source: {document.source}
Published at: {document.published_at}

Document:
{document.text}
"""


def main() -> None:
    document = FinancialDocument(
        source="educational_sample",
        published_at="2026-01-15T13:00:00Z",
        text=(
            "Quarterly revenue increased 8%. Operating margin declined by 1.2 percentage points. "
            "Management maintained full-year revenue guidance and cited foreign-exchange risk."
        ),
    )
    print(build_extraction_prompt(document))
    print("\nIn production, validate structured outputs against a schema and preserve source citations.")


if __name__ == "__main__":
    main()
