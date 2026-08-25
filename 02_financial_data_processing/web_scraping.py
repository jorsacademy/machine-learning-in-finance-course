"""Minimal, respectful web-scraping example for financial pages.

Before scraping any website, review its terms of service, robots policy, and
applicable law. Prefer an official API when one is available.
"""

from bs4 import BeautifulSoup
import requests


def fetch_title(url: str) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; JorsAcademyEducationalExample/1.0; "
            "+https://github.com/jorsacademy/machine-learning-in-finance-course)"
        )
    }
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else "No title found"
    return title


def main() -> None:
    url = "https://www.sec.gov/"
    print(fetch_title(url))


if __name__ == "__main__":
    main()
