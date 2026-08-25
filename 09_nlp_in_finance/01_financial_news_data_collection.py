"""Financial news collection with timestamp and provenance controls.

The example intentionally separates collection from modeling. Real news feeds may
require licenses, authentication, rate limits, and redistribution restrictions.
"""

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import xml.etree.ElementTree as ET

import pandas as pd


@dataclass
class NewsItem:
    published_at: str
    title: str
    link: str
    source: str


def parse_rss(xml_text: str, source: str) -> list[NewsItem]:
    root = ET.fromstring(xml_text)
    items: list[NewsItem] = []

    for item in root.findall(".//item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub_date = (item.findtext("pubDate") or "").strip()
        if title:
            items.append(NewsItem(pub_date, title, link, source))
    return items


def sample_feed() -> str:
    now = datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S GMT")
    return f"""<?xml version='1.0'?>
    <rss version='2.0'><channel>
      <title>Educational Finance Feed</title>
      <item><title>Company raises full-year revenue guidance</title>
      <link>https://example.com/a</link><pubDate>{now}</pubDate></item>
      <item><title>Central bank keeps policy rate unchanged</title>
      <link>https://example.com/b</link><pubDate>{now}</pubDate></item>
    </channel></rss>"""


def main() -> None:
    records = parse_rss(sample_feed(), source="educational_sample")
    df = pd.DataFrame(asdict(item) for item in records)
    print(df)
    print("\nStore source, publication timestamp, ingestion timestamp, and license metadata in real pipelines.")


if __name__ == "__main__":
    main()
