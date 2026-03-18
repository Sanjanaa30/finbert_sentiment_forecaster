"""Yahoo Finance RSS feed fetcher — no API key required."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.error import URLError
from urllib.request import Request as UrlRequest, urlopen

from src.news_sources.filters import is_relevant, relevance_terms

YAHOO_RSS_FEEDS = [
    ("https://finance.yahoo.com/news/rssindex", "yahoo_finance"),
    ("https://finance.yahoo.com/rss/topfinstories", "yahoo_top_finance"),
]


def fetch() -> tuple[list[dict], list[str]]:
    """
    Fetch headlines from Yahoo Finance RSS feeds.
    Returns (records, errors).
    """
    seen_titles: set[str] = set()
    records: list[dict] = []
    errors: list[str] = []

    for url, feed_type in YAHOO_RSS_FEEDS:
        request = UrlRequest(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; FinBERTDashboard/1.0)"},
        )
        try:
            with urlopen(request, timeout=15) as response:
                raw_body = response.read()
        except URLError as exc:
            errors.append(f"{feed_type}:{exc.reason}")
            continue

        try:
            root = ET.fromstring(raw_body)
        except ET.ParseError as exc:
            errors.append(f"{feed_type}:xml parse error ({exc})")
            continue

        for item in root.iter("item"):
            title = (item.findtext("title") or "").strip()
            if not title or title in seen_titles:
                continue

            terms = relevance_terms(title)
            if not terms or not is_relevant(title):
                continue

            seen_titles.add(title)
            pub_date = _parse_rss_date(item.findtext("pubDate"))
            records.append({
                "headline": title,
                "original_headline": title,
                "link": item.findtext("link") or "",
                "published_at": pub_date,
                "source": "finance.yahoo.com",
                "feed_type": feed_type,
                "query_bucket": None,
                "language": "en",
                "was_translated": False,
                "relevance_terms": terms,
            })

    records.sort(key=lambda r: r.get("published_at") or "", reverse=True)
    return records, errors


def _parse_rss_date(raw: str | None) -> str | None:
    if not raw:
        return None
    try:
        return parsedate_to_datetime(raw).astimezone(timezone.utc).isoformat()
    except Exception:
        return None
