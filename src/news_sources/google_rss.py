"""Google News RSS fetcher — no API key required."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request as UrlRequest, urlopen


_BASE = "https://news.google.com/rss/search"
_QUERIES = [
    "stock market S&P 500 Nasdaq earnings",
    "Federal Reserve inflation interest rates economy",
]


def fetch() -> tuple[list[dict], list[str]]:
    """
    Fetch financial headlines from Google News RSS.
    Returns (records, errors).
    """
    seen_titles: set[str] = set()
    records: list[dict] = []
    errors: list[str] = []

    for query in _QUERIES:
        params = urlencode({"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"})
        request = UrlRequest(
            f"{_BASE}?{params}",
            headers={"User-Agent": "Mozilla/5.0 (compatible; FinBERTDashboard/1.0)"},
        )
        try:
            with urlopen(request, timeout=15) as response:
                raw_body = response.read()
        except URLError as exc:
            errors.append(f"google_rss:{exc.reason}")
            continue

        try:
            root = ET.fromstring(raw_body)
        except ET.ParseError as exc:
            errors.append(f"google_rss:xml parse error ({exc})")
            continue

        for item in root.iter("item"):
            title = (item.findtext("title") or "").strip()
            # Google RSS appends " - Source Name" — strip it
            if " - " in title:
                title = title.rsplit(" - ", 1)[0].strip()
            if not title or title in seen_titles:
                continue

            seen_titles.add(title)
            pub_date = _parse_rss_date(item.findtext("pubDate"))
            link = item.findtext("link") or ""

            records.append({
                "headline": title,
                "original_headline": title,
                "link": link,
                "published_at": pub_date,
                "source": "news.google.com",
                "feed_type": "google_rss",
                "query_bucket": query,
                "language": "en",
                "was_translated": False,
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
