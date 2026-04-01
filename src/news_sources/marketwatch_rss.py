"""MarketWatch / CNBC / Reuters / Investing.com RSS feeds — no API key required."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.error import URLError
from urllib.request import Request as UrlRequest, urlopen


RSS_FEEDS = [
    ("https://feeds.content.dowjones.io/public/rss/mw_topstories", "marketwatch_top"),
    ("https://feeds.content.dowjones.io/public/rss/mw_marketpulse", "marketwatch_pulse"),
    ("https://feeds.content.dowjones.io/public/rss/mw_realtimeheadlines", "marketwatch_realtime"),
    ("https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114", "cnbc_finance"),
    ("https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664", "cnbc_economy"),
    ("https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=15839069", "cnbc_markets"),
    ("https://www.investing.com/rss/news.rss", "investing_com"),
    ("https://www.reutersagency.com/feed/?taxonomy=best-sectors&post_type=best", "reuters_best"),
]


def fetch() -> tuple[list[dict], list[str]]:
    """
    Fetch headlines from multiple financial RSS feeds.
    Returns (records, errors).
    """
    seen_titles: set[str] = set()
    records: list[dict] = []
    errors: list[str] = []

    for url, feed_type in RSS_FEEDS:
        request = UrlRequest(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; FinBERTDashboard/1.0)"},
        )
        try:
            with urlopen(request, timeout=15) as response:
                raw_body = response.read()
        except (URLError, Exception) as exc:
            errors.append(f"{feed_type}:{exc}")
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

            seen_titles.add(title)
            pub_date = _parse_rss_date(item.findtext("pubDate"))
            source_domain = feed_type.split("_")[0]
            records.append({
                "headline": title,
                "original_headline": title,
                "link": item.findtext("link") or "",
                "published_at": pub_date,
                "source": source_domain,
                "feed_type": feed_type,
                "query_bucket": None,
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
