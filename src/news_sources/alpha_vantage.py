"""
Alpha Vantage News Sentiment API fetcher.

Free tier: 25 requests/day.
Set ALPHA_VANTAGE_API_KEY in your .env file.
If the key is missing, this source is silently skipped.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request as UrlRequest, urlopen

from src.news_sources.filters import is_relevant, relevance_terms

AV_NEWS_URL = "https://www.alphavantage.co/query"
AV_TICKERS = "SPY,QQQ,GLD,TLT,DIA"   # broad market ETFs for financial coverage
AV_TOPICS  = "financial_markets,economy_fiscal,economy_monetary,economy_macro"


def fetch(days: int = 1) -> tuple[list[dict], list[str]]:
    """
    Fetch news from Alpha Vantage for the past `days` days.
    Returns (records, errors).
    Silently returns ([], []) if API key is not configured.
    """
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY", "").strip()
    if not api_key:
        return [], []

    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=days)

    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": AV_TICKERS,
        "topics": AV_TOPICS,
        "time_from": start_dt.strftime("%Y%m%dT%H%M"),
        "time_to": end_dt.strftime("%Y%m%dT%H%M"),
        "limit": 200,
        "apikey": api_key,
    }
    request = UrlRequest(
        f"{AV_NEWS_URL}?{urlencode(params)}",
        headers={"User-Agent": "Mozilla/5.0 (compatible; FinBERTDashboard/1.0)"},
    )

    try:
        with urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8", errors="replace"))
    except URLError as exc:
        return [], [f"alpha_vantage:{exc.reason}"]
    except json.JSONDecodeError:
        return [], ["alpha_vantage:non-JSON response"]

    if "Information" in payload:
        # Rate limit message from Alpha Vantage
        return [], [f"alpha_vantage:{payload['Information']}"]

    seen_titles: set[str] = set()
    records: list[dict] = []

    for item in payload.get("feed", []):
        title = (item.get("title") or "").strip()
        if not title or title in seen_titles:
            continue

        terms = relevance_terms(title)
        if not terms or not is_relevant(title):
            continue

        seen_titles.add(title)
        published_at = _parse_av_date(item.get("time_published"))
        source = (item.get("source") or "alpha_vantage").lower().replace(" ", "_")

        records.append({
            "headline": title,
            "original_headline": title,
            "link": item.get("url") or "",
            "published_at": published_at,
            "source": source,
            "feed_type": "alpha_vantage",
            "query_bucket": None,
            "language": "en",
            "was_translated": False,
            "relevance_terms": terms,
        })

    records.sort(key=lambda r: r.get("published_at") or "", reverse=True)
    return records, []


def _parse_av_date(raw: str | None) -> str | None:
    """Parse Alpha Vantage date format: 20240315T143000."""
    if not raw:
        return None
    try:
        dt = datetime.strptime(raw, "%Y%m%dT%H%M%S")
        return dt.replace(tzinfo=timezone.utc).isoformat()
    except ValueError:
        return None
