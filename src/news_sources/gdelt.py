"""GDELT DOC API v2 news fetcher."""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from json import JSONDecodeError
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request as UrlRequest, urlopen

from src.news_sources.filters import is_relevant, relevance_terms

GDELT_API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_QUERIES = (
    '("S&P 500" OR SPY OR Nasdaq OR "Dow Jones" OR "stock index" OR indexes OR futures)',
    '("Federal Reserve" OR fed OR inflation OR rates OR yields OR treasury OR bonds OR recession OR economy)',
    '(earnings OR guidance OR "short interest" OR rally OR selloff OR volatility OR oil OR gold OR stocks OR shares OR market)',
)
_SLEEP_BETWEEN_QUERIES = 3  # seconds — avoids GDELT rate limits


def fetch(
    start_dt: datetime,
    end_dt: datetime,
    max_records: int = 80,
) -> tuple[list[dict], list[str]]:
    """
    Fetch financial headlines from GDELT for the given UTC window.
    Returns (records, errors).
    """
    seen_titles: set[str] = set()
    records: list[dict] = []
    errors: list[str] = []

    for query in GDELT_QUERIES:
        params = {
            "query": query,
            "mode": "ArtList",
            "format": "json",
            "maxrecords": min(int(max_records), 250),
            "sort": "datedesc",
            "startdatetime": start_dt.strftime("%Y%m%d%H%M%S"),
            "enddatetime": end_dt.strftime("%Y%m%d%H%M%S"),
        }
        request = UrlRequest(
            f"{GDELT_API_URL}?{urlencode(params)}",
            headers={"User-Agent": "Mozilla/5.0 (compatible; FinBERTDashboard/1.0)"},
        )
        try:
            with urlopen(request, timeout=20) as response:
                raw_body = response.read().decode("utf-8", errors="replace")
                payload = json.loads(raw_body)
        except URLError as exc:
            errors.append(f"gdelt:{exc.reason}")
            time.sleep(_SLEEP_BETWEEN_QUERIES)
            continue
        except JSONDecodeError:
            snippet = (raw_body[:120].replace("\n", " ").strip()
                       if raw_body else "empty response")
            errors.append(f"gdelt:non-JSON ({snippet})")
            time.sleep(_SLEEP_BETWEEN_QUERIES)
            continue

        for article in payload.get("articles", []):
            title = (article.get("title") or "").strip()
            if not title or title in seen_titles:
                continue
            seen_titles.add(title)

            published_raw = article.get("seendate")
            published_at = _parse_datetime(published_raw)

            records.append({
                "headline": title,
                "original_headline": title,
                "link": article.get("url") or "",
                "published_at": published_at,
                "source": article.get("domain") or "gdelt",
                "feed_type": "gdelt",
                "query_bucket": query,
                "language": "en",
                "was_translated": False,
                "relevance_terms": relevance_terms(title),
            })

        time.sleep(_SLEEP_BETWEEN_QUERIES)

    # Filter noise
    records = [r for r in records if is_relevant(r["headline"])]
    records.sort(key=lambda r: r.get("published_at") or "", reverse=True)
    return records, errors


def _parse_datetime(raw: str | None) -> str | None:
    if not raw:
        return None
    try:
        return (datetime.fromisoformat(raw.replace("Z", "+00:00"))
                .astimezone(timezone.utc).isoformat())
    except ValueError:
        pass
    try:
        parsed = datetime.strptime(raw, "%Y%m%d%H%M%S")
        return parsed.replace(tzinfo=timezone.utc).isoformat()
    except ValueError:
        return None
