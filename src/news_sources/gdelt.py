"""GDELT DOC API v2 news fetcher."""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timedelta, timezone
from json import JSONDecodeError
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request as UrlRequest, urlopen


GDELT_API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_QUERY = (
    '("S&P 500" OR SPY OR Nasdaq OR "Dow Jones" OR futures OR '
    '"Federal Reserve" OR inflation OR rates OR treasury OR recession OR '
    'earnings OR rally OR selloff OR volatility OR stocks OR market)'
)
_MIN_INTERVAL = 5  # minimum seconds between any two GDELT requests (global)
_rate_lock = threading.Lock()
_last_call_time: float = 0.0


def _gdelt_sleep() -> None:
    """Ensure at least _MIN_INTERVAL seconds between consecutive GDELT calls."""
    global _last_call_time
    with _rate_lock:
        elapsed = time.monotonic() - _last_call_time
        if elapsed < _MIN_INTERVAL:
            time.sleep(_MIN_INTERVAL - elapsed)
        _last_call_time = time.monotonic()


def fetch(
    start_dt: datetime,
    end_dt: datetime,
    max_records: int = 80,
) -> tuple[list[dict], list[str]]:
    """
    Fetch financial headlines from GDELT for the given UTC window.
    Returns (records, errors).
    """
    records: list[dict] = []
    errors: list[str] = []

    _gdelt_sleep()
    params = {
        "query": GDELT_QUERY,
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
        return records, errors
    except JSONDecodeError:
        snippet = (raw_body[:120].replace("\n", " ").strip() if raw_body else "empty response")
        errors.append(f"gdelt:non-JSON ({snippet})")
        return records, errors

    seen_titles: set[str] = set()
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
            "query_bucket": GDELT_QUERY,
            "language": "en",
            "was_translated": False,
        })

    # Reject articles outside the requested date window
    start_iso = start_dt.astimezone(timezone.utc).isoformat()
    end_iso = end_dt.astimezone(timezone.utc).isoformat()
    records = [
        r for r in records
        if r.get("published_at") and start_iso <= r["published_at"] <= end_iso
    ]

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
