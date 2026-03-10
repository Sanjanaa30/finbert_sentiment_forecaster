from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request as UrlRequest, urlopen
from urllib.error import URLError

import torch


GDELT_QUERY = '("stock market" OR "S&P 500" OR SPY OR "Federal Reserve" OR inflation OR Nasdaq OR "Dow Jones")'
GDELT_API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"


def _market_mood(score: float) -> str:
    if score > 0.02:
        return "Positive"
    if score < -0.02:
        return "Negative"
    return "Neutral"


def _parse_datetime(raw_value: str | None) -> str | None:
    if not raw_value:
        return None
    try:
        return datetime.fromisoformat(raw_value.replace("Z", "+00:00")).astimezone(timezone.utc).isoformat()
    except ValueError:
        pass
    try:
        parsed = datetime.strptime(raw_value, "%a, %d %b %Y %H:%M:%S %Z")
        return parsed.replace(tzinfo=timezone.utc).isoformat()
    except ValueError:
        return None


def fetch_gdelt_headlines(days: int, max_records: int = 100) -> list[dict]:
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=days)
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
    with urlopen(request, timeout=20) as response:
        payload = json.loads(response.read().decode("utf-8"))

    seen_titles = set()
    records = []
    for article in payload.get("articles", []):
        title = (article.get("title") or "").strip()
        if not title or title in seen_titles:
            continue
        seen_titles.add(title)
        records.append(
            {
                "headline": title,
                "link": article.get("url") or "",
                "published_at": _parse_datetime(article.get("seendate")),
                "source": article.get("domain") or "gdelt",
                "feed_type": "live_gdelt",
            }
        )
    return records


def score_headlines(records: list[dict], tokenizer, model) -> list[dict]:
    if not records:
        return []

    enc = tokenizer(
        [record["headline"] for record in records],
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt",
    )
    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits, dim=-1).cpu().numpy()

    labels = ["negative", "neutral", "positive"]
    scored_records = []
    for record, prob in zip(records, probs):
        p_neg, p_neu, p_pos = float(prob[0]), float(prob[1]), float(prob[2])
        scored_records.append(
            {
                **record,
                "label": labels[int(prob.argmax())],
                "score": p_pos - p_neg,
                "negative_prob": p_neg,
                "neutral_prob": p_neu,
                "positive_prob": p_pos,
            }
        )
    return scored_records


def summarize_scored(records: list[dict], window_label: str) -> dict:
    if not records:
        return {
            "window": window_label,
            "sentiment_index": None,
            "market_mood": "Unavailable",
            "headlines_analyzed": 0,
            "positive_share": None,
            "neutral_share": None,
            "negative_share": None,
            "latest_published_at": None,
            "top_positive_headline": None,
            "top_negative_headline": None,
            "sample_headlines": [],
            "feed_type": None,
        }

    scores = [record["score"] for record in records]
    labels = [record["label"] for record in records]
    sentiment_index = sum(scores) / len(scores)
    latest_published = max((record.get("published_at") for record in records if record.get("published_at")), default=None)
    sorted_positive = sorted(records, key=lambda item: item["score"], reverse=True)
    sorted_negative = sorted(records, key=lambda item: item["score"])

    return {
        "window": window_label,
        "sentiment_index": sentiment_index,
        "market_mood": _market_mood(sentiment_index),
        "headlines_analyzed": len(records),
        "positive_share": labels.count("positive") / len(labels),
        "neutral_share": labels.count("neutral") / len(labels),
        "negative_share": labels.count("negative") / len(labels),
        "latest_published_at": latest_published,
        "top_positive_headline": sorted_positive[0]["headline"],
        "top_negative_headline": sorted_negative[0]["headline"],
        "sample_headlines": records[:10],
        "feed_type": records[0].get("feed_type"),
    }


def build_live_snapshot(tokenizer, model, model_version: str, output_path: Path | None = None) -> dict:
    windows_config = {
        "1d": {"days": 1, "max_records": 40},
        "7d": {"days": 7, "max_records": 100},
        "30d": {"days": 30, "max_records": 150},
    }

    window_summaries = {}
    gdelt_error = None

    for window_label, cfg in windows_config.items():
        try:
            records = fetch_gdelt_headlines(cfg["days"], max_records=cfg["max_records"])
        except URLError as exc:
            gdelt_error = str(exc.reason)
            records = []

        scored = score_headlines(records, tokenizer, model)
        window_summaries[window_label] = summarize_scored(scored, window_label)

    live_window = window_summaries["1d"]
    snapshot = {
        "sentiment_index": live_window["sentiment_index"],
        "market_mood": live_window["market_mood"],
        "headlines_analyzed": live_window["headlines_analyzed"],
        "latest_update": datetime.now(timezone.utc).date().isoformat(),
        "latest_published_at": live_window["latest_published_at"],
        "positive_share": live_window["positive_share"],
        "neutral_share": live_window["neutral_share"],
        "negative_share": live_window["negative_share"],
        "top_positive_headline": live_window["top_positive_headline"],
        "top_negative_headline": live_window["top_negative_headline"],
        "sample_headlines": live_window["sample_headlines"],
        "feed_type": live_window["feed_type"],
        "model_version": model_version,
        "source_policy": {
            "primary_source": "GDELT",
            "fallback_source": None,
            "gdelt_status": "succeeded" if not gdelt_error else "failed",
        },
        "window_summaries": window_summaries,
        "gdelt_error": gdelt_error,
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")

    return snapshot
