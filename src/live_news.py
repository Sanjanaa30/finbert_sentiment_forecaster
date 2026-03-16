from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request as UrlRequest, urlopen
from urllib.error import URLError
from zoneinfo import ZoneInfo

import torch
from deep_translator import GoogleTranslator
from langdetect import DetectorFactory, LangDetectException, detect


GDELT_QUERIES = (
    '("S&P 500" OR SPY OR Nasdaq OR "Dow Jones" OR "stock index" OR indexes OR futures)',
    '("Federal Reserve" OR fed OR inflation OR rates OR yields OR treasury OR bonds OR recession OR economy)',
    '(earnings OR guidance OR "short interest" OR rally OR selloff OR volatility OR oil OR gold OR stocks OR shares OR market)',
)
GDELT_API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
FINANCE_PATTERNS = {
    "spy": r"\bspy\b",
    "s&p 500": r"\bs\s*&\s*p\s*500\b|\bs&p 500\b",
    "nasdaq": r"\bnasdaq\b",
    "dow": r"\bdow\b|\bdow jones\b",
    "fed": r"\bfed\b",
    "federal reserve": r"\bfederal reserve\b",
    "inflation": r"\binflation\b",
    "rates": r"\brates?\b|\binterest rates?\b",
    "earnings": r"\bearnings?\b",
    "guidance": r"\bguidance\b",
    "stocks": r"\bstocks?\b",
    "shares": r"\bshares?\b",
    "market": r"\bmarkets?\b",
    "wall street": r"\bwall street\b",
    "equities": r"\bequities\b|\bequity\b",
    "index": r"\bindex\b|\bindexes\b|\bindices\b",
    "futures": r"\bfutures\b",
    "etf": r"\betf\b|\betfs\b",
    "bonds": r"\bbonds?\b",
    "central bank": r"\bcentral bank\b",
    "volatility": r"\bvolatility\b",
    "rally": r"\brally\b",
    "selloff": r"\bselloff\b|\bsell-off\b",
    "short interest": r"\bshort interest\b",
    "treasury": r"\btreasury\b|\btreasuries\b",
    "yield": r"\byields?\b",
    "oil": r"\boil\b",
    "gold": r"\bgold\b",
    "tariff": r"\btariffs?\b",
    "recession": r"\brecession\b",
    "economy": r"\beconom(y|ic)\b",
}
WEAK_SCORE_THRESHOLD = 0.03
DetectorFactory.seed = 0
LOCAL_TZ = ZoneInfo("America/New_York")
LIVE_COVERAGE_LABEL = "Global market news"
DEFAULT_CALIBRATION = {
    "strong_negative_max": -0.25,
    "slight_negative_max": -0.05,
    "slight_positive_min": 0.05,
    "strong_positive_min": 0.25,
    "source": "default",
}
DEFAULT_HISTORY_FILENAME = "live_sentiment_history.json"


def _market_mood(score: float) -> str:
    if score > 0.25:
        return "Strongly Positive"
    if score > 0.05:
        return "Slightly Positive"
    if score >= -0.05:
        return "Neutral"
    if score >= -0.25:
        return "Slightly Negative"
    return "Strongly Negative"


def _market_mood_from_calibration(score: float, calibration: dict | None) -> str:
    calibration = calibration or DEFAULT_CALIBRATION
    if score >= calibration["strong_positive_min"]:
        return "Strongly Positive"
    if score >= calibration["slight_positive_min"]:
        return "Slightly Positive"
    if score > calibration["slight_negative_max"]:
        return "Neutral"
    if score > calibration["strong_negative_max"]:
        return "Slightly Negative"
    return "Strongly Negative"


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


def fetch_gdelt_headlines(days: int, max_records: int = 80) -> tuple[list[dict], list[str]]:
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=days)
    seen_titles = set()
    records = []
    errors = []

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
                payload = json.loads(response.read().decode("utf-8"))
        except URLError as exc:
            errors.append(f"{query}: {exc.reason}")
            continue

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
                    "query_bucket": query,
                }
            )

    records.sort(key=lambda item: item.get("published_at") or "", reverse=True)
    return records, errors


def _detect_language(text: str) -> str:
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


def _translate_to_english(text: str) -> tuple[str, bool]:
    language = _detect_language(text)
    if language in {"en", "unknown"}:
        return text, False
    try:
        translated = GoogleTranslator(source="auto", target="en").translate(text)
        return translated or text, True
    except Exception:
        return text, False


def _relevance_terms(text: str) -> list[str]:
    lowered = text.lower()
    return [label for label, pattern in FINANCE_PATTERNS.items() if re.search(pattern, lowered)]


def _prepare_records(records: list[dict]) -> list[dict]:
    prepared = []
    for record in records:
        original_headline = record["headline"]
        translated_headline, was_translated = _translate_to_english(original_headline)
        relevance_terms = _relevance_terms(translated_headline)
        if not relevance_terms:
            continue

        prepared.append(
            {
                **record,
                "headline": translated_headline,
                "original_headline": original_headline,
                "was_translated": was_translated,
                "language": _detect_language(original_headline),
                "relevance_terms": relevance_terms,
            }
        )
    return prepared


def score_headlines(records: list[dict], tokenizer, model) -> list[dict]:
    if not records:
        return []

    prepared_records = _prepare_records(records)
    if not prepared_records:
        return []

    enc = tokenizer(
        [record["headline"] for record in prepared_records],
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt",
    )
    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits, dim=-1).cpu().numpy()

    labels = ["negative", "neutral", "positive"]
    scored_records = []
    for record, prob in zip(prepared_records, probs):
        p_neg, p_neu, p_pos = float(prob[0]), float(prob[1]), float(prob[2])
        score = p_pos - p_neg
        display_label = "Weak / Neutral" if abs(score) < WEAK_SCORE_THRESHOLD else labels[int(prob.argmax())]
        scored_records.append(
            {
                **record,
                "label": labels[int(prob.argmax())],
                "display_label": display_label,
                "score": score,
                "abs_score": abs(score),
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
    strongest = sorted(records, key=lambda item: item["abs_score"], reverse=True)

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
        "sample_headlines": strongest[:10],
        "feed_type": records[0].get("feed_type"),
    }


def _local_date_key(published_at: str | None) -> str | None:
    if not published_at:
        return None
    try:
        dt = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt.astimezone(LOCAL_TZ).date().isoformat()


def _build_calibration(records: list[dict]) -> dict:
    daily_scores: dict[str, list[float]] = {}
    for record in records:
        day_key = _local_date_key(record.get("published_at"))
        if not day_key:
            continue
        daily_scores.setdefault(day_key, []).append(float(record["score"]))

    if len(daily_scores) < 5:
        return dict(DEFAULT_CALIBRATION)

    daily_averages = sorted(sum(values) / len(values) for values in daily_scores.values())
    size = len(daily_averages)

    def pick(q: float) -> float:
        idx = min(max(round((size - 1) * q), 0), size - 1)
        return float(daily_averages[idx])

    strong_negative_max = pick(0.20)
    slight_negative_max = pick(0.40)
    slight_positive_min = pick(0.60)
    strong_positive_min = pick(0.80)

    if not (
        strong_negative_max <= slight_negative_max <= slight_positive_min <= strong_positive_min
    ):
        return dict(DEFAULT_CALIBRATION)

    return {
        "strong_negative_max": strong_negative_max,
        "slight_negative_max": slight_negative_max,
        "slight_positive_min": slight_positive_min,
        "strong_positive_min": strong_positive_min,
        "source": "live_recent_quantiles",
        "days_observed": size,
    }



def _window_bounds(days: int) -> tuple[datetime, datetime]:
    now_local = datetime.now(LOCAL_TZ)
    start_local = datetime.combine(
        (now_local - timedelta(days=days - 1)).date(),
        datetime.min.time(),
        tzinfo=LOCAL_TZ,
    )
    return start_local.astimezone(timezone.utc), now_local.astimezone(timezone.utc)


def _historical_window_bounds(days: int) -> tuple[datetime, datetime]:
    now_local = datetime.now(LOCAL_TZ)
    end_local = datetime.combine(now_local.date(), datetime.min.time(), tzinfo=LOCAL_TZ)
    start_local = end_local - timedelta(days=days)
    return start_local.astimezone(timezone.utc), end_local.astimezone(timezone.utc)


def _records_between(records: list[dict], start_utc: datetime, end_utc: datetime) -> list[dict]:
    filtered = []
    for record in records:
        published_at = record.get("published_at")
        if not published_at:
            continue
        try:
            published_dt = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
        except ValueError:
            continue
        if start_utc <= published_dt <= end_utc:
            filtered.append(record)
    return filtered


def _attach_window_metadata(summary: dict, start_utc: datetime, end_utc: datetime) -> dict:
    summary = dict(summary)
    summary["window_start_utc"] = start_utc.isoformat()
    summary["window_end_utc"] = end_utc.isoformat()
    summary["window_start_local"] = start_utc.astimezone(LOCAL_TZ).isoformat()
    summary["window_end_local"] = end_utc.astimezone(LOCAL_TZ).isoformat()
    summary["timezone"] = str(LOCAL_TZ)
    summary["coverage"] = LIVE_COVERAGE_LABEL
    return summary


def _has_usable_snapshot(snapshot: dict | None) -> bool:
    if not snapshot:
        return False
    return (
        isinstance(snapshot.get("sentiment_index"), (int, float))
        and snapshot.get("feed_type") == "live_gdelt"
        and int(snapshot.get("headlines_analyzed") or 0) > 0
    )


def _history_path_from_output(output_path: Path | None) -> Path | None:
    if output_path is None:
        return None
    return output_path.with_name(DEFAULT_HISTORY_FILENAME)


def _load_history(history_path: Path | None) -> list[dict]:
    if history_path is None or not history_path.exists():
        return []
    try:
        payload = json.loads(history_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if isinstance(payload, list):
        return payload
    return []


def _store_daily_history(snapshot: dict, history_path: Path | None) -> list[dict]:
    history = _load_history(history_path)
    if history_path is None or not _has_usable_snapshot(snapshot):
        return history

    entry = {
        "date": snapshot.get("latest_update"),
        "sentiment_index": snapshot.get("sentiment_index"),
        "market_mood": snapshot.get("market_mood"),
        "headlines_analyzed": snapshot.get("headlines_analyzed"),
        "latest_published_at": snapshot.get("latest_published_at"),
        "positive_share": snapshot.get("positive_share"),
        "neutral_share": snapshot.get("neutral_share"),
        "negative_share": snapshot.get("negative_share"),
        "feed_type": snapshot.get("feed_type"),
        "coverage": snapshot.get("coverage"),
        "timezone": snapshot.get("timezone"),
        "window": snapshot.get("window"),
    }

    retained = [item for item in history if item.get("date") != entry["date"]]
    retained.append(entry)
    retained.sort(key=lambda item: item.get("date") or "")

    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(retained, indent=2), encoding="utf-8")
    return retained


def _aggregate_history_window(
    history: list[dict],
    days: int,
    now_local: datetime,
) -> dict:
    end_local = datetime.combine(now_local.date(), datetime.min.time(), tzinfo=LOCAL_TZ)
    start_local = end_local - timedelta(days=days)
    start_date = start_local.date().isoformat()
    end_date = end_local.date().isoformat()

    window_entries = [
        item
        for item in history
        if isinstance(item.get("date"), str) and start_date <= item["date"] < end_date
    ]

    if not window_entries:
        return _attach_window_metadata(
            summarize_scored([], f"{days}d"),
            start_local.astimezone(timezone.utc),
            end_local.astimezone(timezone.utc),
        )

    total_headlines = sum(int(item.get("headlines_analyzed") or 0) for item in window_entries)
    if total_headlines <= 0:
        return _attach_window_metadata(
            summarize_scored([], f"{days}d"),
            start_local.astimezone(timezone.utc),
            end_local.astimezone(timezone.utc),
        )

    weighted = lambda key: sum(
        float(item.get(key) or 0.0) * int(item.get("headlines_analyzed") or 0)
        for item in window_entries
    ) / total_headlines

    summary = {
        "window": f"{days}d",
        "sentiment_index": weighted("sentiment_index"),
        "market_mood": "Unavailable",
        "headlines_analyzed": total_headlines,
        "positive_share": weighted("positive_share"),
        "neutral_share": weighted("neutral_share"),
        "negative_share": weighted("negative_share"),
        "latest_published_at": max(
            (item.get("latest_published_at") for item in window_entries if item.get("latest_published_at")),
            default=None,
        ),
        "top_positive_headline": None,
        "top_negative_headline": None,
        "sample_headlines": [],
        "feed_type": "live_snapshot_history",
    }
    return _attach_window_metadata(
        summary,
        start_local.astimezone(timezone.utc),
        end_local.astimezone(timezone.utc),
    )


def build_live_snapshot(tokenizer, model, model_version: str, output_path: Path | None = None) -> dict:
    window_summaries = {}
    gdelt_error = None
    source_status = "succeeded"
    now_local = datetime.now(LOCAL_TZ)

    try:
        all_records, gdelt_errors = fetch_gdelt_headlines(30, max_records=80)
        scored_records = score_headlines(all_records, tokenizer, model)
        calibration = _build_calibration(scored_records)
        if gdelt_errors:
            gdelt_error = " | ".join(gdelt_errors)
            source_status = "partial" if scored_records else "failed"
    except URLError as exc:
        gdelt_error = str(exc.reason)
        source_status = "failed"
        scored_records = []
        calibration = dict(DEFAULT_CALIBRATION)

    start_1d, end_1d = _window_bounds(1)

    window_summaries["1d"] = _attach_window_metadata(
        summarize_scored(_records_between(scored_records, start_1d, end_1d), "1d"),
        start_1d,
        end_1d,
    )
    for key in ("1d",):
        summary = window_summaries[key]
        if summary["sentiment_index"] is not None:
            summary["market_mood"] = _market_mood_from_calibration(summary["sentiment_index"], calibration)

    live_window = window_summaries["1d"]
    if live_window["headlines_analyzed"] == 0:
        live_window = window_summaries["7d"]
    if live_window["headlines_analyzed"] == 0:
        live_window = window_summaries["30d"]
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
        "window": live_window["window"],
        "window_start_utc": live_window["window_start_utc"],
        "window_end_utc": live_window["window_end_utc"],
        "window_start_local": live_window["window_start_local"],
        "window_end_local": live_window["window_end_local"],
        "timezone": live_window["timezone"],
        "coverage": LIVE_COVERAGE_LABEL,
        "calibration": calibration,
        "model_version": model_version,
        "source_policy": {
            "primary_source": "GDELT",
            "fallback_source": None,
            "gdelt_status": source_status,
        },
        "window_summaries": window_summaries,
        "gdelt_error": gdelt_error,
    }

    history_path = _history_path_from_output(output_path)
    history = _store_daily_history(snapshot, history_path)
    window_summaries["7d"] = _aggregate_history_window(history, 7, now_local)
    window_summaries["30d"] = _aggregate_history_window(history, 30, now_local)
    for key in ("7d", "30d"):
        summary = window_summaries[key]
        if summary["sentiment_index"] is not None:
            summary["market_mood"] = _market_mood_from_calibration(summary["sentiment_index"], calibration)
    snapshot["window_summaries"] = window_summaries
    snapshot["history_path"] = str(history_path) if history_path else None

    if output_path is not None and not _has_usable_snapshot(snapshot) and output_path.exists():
        cached_snapshot = json.loads(output_path.read_text(encoding="utf-8"))
        if _has_usable_snapshot(cached_snapshot):
            cached_snapshot["source_policy"] = {
                "primary_source": "GDELT",
                "fallback_source": None,
                "gdelt_status": "cached_previous_snapshot",
            }
            cached_snapshot["gdelt_error"] = gdelt_error
            cached_snapshot["latest_update"] = datetime.now(timezone.utc).date().isoformat()
            output_path.write_text(json.dumps(cached_snapshot, indent=2), encoding="utf-8")
            return cached_snapshot

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")

    return snapshot
