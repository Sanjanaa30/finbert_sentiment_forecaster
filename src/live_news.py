"""
Live news pipeline — fetch, translate, score, store.

Flow:
    fetch_all_sources()
        → translate non-English headlines
        → filter noise
        → store raw to SQLite
        → score with FinBERT
        → store scored to SQLite
        → aggregate daily_sentiment
        → return summary for API
"""
from __future__ import annotations

import re
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import torch
from deep_translator import GoogleTranslator
from langdetect import DetectorFactory, LangDetectException, detect

from src.database import (
    init_db,
    insert_raw_headlines,
    insert_scored_headlines,
    upsert_daily_sentiment,
    fetch_scored_for_window,
    fetch_today_top_headlines,
    count_distinct_days,
    _market_mood,
)
from src.news_sources import gdelt, yahoo_rss, alpha_vantage
from src.news_sources.filters import is_relevant, relevance_terms

DetectorFactory.seed = 0
LOCAL_TZ = ZoneInfo("America/New_York")
WEAK_SCORE_THRESHOLD = 0.03

# Backfill state: prevent concurrent and repeated backfills
_backfill_lock = threading.Lock()
_backfill_attempted: dict[int, datetime] = {}   # days → last attempt time
BACKFILL_COOLDOWN_MINUTES = 30


# ---------------------------------------------------------------------------
# Translation helpers
# ---------------------------------------------------------------------------

def _detect_language(text: str) -> str:
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


def _translate(text: str) -> tuple[str, bool, str]:
    """Returns (translated_text, was_translated, language)."""
    lang = _detect_language(text)
    if lang in {"en", "unknown"}:
        return text, False, lang
    try:
        translated = GoogleTranslator(source="auto", target="en").translate(text)
        return translated or text, True, lang
    except Exception:
        return text, False, lang


# ---------------------------------------------------------------------------
# Fetch from all sources
# ---------------------------------------------------------------------------

def fetch_all_sources(days: int = 1) -> tuple[list[dict], list[str]]:
    """
    Fetch from all configured news sources for the past `days` days.
    Deduplicates by headline text across sources.
    Returns (records, errors).
    """
    all_records: list[dict] = []
    all_errors: list[str] = []
    seen_titles: set[str] = set()

    # 1. Yahoo Finance RSS (no API key, always runs)
    yahoo_records, yahoo_errors = yahoo_rss.fetch()
    all_errors.extend(yahoo_errors)
    for r in yahoo_records:
        if r["headline"] not in seen_titles:
            seen_titles.add(r["headline"])
            all_records.append(r)

    # 2. Alpha Vantage (skipped silently if no API key)
    av_records, av_errors = alpha_vantage.fetch(days=days)
    all_errors.extend(av_errors)
    for r in av_records:
        if r["headline"] not in seen_titles:
            seen_titles.add(r["headline"])
            all_records.append(r)

    # 3. GDELT (free, rate-limited — used as supplementary)
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=days)
    gdelt_records, gdelt_errors = gdelt.fetch(start_dt, end_dt, max_records=80)
    all_errors.extend(gdelt_errors)
    for r in gdelt_records:
        if r["headline"] not in seen_titles:
            seen_titles.add(r["headline"])
            all_records.append(r)

    return all_records, all_errors


# ---------------------------------------------------------------------------
# Translate + filter
# ---------------------------------------------------------------------------

def prepare_records(records: list[dict]) -> list[dict]:
    """
    Translate non-English headlines, re-filter after translation,
    attach relevance_terms.
    """
    prepared = []
    for record in records:
        translated, was_translated, lang = _translate(record["headline"])
        terms = relevance_terms(translated)
        if not terms or not is_relevant(translated):
            continue
        prepared.append({
            **record,
            "headline": translated,
            "original_headline": record.get("original_headline") or record["headline"],
            "was_translated": was_translated,
            "language": lang,
            "relevance_terms": terms,
        })
    return prepared


# ---------------------------------------------------------------------------
# Score with FinBERT
# ---------------------------------------------------------------------------

def score_records(records: list[dict], tokenizer, model) -> list[dict]:
    """Run FinBERT inference on a list of prepared records."""
    if not records:
        return []

    enc = tokenizer(
        [r["headline"] for r in records],
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt",
    )
    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits, dim=-1).cpu().numpy()

    labels = ["negative", "neutral", "positive"]
    scored = []
    for record, prob in zip(records, probs):
        p_neg, p_neu, p_pos = float(prob[0]), float(prob[1]), float(prob[2])
        score = p_pos - p_neg
        label = labels[int(prob.argmax())]
        display_label = "Weak / Neutral" if abs(score) < WEAK_SCORE_THRESHOLD else label
        scored.append({
            **record,
            "label": label,
            "display_label": display_label,
            "score": score,
            "abs_score": abs(score),
            "positive_prob": p_pos,
            "neutral_prob": p_neu,
            "negative_prob": p_neg,
        })
    return scored


# ---------------------------------------------------------------------------
# Summarize for API response
# ---------------------------------------------------------------------------

def summarize(records: list[dict], window: str) -> dict:
    """Build a summary dict from a list of scored records."""
    if not records:
        return {
            "window": window,
            "sentiment_index": None,
            "market_mood": "Unavailable",
            "headlines_analyzed": 0,
            "positive_count": 0,
            "neutral_count": 0,
            "negative_count": 0,
            "positive_share": None,
            "neutral_share": None,
            "negative_share": None,
            "top_positive_headline": None,
            "top_negative_headline": None,
            "positive_headlines": [],
            "neutral_headlines": [],
            "negative_headlines": [],
            "sample_headlines": [],
        }

    scores = [r["score"] for r in records]
    labels = [r["label"] for r in records]
    total = len(records)
    pos = labels.count("positive")
    neu = labels.count("neutral")
    neg = labels.count("negative")
    sentiment_index = sum(scores) / total

    sorted_pos = sorted(records, key=lambda r: r["score"], reverse=True)
    sorted_neg = sorted(records, key=lambda r: r["score"])
    sorted_abs = sorted(records, key=lambda r: r["abs_score"], reverse=True)

    grouped = {
        "positive": sorted([r for r in records if r["label"] == "positive"], key=lambda r: r["score"], reverse=True),
        "neutral":  sorted([r for r in records if r["label"] == "neutral"],  key=lambda r: r["abs_score"]),
        "negative": sorted([r for r in records if r["label"] == "negative"], key=lambda r: r["score"]),
    }

    return {
        "window": window,
        "sentiment_index": sentiment_index,
        "market_mood": _market_mood(sentiment_index),
        "headlines_analyzed": total,
        "positive_count": pos,
        "neutral_count": neu,
        "negative_count": neg,
        "positive_share": pos / total,
        "neutral_share": neu / total,
        "negative_share": neg / total,
        "top_positive_headline": sorted_pos[0]["headline"] if sorted_pos else None,
        "top_negative_headline": sorted_neg[0]["headline"] if sorted_neg else None,
        "positive_headlines": grouped["positive"][:5],
        "neutral_headlines":  grouped["neutral"][:5],
        "negative_headlines": grouped["negative"][:5],
        "sample_headlines":   sorted_abs[:10],
    }


# ---------------------------------------------------------------------------
# Window bounds
# ---------------------------------------------------------------------------

def _today_window() -> tuple[datetime, datetime]:
    """1D: today midnight ET → now."""
    now_local = datetime.now(LOCAL_TZ)
    start = datetime.combine(now_local.date(), datetime.min.time(), tzinfo=LOCAL_TZ)
    return start.astimezone(timezone.utc), now_local.astimezone(timezone.utc)


def _historical_window(days: int) -> tuple[datetime, datetime]:
    """
    7D/30D: `days` days ago midnight ET → now.
    Includes today so data is available from day 1.
    """
    now_local = datetime.now(LOCAL_TZ)
    start = datetime.combine((now_local - timedelta(days=days - 1)).date(), datetime.min.time(), tzinfo=LOCAL_TZ)
    return start.astimezone(timezone.utc), now_local.astimezone(timezone.utc)


# ---------------------------------------------------------------------------
# Backfill — fetch historical window from GDELT and store permanently
# ---------------------------------------------------------------------------

MIN_DAYS_REQUIRED = {7: 1, 30: 1}  # accept any stored data before stopping backfill attempts


def _backfill_window(
    days: int,
    start_utc: datetime,
    end_utc: datetime,
    tokenizer,
    model,
) -> None:
    """
    Fetch from GDELT for the historical window, score, and store in SQLite.
    Protected by a lock + cooldown so concurrent requests don't pile up.
    """
    now = datetime.now(timezone.utc)

    # Skip if a backfill for this window was attempted recently
    last = _backfill_attempted.get(days)
    if last and (now - last).total_seconds() < BACKFILL_COOLDOWN_MINUTES * 60:
        print(f"[backfill] Skipping {days}d — attempted {int((now - last).total_seconds() / 60)}m ago")
        return

    # Only one backfill at a time across all windows
    if not _backfill_lock.acquire(blocking=False):
        print(f"[backfill] Skipping {days}d — another backfill is already running")
        return

    try:
        _backfill_attempted[days] = now
        print(f"[backfill] Fetching {days}d window from GDELT...")
        raw_records, errors = gdelt.fetch(start_utc, end_utc, max_records=80)
        if errors:
            print(f"[backfill] GDELT errors: {errors}")
        if not raw_records:
            print(f"[backfill] No records returned for {days}d window.")
            return

        prepared = prepare_records(raw_records)
        raw_ids = insert_raw_headlines(prepared)
        scored = score_records(prepared, tokenizer, model)
        insert_scored_headlines(scored, raw_ids)

        from collections import defaultdict
        from datetime import date as date_type
        daily_buckets: dict[str, list[tuple[dict, int | None]]] = defaultdict(list)
        for record, raw_id in zip(scored, raw_ids):
            pub = record.get("published_at")
            if pub:
                try:
                    day_key = datetime.fromisoformat(pub).astimezone(LOCAL_TZ).date()
                    daily_buckets[day_key.isoformat()].append((record, raw_id))
                except ValueError:
                    pass
        for day_str, pairs in daily_buckets.items():
            day_records = [p[0] for p in pairs]
            day_raw_ids = [p[1] for p in pairs]
            upsert_daily_sentiment(date_type.fromisoformat(day_str), day_records, day_raw_ids)

        print(f"[backfill] Stored {len(scored)} headlines across {len(daily_buckets)} days.")
    finally:
        _backfill_lock.release()


def get_historical_window_summary(
    days: int,
    tokenizer,
    model,
    window_label: str,
) -> dict:
    """
    Hybrid: use SQLite if any data stored, else attempt one GDELT backfill.
    7D/30D exclude today for stability.
    """
    start_utc, end_utc = _historical_window(days)
    min_days = MIN_DAYS_REQUIRED.get(days, 1)
    stored_days = count_distinct_days(start_utc, end_utc)

    if stored_days < min_days:
        _backfill_window(days, start_utc, end_utc, tokenizer, model)

    rows = fetch_scored_for_window(start_utc, end_utc)
    summary = summarize(rows, window_label)
    summary["window_start_local"] = start_utc.astimezone(LOCAL_TZ).isoformat()
    summary["window_end_local"] = end_utc.astimezone(LOCAL_TZ).isoformat()
    summary["source"] = "SQLite (historical)" if stored_days >= min_days else "SQLite (backfilled from GDELT)"
    return summary


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------

def run_pipeline(tokenizer, model, model_version: str) -> dict:
    """
    Full pipeline:
    1. Fetch from all sources
    2. Translate + filter noise
    3. Store raw to SQLite
    4. Score with FinBERT
    5. Store scored to SQLite
    6. Aggregate daily_sentiment
    7. Return summary for today
    """
    init_db()

    now_local = datetime.now(LOCAL_TZ)
    today = now_local.date()

    # --- Fetch ---
    raw_records, errors = fetch_all_sources(days=1)

    # --- Translate + filter ---
    prepared = prepare_records(raw_records)

    # --- Store raw ---
    raw_ids = insert_raw_headlines(prepared)

    # --- Score ---
    scored = score_records(prepared, tokenizer, model)

    # --- Store scored ---
    insert_scored_headlines(scored, raw_ids)

    # --- Aggregate today ---
    upsert_daily_sentiment(today, scored, raw_ids)

    # --- 1D: today midnight → now ---
    start_1d, end_1d = _today_window()
    today_records = fetch_scored_for_window(start_1d, end_1d)
    today_summary = summarize(today_records, "1d")
    today_summary["window_start_local"] = start_1d.astimezone(LOCAL_TZ).isoformat()
    today_summary["window_end_local"] = end_1d.astimezone(LOCAL_TZ).isoformat()
    today_summary["source"] = "Live (SQLite today)"

    # --- 7D/30D: hybrid (SQLite if enough data, else GDELT backfill) ---
    window_summaries = {"1d": today_summary}
    for days, key in [(7, "7d"), (30, "30d")]:
        window_summaries[key] = get_historical_window_summary(
            days=days, tokenizer=tokenizer, model=model, window_label=key
        )

    return {
        "latest_update": today.isoformat(),
        "model_version": model_version,
        "source_errors": errors,
        "sources_used": list({r.get("feed_type") for r in prepared if r.get("feed_type")}),
        "window_summaries": window_summaries,
        **today_summary,
    }
