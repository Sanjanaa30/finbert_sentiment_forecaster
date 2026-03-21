"""
Live news pipeline — fetch, translate, score, store.

Flow:
    fetch_all_sources()
        → translate non-English headlines
        → store ALL to raw_headlines (before filtering)
        → filter noise / relevance
        → score with FinBERT (only relevant headlines)
        → store scored to scored_headlines
        → aggregate daily_sentiment
        → return summary for API
"""
from __future__ import annotations

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
    get_last_run_at,
    set_last_run_at,
    _market_mood,
)
from src.news_sources import gdelt, yahoo_rss, alpha_vantage, google_rss
from src.news_sources.filters import is_relevant, relevance_terms

DetectorFactory.seed = 0
LOCAL_TZ = ZoneInfo("America/New_York")
WEAK_SCORE_THRESHOLD = 0.03



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

    # 2. Google News RSS (no API key, no rate limits)
    google_records, google_errors = google_rss.fetch()
    all_errors.extend(google_errors)
    for r in google_records:
        if r["headline"] not in seen_titles:
            seen_titles.add(r["headline"])
            all_records.append(r)

    # 3. Alpha Vantage (skipped silently if no API key)
    av_records, av_errors = alpha_vantage.fetch(days=days)
    all_errors.extend(av_errors)
    for r in av_records:
        if r["headline"] not in seen_titles:
            seen_titles.add(r["headline"])
            all_records.append(r)

    return all_records, all_errors


# ---------------------------------------------------------------------------
# Translate + filter
# ---------------------------------------------------------------------------

def _is_recent(published_at: str | None, max_age_days: int = 31) -> bool:
    """Reject any record with published_at older than max_age_days."""
    if not published_at:
        return True  # no date = keep (will use fetched_at as fallback)
    try:
        dt = datetime.fromisoformat(published_at)
        cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        return dt >= cutoff
    except (ValueError, TypeError):
        return False


def translate_records(records: list[dict]) -> list[dict]:
    """
    Translate non-English headlines. Returns ALL records (no filtering).
    Rejects any record with published_at before 2026.
    These go into raw_headlines as-is.
    """
    translated_records = []
    for record in records:
        if not _is_recent(record.get("published_at")):
            continue
        translated, was_translated, lang = _translate(record["headline"])
        translated_records.append({
            **record,
            "headline": translated,
            "original_headline": record.get("original_headline") or record["headline"],
            "was_translated": was_translated,
            "language": lang,
        })
    return translated_records


def filter_relevant(records: list[dict]) -> list[dict]:
    """
    Filter for financial relevance and attach relevance_terms.
    Only these go into scored_headlines.
    """
    filtered = []
    for record in records:
        terms = relevance_terms(record["headline"])
        if not terms or not is_relevant(record["headline"]):
            continue
        filtered.append({
            **record,
            "relevance_terms": terms,
        })
    return filtered


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



def get_historical_window_summary(
    days: int,
    tokenizer,
    model,
    window_label: str,
) -> dict:
    """
    Read 7D/30D window from SQLite. Backfill is handled in run_pipeline,
    not here — so this is always a fast read.
    """
    start_utc, end_utc = _historical_window(days)
    rows = fetch_scored_for_window(start_utc, end_utc)
    summary = summarize(rows, window_label)
    summary["window_start_local"] = start_utc.astimezone(LOCAL_TZ).isoformat()
    summary["window_end_local"] = end_utc.astimezone(LOCAL_TZ).isoformat()
    summary["source"] = "SQLite (historical)"
    return summary


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------

def _store_and_score(records: list[dict], tokenizer, model) -> int:
    """
    Translate → store raw → filter → score → store scored.
    Returns number of scored headlines.
    """
    translated = translate_records(records)
    raw_ids = insert_raw_headlines(translated)

    relevant = filter_relevant(translated)
    headline_to_raw_id = {}
    for rec, rid in zip(translated, raw_ids):
        if rid is not None:
            headline_to_raw_id[rec["headline"]] = rid
    relevant_raw_ids = [headline_to_raw_id.get(r["headline"]) for r in relevant]

    scored = score_records(relevant, tokenizer, model)
    insert_scored_headlines(scored, relevant_raw_ids)
    return len(scored)


def _backfill_date_range(
    start_utc: datetime,
    end_utc: datetime,
    tokenizer,
    model,
) -> tuple[int, list[str]]:
    """
    Fetch from GDELT for a specific date range, store, and
    create daily_sentiment rows for each day in that range.
    Returns (scored_count, errors).
    """
    raw_records, errors = gdelt.fetch(start_utc, end_utc, max_records=80)
    if not raw_records:
        return 0, errors

    count = _store_and_score(raw_records, tokenizer, model)

    # Create daily_sentiment rows for each day in the range
    current = start_utc.astimezone(LOCAL_TZ).date()
    end_date = end_utc.astimezone(LOCAL_TZ).date()
    while current <= end_date:
        day_start = datetime.combine(current, datetime.min.time(), tzinfo=LOCAL_TZ).astimezone(timezone.utc)
        day_end = datetime.combine(current, datetime.max.time(), tzinfo=LOCAL_TZ).astimezone(timezone.utc)
        day_rows = fetch_scored_for_window(day_start, day_end)
        if day_rows:
            upsert_daily_sentiment(current, day_rows)
        current += timedelta(days=1)

    return count, errors


def run_pipeline(tokenizer, model, model_version: str) -> dict:
    """
    Full pipeline:

    First run (no last_run_at):
        1. Fetch today from Yahoo RSS + Google RSS
        2. Backfill last 7 days from GDELT (yesterday → 7 days ago)
        3. Backfill rest of month from GDELT (8 days ago → 1st of month)

    Subsequent runs:
        1. Fetch today from Yahoo RSS + Google RSS
        2. If gap > 1 day since last run, backfill gap from GDELT
        3. Update last_run_at
    """
    init_db()

    now_utc = datetime.now(timezone.utc)
    now_local = datetime.now(LOCAL_TZ)
    today = now_local.date()
    all_errors: list[str] = []

    last_run = get_last_run_at()

    # --- Step 1: Fetch today from RSS sources ---
    raw_records, errors = fetch_all_sources(days=1)
    all_errors.extend(errors)
    today_scored = _store_and_score(raw_records, tokenizer, model)
    sources_used = list({r.get("feed_type") for r in raw_records if r.get("feed_type")})

    if last_run is None:
        # --- First run: backfill the current month ---
        print("[pipeline] First run detected — backfilling current month from GDELT")

        # 7D window: yesterday → 7 days ago
        seven_days_ago = datetime.combine(
            (now_local - timedelta(days=7)).date(), datetime.min.time(), tzinfo=LOCAL_TZ
        ).astimezone(timezone.utc)
        yesterday_end = datetime.combine(
            (now_local - timedelta(days=1)).date(), datetime.max.time(), tzinfo=LOCAL_TZ
        ).astimezone(timezone.utc)
        count_7d, errs_7d = _backfill_date_range(seven_days_ago, yesterday_end, tokenizer, model)
        all_errors.extend(errs_7d)
        print(f"[pipeline] 7D backfill: {count_7d} headlines scored")

        import time
        time.sleep(5)  # avoid GDELT rate limit between backfill calls

        # 30D window: 1st of month → 8 days ago
        first_of_month = datetime.combine(
            now_local.date().replace(day=1), datetime.min.time(), tzinfo=LOCAL_TZ
        ).astimezone(timezone.utc)
        eight_days_ago_end = datetime.combine(
            (now_local - timedelta(days=8)).date(), datetime.max.time(), tzinfo=LOCAL_TZ
        ).astimezone(timezone.utc)
        if first_of_month < eight_days_ago_end:
            count_30d, errs_30d = _backfill_date_range(first_of_month, eight_days_ago_end, tokenizer, model)
            all_errors.extend(errs_30d)
            print(f"[pipeline] 30D backfill: {count_30d} headlines scored")

        if sources_used and "gdelt" not in sources_used:
            sources_used.append("gdelt")

    else:
        # --- Subsequent run: check for gaps ---
        gap_hours = (now_utc - last_run).total_seconds() / 3600
        if gap_hours > 24:
            # Fill the gap from GDELT
            gap_start = last_run
            gap_end = datetime.combine(
                (now_local - timedelta(days=1)).date(), datetime.max.time(), tzinfo=LOCAL_TZ
            ).astimezone(timezone.utc)
            if gap_start < gap_end:
                gap_days = int((gap_end - gap_start).total_seconds() / 86400) + 1
                print(f"[pipeline] Gap detected: {gap_days} days since last run — backfilling from GDELT")
                count_gap, errs_gap = _backfill_date_range(gap_start, gap_end, tokenizer, model)
                all_errors.extend(errs_gap)
                print(f"[pipeline] Gap backfill: {count_gap} headlines scored")

    # --- Aggregate today's daily_sentiment ---
    start_today, end_today = _today_window()
    all_today = fetch_scored_for_window(start_today, end_today)
    if all_today:
        upsert_daily_sentiment(today, all_today)

    # --- Update last_run_at ---
    set_last_run_at(now_utc)

    # --- Build summaries ---
    start_1d, end_1d = _today_window()
    today_records = fetch_scored_for_window(start_1d, end_1d)
    today_summary = summarize(today_records, "1d")
    today_summary["window_start_local"] = start_1d.astimezone(LOCAL_TZ).isoformat()
    today_summary["window_end_local"] = end_1d.astimezone(LOCAL_TZ).isoformat()
    today_summary["source"] = "Live (SQLite today)"

    window_summaries = {"1d": today_summary}
    for days, key in [(7, "7d"), (30, "30d")]:
        window_summaries[key] = get_historical_window_summary(
            days=days, tokenizer=tokenizer, model=model, window_label=key
        )

    return {
        "latest_update": today.isoformat(),
        "model_version": model_version,
        "source_errors": all_errors,
        "sources_used": sources_used,
        "window_summaries": window_summaries,
        **today_summary,
    }
