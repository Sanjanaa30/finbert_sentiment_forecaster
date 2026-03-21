from datetime import datetime, date, timedelta, timezone
from pathlib import Path
import json
import threading

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from time import perf_counter
from fastapi import Request

from app.model_store import load_bundle, get_bundle
from app.schemas import (
    ScoreRequest,
    ScoreResponse,
    ScoreSingleResponse,
    ScoreItem,
    ForecastRequest,
    ForecastResponse,
)
from app.security import verify_api_key
from src.live_news import run_pipeline, get_historical_window_summary, _today_window
from src.database import (
    init_db,
    rebuild_daily_sentiment,
    fetch_scored_for_window,
    fetch_daily_sentiment_range,
    fetch_today_top_headlines,
    _market_mood,
)
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "artifacts" / "walkforward_metrics.csv"
MODEL_DIR = ROOT / "models" / "finbert_sentiment"
LOCAL_TZ = ZoneInfo("America/New_York")

app = FastAPI(title="FinBERT Forecast API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_tokenizer = None
_sent_model = None
_scheduler: BackgroundScheduler | None = None


def _run_pipeline_job() -> None:
    """Background job: run the full news pipeline and log results."""
    try:
        print(f"[scheduler] Starting pipeline run at {datetime.now(LOCAL_TZ).strftime('%Y-%m-%d %H:%M %Z')}")
        result = run_pipeline(
            tokenizer=_tokenizer,
            model=_sent_model,
            model_version=get_bundle().model_version,
        )
        si = result.get("sentiment_index")
        mood = result.get("market_mood")
        count = result.get("headlines_analyzed")
        sources = result.get("sources_used", [])
        errors = result.get("source_errors", [])
        print(
            f"[scheduler] Done — sentiment: {f'{si:.3f}' if si is not None else 'N/A'}, "
            f"mood: {mood}, headlines: {count}, sources: {sources}"
        )
        if errors:
            print(f"[scheduler] Source errors: {errors}")
    except Exception as exc:
        print(f"[scheduler] Pipeline error: {exc}")


@app.on_event("startup")
def startup() -> None:
    global _tokenizer, _sent_model, _scheduler
    init_db()
    repaired = rebuild_daily_sentiment()
    if repaired:
        print(f"[startup] Rebuilt daily_sentiment for {repaired} missing days.")
    load_bundle()
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    _sent_model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).eval()

    # Run pipeline immediately on startup (in background so server starts fast)
    threading.Thread(target=_run_pipeline_job, daemon=True).start()

    # Schedule pipeline at 9AM, 12PM, 3PM, 6PM ET, Mon–Fri
    _scheduler = BackgroundScheduler(timezone="America/New_York")
    _scheduler.add_job(
        _run_pipeline_job,
        CronTrigger(hour="9,12,15,18", day_of_week="mon-fri", timezone="America/New_York"),
        id="live_pipeline",
        replace_existing=True,
    )
    _scheduler.start()
    print("[scheduler] Scheduled pipeline at 9AM, 12PM, 3PM, 6PM ET (Mon–Fri)")


@app.on_event("shutdown")
def shutdown() -> None:
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        print("[scheduler] Scheduler stopped.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _window_bounds(days: int) -> tuple[datetime, datetime]:
    now_local = datetime.now(LOCAL_TZ)
    start_local = datetime.combine(
        (now_local - timedelta(days=days - 1)).date(),
        datetime.min.time(),
        tzinfo=LOCAL_TZ,
    )
    return start_local.astimezone(timezone.utc), now_local.astimezone(timezone.utc)


def _summarize_rows(rows: list[dict], window: str) -> dict:
    if not rows:
        return {
            "window": window, "sentiment_index": None,
            "market_mood": "Unavailable", "headlines_analyzed": 0,
            "positive_count": 0, "neutral_count": 0, "negative_count": 0,
            "positive_share": None, "neutral_share": None, "negative_share": None,
            "top_positive_headline": None, "top_negative_headline": None,
            "positive_headlines": [], "neutral_headlines": [], "negative_headlines": [],
        }
    scores = [r["score"] for r in rows]
    labels = [r["label"] for r in rows]
    total = len(rows)
    pos = labels.count("positive")
    neu = labels.count("neutral")
    neg = labels.count("negative")
    si = sum(scores) / total
    sorted_pos = sorted(rows, key=lambda r: r["score"], reverse=True)
    sorted_neg = sorted(rows, key=lambda r: r["score"])
    return {
        "window": window,
        "sentiment_index": si,
        "market_mood": _market_mood(si),
        "headlines_analyzed": total,
        "positive_count": pos, "neutral_count": neu, "negative_count": neg,
        "positive_share": pos / total,
        "neutral_share": neu / total,
        "negative_share": neg / total,
        "top_positive_headline": sorted_pos[0]["headline"] if sorted_pos else None,
        "top_negative_headline": sorted_neg[0]["headline"] if sorted_neg else None,
        "positive_headlines": sorted_pos[:5],
        "neutral_headlines": sorted([r for r in rows if r["label"] == "neutral"], key=lambda r: r["abs_score"])[:5],
        "negative_headlines": sorted_neg[:5],
        "sample_headlines": sorted(rows, key=lambda r: r["abs_score"], reverse=True)[:10],
    }


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health", dependencies=[Depends(verify_api_key)])
def health():
    b = get_bundle()
    return {"status": "ok", "model_loaded": True, "model_version": b.model_version}


# ---------------------------------------------------------------------------
# Live overview — today's sentiment from SQLite
# ---------------------------------------------------------------------------

@app.get("/dashboard/live-overview", dependencies=[Depends(verify_api_key)])
def dashboard_live_overview():
    start_utc, end_utc = _window_bounds(1)
    rows = fetch_scored_for_window(start_utc, end_utc)
    summary = _summarize_rows(rows, "1d")
    top = fetch_today_top_headlines()
    summary["positive_headlines"] = top["positive"]
    summary["neutral_headlines"] = top["neutral"]
    summary["negative_headlines"] = top["negative"]
    summary["window_start_local"] = start_utc.astimezone(LOCAL_TZ).isoformat()
    summary["window_end_local"] = end_utc.astimezone(LOCAL_TZ).isoformat()
    summary["latest_update"] = date.today().isoformat()
    summary["feed_type"] = "live_sqlite"
    summary["coverage"] = "Global market news"
    summary["timezone"] = "America/New_York"
    return summary


# ---------------------------------------------------------------------------
# Live vs history — 1d / 7d / 30d from SQLite
# ---------------------------------------------------------------------------

@app.get("/dashboard/live-vs-history", dependencies=[Depends(verify_api_key)])
def dashboard_live_vs_history():
    # 1D: today midnight → now
    start_1d, end_1d = _today_window()
    rows_1d = fetch_scored_for_window(start_1d, end_1d)
    live = _summarize_rows(rows_1d, "1d")
    live["window_start_local"] = start_1d.astimezone(LOCAL_TZ).isoformat()
    live["window_end_local"] = end_1d.astimezone(LOCAL_TZ).isoformat()
    live["window_source_label"] = "Live (today)"

    # 7D/30D: hybrid — stable, excludes today
    recent_7d = get_historical_window_summary(7, _tokenizer, _sent_model, "7d")
    recent_30d = get_historical_window_summary(30, _tokenizer, _sent_model, "30d")

    return {
        "live": live,
        "recent_7d": recent_7d,
        "recent_30d": recent_30d,
        "source_policy": {
            "primary_source": "Yahoo Finance RSS + Alpha Vantage + GDELT",
            "gdelt_status": "supplementary",
        },
        "timestamp": datetime.now(timezone.utc),
    }


# ---------------------------------------------------------------------------
# Live sentiment trend — daily aggregates from SQLite
# ---------------------------------------------------------------------------

@app.get("/dashboard/live-sentiment-trend", dependencies=[Depends(verify_api_key)])
def dashboard_live_sentiment_trend(days: int = 30):
    end = date.today()
    start = end - timedelta(days=days)
    rows = fetch_daily_sentiment_range(start, end)
    return {"rows": rows}


# ---------------------------------------------------------------------------
# Headline volume — daily counts from SQLite
# ---------------------------------------------------------------------------

@app.get("/dashboard/headline-volume", dependencies=[Depends(verify_api_key)])
def dashboard_headline_volume(days: int = 60):
    end = date.today()
    start = end - timedelta(days=days)
    rows = fetch_daily_sentiment_range(start, end)
    return {
        "rows": [{"date": r["date"], "headline_volume": r["headline_volume"]} for r in rows]
    }


# ---------------------------------------------------------------------------
# Live refresh — re-run full pipeline
# ---------------------------------------------------------------------------

@app.post("/dashboard/live-refresh", dependencies=[Depends(verify_api_key)])
def dashboard_live_refresh():
    try:
        return run_pipeline(
            tokenizer=_tokenizer,
            model=_sent_model,
            model_version=get_bundle().model_version,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Pipeline failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Score headline
# ---------------------------------------------------------------------------

@app.post(
    "/score_headline",
    response_model=ScoreSingleResponse | ScoreResponse,
    dependencies=[Depends(verify_api_key)],
)
def score_headline(req: ScoreRequest):
    single = isinstance(req.headline, str)
    headlines = [req.headline] if single else req.headline
    if not headlines:
        raise HTTPException(status_code=400, detail="headline list is empty")

    enc = _tokenizer(headlines, truncation=True, padding=True,
                     max_length=128, return_tensors="pt")
    with torch.no_grad():
        probs = torch.softmax(_sent_model(**enc).logits, dim=-1).cpu().numpy()

    labels = ["negative", "neutral", "positive"]
    out = []
    for h, p in zip(headlines, probs):
        p_neg, p_neu, p_pos = float(p[0]), float(p[1]), float(p[2])
        out.append(ScoreItem(
            headline=h,
            label=labels[int(p.argmax())],
            score=p_pos - p_neg,
            probabilities={"negative": p_neg, "neutral": p_neu, "positive": p_pos},
        ))

    ts = datetime.now(timezone.utc)
    model_version = get_bundle().model_version
    if single:
        return ScoreSingleResponse(result=out[0], timestamp=ts, model_version=model_version)
    return ScoreResponse(results=out, timestamp=ts, model_version=model_version)


# ---------------------------------------------------------------------------
# Model summary
# ---------------------------------------------------------------------------

@app.get("/dashboard/model-summary", dependencies=[Depends(verify_api_key)])
def dashboard_model_summary():
    if not METRICS_PATH.exists():
        raise HTTPException(status_code=404, detail="walkforward_metrics.csv not found")
    metrics_df = pd.read_csv(METRICS_PATH)
    bundle = get_bundle()
    config = bundle.config

    selected = metrics_df[
        (metrics_df["horizon_days"] == config.get("horizon_days"))
        & (metrics_df["threshold_bps"] == config.get("threshold_bps"))
        & (metrics_df["feature_set"] == config.get("feature_set"))
        & (metrics_df["model"] == config.get("model"))
    ].copy()

    windows = []
    if not selected.empty:
        selected = selected.sort_values("test_start")
        for _, row in selected.iterrows():
            windows.append({
                "test_start": row["test_start"],
                "test_end": row["test_end"],
                "accuracy": float(row["accuracy"]),
                "f1": float(row["f1"]),
                "roc_auc": float(row["roc_auc"]),
            })

    return {
        "selected_config": config,
        "average_roc_auc": float(selected["roc_auc"].mean()) if not selected.empty else None,
        "windows": windows,
        "model_version": bundle.model_version,
    }


# ---------------------------------------------------------------------------
# Request logging middleware
# ---------------------------------------------------------------------------

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = perf_counter()
    response = await call_next(request)
    print(
        f"[api] {request.method} {request.url.path} "
        f"status={response.status_code} "
        f"latency_ms={(perf_counter() - start) * 1000:.2f}"
    )
    return response
