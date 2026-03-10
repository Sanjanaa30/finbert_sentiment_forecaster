from datetime import datetime, timezone
from pathlib import Path
import json

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

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
from src.live_news import build_live_snapshot

ROOT = Path(__file__).resolve().parents[1]
SENT_PATH = ROOT / "data" / "processed" / "daily_sentiment_index.csv"
SPY_PATH = ROOT / "data" / "processed" / "spy_stooq.csv"
METRICS_PATH = ROOT / "artifacts" / "walkforward_metrics.csv"
MODEL_DIR = ROOT / "models" / "finbert_sentiment"
LIVE_SNAPSHOT_PATH = ROOT / "artifacts" / "live_sentiment_snapshot.json"

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


def _require_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{label} not found")
    df = pd.read_csv(path)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"{label} is empty")
    return df


def _market_mood(score: float) -> str:
    if score > 0.02:
        return "Positive"
    if score < -0.02:
        return "Negative"
    return "Neutral"


def _load_live_snapshot() -> dict:
    if not LIVE_SNAPSHOT_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="live_sentiment_snapshot.json not found. Run the live snapshot builder first.",
        )
    return json.loads(LIVE_SNAPSHOT_PATH.read_text(encoding="utf-8"))


@app.on_event("startup")
def startup() -> None:
    global _tokenizer, _sent_model
    load_bundle()  # loads final_model.pkl + final_selected_config.json
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    _sent_model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).eval()

@app.get("/health", dependencies=[Depends(verify_api_key)])
def health():
    b = get_bundle()
    return {
        "status": "ok",
        "model_loaded": True,
        "model_version": b.model_version,
    }

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

    enc = _tokenizer(headlines, truncation=True, padding=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        probs = torch.softmax(_sent_model(**enc).logits, dim=-1).cpu().numpy()

    # FinBERT label order is typically [negative, neutral, positive]
    out = []
    labels = ["negative", "neutral", "positive"]
    for h, p in zip(headlines, probs):
        p_neg, p_neu, p_pos = float(p[0]), float(p[1]), float(p[2])
        score = p_pos - p_neg
        out.append(
            ScoreItem(
                headline=h,
                label=labels[int(p.argmax())],
                score=score,
                probabilities={"negative": p_neg, "neutral": p_neu, "positive": p_pos},
            )
        )

    ts = datetime.now(timezone.utc)
    model_version = get_bundle().model_version

    if single:
        return ScoreSingleResponse(result=out[0], timestamp=ts, model_version=model_version)

    return ScoreResponse(results=out, timestamp=ts, model_version=model_version)

@app.get("/sentiment_index/latest", dependencies=[Depends(verify_api_key)])
def sentiment_index_latest():
    df = _require_csv(SENT_PATH, "daily_sentiment_index.csv")
    row = df.iloc[-1].to_dict()
    return {
        "latest": row,
        "timestamp": datetime.now(timezone.utc),
        "model_version": get_bundle().model_version,
    }

@app.post("/forecast/next_day", response_model=ForecastResponse, dependencies=[Depends(verify_api_key)])
def forecast_next_day(req: ForecastRequest):
    b = get_bundle()
    missing = [f for f in b.features if f not in req.features]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing features: {missing}")

    x = pd.DataFrame([{f: req.features[f] for f in b.features}])
    model = b.model  # pipeline or estimator from final_model.pkl

    if hasattr(model, "predict_proba"):
        p_up = float(model.predict_proba(x)[0, 1])
    else:
        # fallback
        pred = int(model.predict(x)[0])
        p_up = 1.0 if pred == 1 else 0.0

    p_down = 1.0 - p_up
    pred_label = "up" if p_up >= 0.5 else "down"

    return ForecastResponse(
        forecast_class=pred_label,
        probabilities={"down": p_down, "up": p_up},
        timestamp=datetime.now(timezone.utc),
        model_version=b.model_version,
    )


@app.get("/dashboard/overview", dependencies=[Depends(verify_api_key)])
def dashboard_overview():
    sentiment_df = _require_csv(SENT_PATH, "daily_sentiment_index.csv")
    latest = sentiment_df.iloc[-1].to_dict()
    score = float(latest.get("mean_sentiment", 0.0))
    headline_volume = int(float(latest.get("headline_volume", 0)))
    latest_date = str(latest.get("date", ""))
    bundle = get_bundle()
    return {
        "sentiment_index": score,
        "market_mood": _market_mood(score),
        "headlines_analyzed": headline_volume,
        "latest_update": latest_date,
        "feed_type": "historical_phase4",
        "model": bundle.config.get("model", "unknown"),
        "horizon_days": bundle.config.get("horizon_days"),
        "model_version": bundle.model_version,
    }


@app.get("/dashboard/live-overview", dependencies=[Depends(verify_api_key)])
def dashboard_live_overview(max_headlines: int = 40):
    return _load_live_snapshot()


@app.get("/dashboard/live-vs-history", dependencies=[Depends(verify_api_key)])
def dashboard_live_vs_history():
    snapshot = _load_live_snapshot()
    windows = snapshot.get("window_summaries", {})
    return {
        "live": windows.get("1d"),
        "recent_7d": windows.get("7d"),
        "recent_30d": windows.get("30d"),
        "feed_type": snapshot.get("feed_type"),
        "source_policy": snapshot.get("source_policy"),
        "gdelt_error": snapshot.get("gdelt_error"),
        "timestamp": datetime.now(timezone.utc),
    }


@app.post("/dashboard/live-refresh", dependencies=[Depends(verify_api_key)])
def dashboard_live_refresh():
    try:
        return build_live_snapshot(
            tokenizer=_tokenizer,
            model=_sent_model,
            model_version=get_bundle().model_version,
            output_path=LIVE_SNAPSHOT_PATH,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Live news refresh failed: {exc}") from exc


@app.get("/dashboard/sentiment-trend", dependencies=[Depends(verify_api_key)])
def dashboard_sentiment_trend():
    sentiment_df = _require_csv(SENT_PATH, "daily_sentiment_index.csv").copy()
    spy_df = _require_csv(SPY_PATH, "spy_stooq.csv").copy()

    sentiment_df["date"] = pd.to_datetime(sentiment_df["date"], errors="coerce")
    spy_df["Date"] = pd.to_datetime(spy_df["Date"], errors="coerce")

    trend_df = sentiment_df.merge(
        spy_df[["Date", "Close"]],
        left_on="date",
        right_on="Date",
        how="left",
    ).drop(columns=["Date"])

    trend_df = trend_df.dropna(subset=["date"]).sort_values("date")
    trend_df["date"] = trend_df["date"].dt.strftime("%Y-%m-%d")

    trend_df = trend_df[["date", "mean_sentiment", "headline_volume", "Close"]].rename(
        columns={"Close": "spy_close"}
    )
    trend_df = trend_df.astype(object).where(pd.notna(trend_df), None)
    records = trend_df.to_dict(orient="records")
    return {"rows": records}


@app.get("/dashboard/headline-volume", dependencies=[Depends(verify_api_key)])
def dashboard_headline_volume():
    sentiment_df = _require_csv(SENT_PATH, "daily_sentiment_index.csv").copy()
    sentiment_df["date"] = pd.to_datetime(sentiment_df["date"], errors="coerce")
    sentiment_df = sentiment_df.dropna(subset=["date"]).sort_values("date")
    sentiment_df["date"] = sentiment_df["date"].dt.strftime("%Y-%m-%d")
    return {
        "rows": sentiment_df[["date", "headline_volume"]].to_dict(orient="records")
    }


@app.get("/dashboard/model-summary", dependencies=[Depends(verify_api_key)])
def dashboard_model_summary():
    metrics_df = _require_csv(METRICS_PATH, "walkforward_metrics.csv")
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
            windows.append(
                {
                    "test_start": row["test_start"],
                    "test_end": row["test_end"],
                    "accuracy": float(row["accuracy"]),
                    "f1": float(row["f1"]),
                    "roc_auc": float(row["roc_auc"]),
                }
            )

    avg_auc = float(selected["roc_auc"].mean()) if not selected.empty else None
    return {
        "selected_config": config,
        "average_roc_auc": avg_auc,
        "windows": windows,
        "model_version": bundle.model_version,
    }

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = perf_counter()
    response = await call_next(request)
    latency_ms = (perf_counter() - start) * 1000
    print(
        f"[api] {request.method} {request.url.path} "
        f"status={response.status_code} latency_ms={latency_ms:.2f}"
    )
    return response
