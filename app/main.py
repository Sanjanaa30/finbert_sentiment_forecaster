from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
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

ROOT = Path(__file__).resolve().parents[1]
SENT_PATH = ROOT / "data" / "processed" / "daily_sentiment_index.csv"
MODEL_DIR = ROOT / "models" / "finbert_sentiment"

app = FastAPI(title="FinBERT Forecast API")

_tokenizer = None
_sent_model = None

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
    if not SENT_PATH.exists():
        raise HTTPException(status_code=404, detail="daily_sentiment_index.csv not found")
    df = pd.read_csv(SENT_PATH)
    if df.empty:
        raise HTTPException(status_code=404, detail="daily_sentiment_index.csv is empty")
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
