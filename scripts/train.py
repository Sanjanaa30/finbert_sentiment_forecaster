import argparse
import json
import pickle
import subprocess
import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "forecast_dataset.csv"
METRICS_PATH = ROOT / "artifacts" / "walkforward_metrics.csv"
FINAL_CFG_JSON = ROOT / "artifacts" / "final_selected_config.json"
FINAL_CFG_PKL = ROOT / "artifacts" / "final_selected_config.pkl"
FINAL_MODEL_PKL = ROOT / "artifacts" / "final_model.pkl"

SENTIMENT_FEATURES = [
    "mean_sentiment",
    "sentiment_dispersion",
    "headline_volume",
    "sentiment_ma_7",
    "sentiment_ma_14",
    "news_available",
    "mean_sentiment_cf1",
    "mean_sentiment_cf2",
    "sentiment_dispersion_cf1",
    "sentiment_dispersion_cf2",
    "headline_volume_cf1",
    "headline_volume_cf2",
    "mean_sentiment_lag1",
    "mean_sentiment_lag2",
    "mean_sentiment_lag3",
    "sentiment_dispersion_lag1",
    "sentiment_dispersion_lag2",
    "sentiment_dispersion_lag3",
    "headline_volume_lag1",
    "headline_volume_lag2",
    "headline_volume_lag3",
    "ticker_count",
    "median_ticker_sentiment",
    "ticker_sentiment_dispersion",
    "pct_tickers_pos",
    "pct_tickers_neg",
]

MARKET_FEATURES = [
    "ret_1d",
    "ret_5d",
    "ret_10d",
    "ret_20d",
    "vol_5d",
    "vol_10d",
    "vol_20d",
    "ma_5_gap",
    "ma_10_gap",
    "close_to_ma20",
    "ma5_minus_ma20",
]

FEATURE_SETS = {
    "market_only": MARKET_FEATURES,
    "full": SENTIMENT_FEATURES + MARKET_FEATURES,
}


def run_cmd(args: list[str]) -> None:
    print(f"[cmd] {' '.join(args)}")
    subprocess.run(args, cwd=ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 6 train step")
    parser.add_argument("--target-threshold-bps", type=float, default=0.0)
    parser.add_argument("--horizons", type=str, default="10,15,20")
    parser.add_argument("--thresholds-bps", type=str, default="0,5,10,15")
    return parser.parse_args()


def build_target_frame(df: pd.DataFrame, horizon_days: int, threshold_bps: float) -> pd.DataFrame:
    out = df.copy()
    out["close_next_h"] = out["Close"].shift(-horizon_days)
    out["ret_fwd_h"] = (out["close_next_h"] / out["Close"]) - 1.0
    out = out.dropna(subset=["ret_fwd_h"]).copy()

    threshold = threshold_bps / 10000.0
    if threshold > 0:
        out = out[out["ret_fwd_h"].abs() > threshold].copy()
    out["target"] = (out["ret_fwd_h"] > 0).astype(int)
    return out


def fit_model(df: pd.DataFrame, feature_set: str, model_name: str, horizon_days: int, threshold_bps: float):
    feats = FEATURE_SETS[feature_set]
    work = build_target_frame(df, horizon_days, threshold_bps)
    work = work.dropna(subset=feats).copy()
    X, y = work[feats], work["target"]

    if model_name == "logreg":
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, random_state=42, class_weight="balanced")),
            ]
        )
    elif model_name == "xgboost":
        pos = int((y == 1).sum())
        neg = int((y == 0).sum())
        scale_pos_weight = float(neg / max(pos, 1))
        model = XGBClassifier(
            n_estimators=150,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model.fit(X, y)
    return model, feats, len(work)


def main() -> None:
    args = parse_args()

    run_cmd([sys.executable, "scripts/build_forecast_dataset.py", "--target-threshold-bps", str(args.target_threshold_bps)])
    run_cmd([sys.executable, "scripts/train_walkforward.py", "--horizons", args.horizons, "--thresholds-bps", args.thresholds_bps])

    if not METRICS_PATH.exists():
        raise FileNotFoundError(f"Missing {METRICS_PATH}")

    metrics = pd.read_csv(METRICS_PATH)
    summary = (
        metrics.groupby(["horizon_days", "threshold_bps", "feature_set", "model"], as_index=False)["roc_auc"]
        .mean()
        .sort_values("roc_auc", ascending=False)
    )
    best = summary.iloc[0].to_dict()
    best["avg_roc_auc"] = float(best["roc_auc"])
    del best["roc_auc"]
    print(f"[train] best config: {best}")

    FINAL_CFG_JSON.parent.mkdir(parents=True, exist_ok=True)
    FINAL_CFG_JSON.write_text(json.dumps(best, indent=2))
    with open(FINAL_CFG_PKL, "wb") as f:
        pickle.dump(best, f)

    df = pd.read_csv(DATA_PATH)
    model, feats, train_rows = fit_model(
        df=df,
        feature_set=str(best["feature_set"]),
        model_name=str(best["model"]),
        horizon_days=int(best["horizon_days"]),
        threshold_bps=float(best["threshold_bps"]),
    )
    payload = {
        "model": model,
        "features": feats,
        "config": best,
        "train_rows": train_rows,
    }
    with open(FINAL_MODEL_PKL, "wb") as f:
        pickle.dump(payload, f)
    print(f"[train] saved {FINAL_CFG_JSON}")
    print(f"[train] saved {FINAL_MODEL_PKL}")


if __name__ == "__main__":
    main()
