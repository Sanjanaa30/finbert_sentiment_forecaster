import argparse
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


DATA_PATH = Path("data/processed/forecast_dataset.csv")
OUT_PATH = Path("artifacts/walkforward_metrics.csv")

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

WINDOWS = [
    ("2020-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
    ("2020-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Walk-forward training with threshold sweep and feature-set comparison."
    )
    parser.add_argument(
        "--thresholds-bps",
        type=str,
        default="0,5,10,15",
        help="Comma-separated target thresholds in basis points. Example: 0,5,10,15",
    )
    parser.add_argument(
        "--horizons",
        type=str,
        default="1,3,5",
        help="Comma-separated forward horizons in trading days. Example: 1,3,5",
    )
    return parser.parse_args()


def parse_thresholds(raw: str) -> list[float]:
    vals: list[float] = []
    for x in raw.split(","):
        x = x.strip()
        if not x:
            continue
        vals.append(float(x))
    if not vals:
        raise ValueError("No valid thresholds parsed from --thresholds-bps")
    return vals


def parse_horizons(raw: str) -> list[int]:
    vals: list[int] = []
    for x in raw.split(","):
        x = x.strip()
        if not x:
            continue
        h = int(x)
        if h <= 0:
            raise ValueError(f"Horizon must be positive. Got: {h}")
        vals.append(h)
    if not vals:
        raise ValueError("No valid horizons parsed from --horizons")
    return vals


def evaluate(y_true, y_pred, y_prob) -> dict[str, float]:
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        out["roc_auc"] = float("nan")
    return out


def build_horizon_frame(df: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    out = df.copy()
    out["close_next_h"] = out["Close"].shift(-horizon_days)
    out["ret_fwd_h"] = (out["close_next_h"] / out["Close"]) - 1.0
    out = out.dropna(subset=["ret_fwd_h"]).copy()
    return out


def apply_threshold(df: pd.DataFrame, threshold_bps: float) -> pd.DataFrame:
    threshold = float(threshold_bps) / 10000.0
    out = df.copy()
    if threshold > 0:
        out = out[out["ret_fwd_h"].abs() > threshold].copy()
    out["target"] = (out["ret_fwd_h"] > 0).astype(int)
    return out


def run_window(
    df: pd.DataFrame,
    features: list[str],
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
) -> list[dict]:
    train = df[(df["Date"] >= train_start) & (df["Date"] <= train_end)].copy()
    test = df[(df["Date"] >= test_start) & (df["Date"] <= test_end)].copy()

    if train.empty or test.empty:
        return []

    X_train, y_train = train[features], train["target"]
    X_test, y_test = test[features], test["target"]

    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = float(neg / max(pos, 1))

    results: list[dict] = []

    logreg = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=42, class_weight="balanced")),
        ]
    )
    logreg.fit(X_train, y_train)
    pred_lr = logreg.predict(X_test)
    prob_lr = logreg.predict_proba(X_test)[:, 1]
    m_lr = evaluate(y_test, pred_lr, prob_lr)
    results.append(
        {
            "model": "logreg",
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "train_rows": len(train),
            "test_rows": len(test),
            **m_lr,
        }
    )

    xgb = XGBClassifier(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
    )
    xgb.fit(X_train, y_train)
    pred_xgb = xgb.predict(X_test)
    prob_xgb = xgb.predict_proba(X_test)[:, 1]
    m_xgb = evaluate(y_test, pred_xgb, prob_xgb)
    results.append(
        {
            "model": "xgboost",
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "train_rows": len(train),
            "test_rows": len(test),
            **m_xgb,
        }
    )

    return results


def main() -> None:
    args = parse_args()
    thresholds = parse_thresholds(args.thresholds_bps)
    horizons = parse_horizons(args.horizons)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing: {DATA_PATH}. Run scripts/build_forecast_dataset.py first.")

    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").copy()

    needed = set(MARKET_FEATURES + SENTIMENT_FEATURES + ["Close"])
    missing = sorted(list(needed - set(df.columns)))
    if missing:
        raise ValueError(f"forecast_dataset missing columns: {missing}")

    print(f"Base dataset rows: {len(df)}")
    print("Sentiment zero-rate:")
    for col in ["mean_sentiment", "sentiment_dispersion", "headline_volume", "sentiment_ma_7", "sentiment_ma_14"]:
        print(f"  {col}: {(df[col] == 0).mean():.4f}")

    all_rows: list[dict] = []
    for horizon_days in horizons:
        df_h = build_horizon_frame(df, horizon_days)
        print(f"\n=== Horizon {horizon_days}d ===")
        for threshold_bps in thresholds:
            df_thr = apply_threshold(df_h, threshold_bps)
            print(
                f"\nThreshold {threshold_bps} bps -> rows={len(df_thr)} "
                f"up-rate={df_thr['target'].mean():.4f}"
            )

            for feature_set_name, feature_cols in FEATURE_SETS.items():
                print(f"  Feature set: {feature_set_name}")
                for train_start, train_end, test_start, test_end in WINDOWS:
                    rows = run_window(df_thr, feature_cols, train_start, train_end, test_start, test_end)
                    if not rows:
                        print(
                            f"    [skip] train={train_start}->{train_end} "
                            f"test={test_start}->{test_end} (insufficient rows)"
                        )
                        continue

                    for r in rows:
                        r["horizon_days"] = horizon_days
                        r["threshold_bps"] = threshold_bps
                        r["feature_set"] = feature_set_name
                        all_rows.append(r)
                        print(
                            f"    {r['model']} {r['test_start'][:4]}: "
                            f"acc={r['accuracy']:.4f} f1={r['f1']:.4f} auc={r['roc_auc']:.4f} "
                            f"(train={r['train_rows']}, test={r['test_rows']})"
                        )

    if not all_rows:
        raise RuntimeError("No windows produced results. Check date coverage in forecast_dataset.csv.")

    out = pd.DataFrame(all_rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"\nSaved: {OUT_PATH} rows={len(out)}")

    summary = (
        out.groupby(["horizon_days", "threshold_bps", "feature_set", "model"], as_index=False)["roc_auc"]
        .mean()
        .sort_values("roc_auc", ascending=False)
    )
    print("\nTop average AUC combinations:")
    print(summary.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
