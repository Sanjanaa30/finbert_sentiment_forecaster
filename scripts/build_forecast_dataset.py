import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SENT_PATH = Path("data/processed/daily_sentiment_index.csv")
SPY_PATH = Path("data/processed/spy_stooq.csv")
OUT_PATH = Path("data/processed/forecast_dataset.csv")

SENT_COLS = [
    "mean_sentiment",
    "sentiment_dispersion",
    "headline_volume",
    "sentiment_ma_7",
    "sentiment_ma_14",
]

CARRY_BASE_COLS = [
    "mean_sentiment",
    "sentiment_dispersion",
    "headline_volume",
]

MARKET_COLS = [
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build market-aligned forecasting dataset")
    parser.add_argument(
        "--target-threshold-bps",
        type=float,
        default=0.0,
        help=(
            "Absolute forward-return threshold in basis points for target labeling. "
            "Example: 10 means +0.10%%/-0.10%% cutoff, and near-flat days are dropped."
        ),
    )
    return parser.parse_args()


def to_moments(df: pd.DataFrame) -> pd.DataFrame:
    n = df["headline_volume"].astype(float)
    mean = df["mean_sentiment"].astype(float)
    std = df["sentiment_dispersion"].fillna(0.0).astype(float)
    total = mean * n
    total_sq = (std.pow(2) * (n - 1).clip(lower=0.0)) + (mean.pow(2) * n)
    return pd.DataFrame({"Date": df["Date"], "_n": n, "_sum": total, "_sum_sq": total_sq})


def from_moments(df: pd.DataFrame) -> pd.DataFrame:
    n = df["_n"].astype(float)
    total = df["_sum"].astype(float)
    total_sq = df["_sum_sq"].astype(float)

    mean = total / n
    var = pd.Series(0.0, index=df.index, dtype=float)
    mask = n > 1
    var.loc[mask] = ((total_sq[mask] - (total[mask].pow(2) / n[mask])) / (n[mask] - 1)).clip(lower=0.0)
    std = var.pow(0.5)

    out = pd.DataFrame(
        {
            "Date": df["Date"],
            "mean_sentiment": mean,
            "sentiment_dispersion": std,
            "headline_volume": n.round().astype(int),
        }
    ).sort_values("Date")
    out["sentiment_ma_7"] = out["mean_sentiment"].rolling(7, min_periods=1).mean()
    out["sentiment_ma_14"] = out["mean_sentiment"].rolling(14, min_periods=1).mean()
    return out


def main() -> None:
    args = parse_args()

    if not SENT_PATH.exists():
        raise FileNotFoundError(f"Missing: {SENT_PATH}")
    if not SPY_PATH.exists():
        raise FileNotFoundError(f"Missing: {SPY_PATH}. Run scripts/fetch_spy_stooq.py first.")

    sent = pd.read_csv(SENT_PATH)
    spy = pd.read_csv(SPY_PATH)

    sent["date"] = pd.to_datetime(sent["date"], errors="coerce")
    spy["Date"] = pd.to_datetime(spy["Date"], errors="coerce")

    sent = sent.dropna(subset=["date"]).copy()
    spy = spy.dropna(subset=["Date"]).copy()
    sent = sent.sort_values("date")
    spy = spy.sort_values("Date")

    # Align sentiment day to next valid trading day to avoid weekend/holiday drop.
    trading_dates = spy[["Date"]].drop_duplicates().sort_values("Date")
    sent_aligned = pd.merge_asof(
        sent,
        trading_dates,
        left_on="date",
        right_on="Date",
        direction="forward",
    )
    sent_aligned = sent_aligned.dropna(subset=["Date"]).copy()

    # Combine overlapping mapped days using additive moments.
    sent_daily = (
        to_moments(sent_aligned)
        .groupby("Date", as_index=False)[["_n", "_sum", "_sum_sq"]]
        .sum()
    )
    sent_daily = from_moments(sent_daily)

    # Keep SPY trading days as the master calendar for forecasting.
    df = spy.merge(sent_daily, on="Date", how="left")
    df = df.sort_values("Date")

    # Track whether a trading day had direct aligned sentiment before filling.
    df["news_available"] = df["mean_sentiment"].notna().astype(int)

    # Short carry-forward features (1 and 2 trading days) for delayed reaction.
    for col in CARRY_BASE_COLS:
        df[f"{col}_cf1"] = df[col].ffill(limit=1).fillna(0.0)
        df[f"{col}_cf2"] = df[col].ffill(limit=2).fillna(0.0)

    # Missing sentiment on a trading day is treated as "no sentiment signal".
    for c in SENT_COLS:
        df[c] = df[c].fillna(0.0)

    # Market features from price history.
    df["ret_1d"] = df["Close"].pct_change(1)
    df["ret_5d"] = df["Close"].pct_change(5)
    df["ret_10d"] = df["Close"].pct_change(10)
    df["ret_20d"] = df["Close"].pct_change(20)
    df["vol_5d"] = df["ret_1d"].rolling(5).std()
    df["vol_10d"] = df["ret_1d"].rolling(10).std()
    df["vol_20d"] = df["ret_1d"].rolling(20).std()
    ma_5 = df["Close"].rolling(5).mean()
    ma_10 = df["Close"].rolling(10).mean()
    ma_20 = df["Close"].rolling(20).mean()
    df["ma_5_gap"] = (df["Close"] / ma_5) - 1.0
    df["ma_10_gap"] = (df["Close"] / ma_10) - 1.0
    df["close_to_ma20"] = (df["Close"] / ma_20) - 1.0
    df["ma5_minus_ma20"] = (ma_5 / ma_20) - 1.0

    # Lagged sentiment features to model delayed market reaction.
    for lag in (1, 2, 3):
        df[f"mean_sentiment_lag{lag}"] = df["mean_sentiment"].shift(lag)
        df[f"sentiment_dispersion_lag{lag}"] = df["sentiment_dispersion"].shift(lag)
        df[f"headline_volume_lag{lag}"] = df["headline_volume"].shift(lag)

    # T+1 target: does tomorrow's close exceed today's close?
    df["close_next"] = df["Close"].shift(-1)
    df["ret_fwd_1d"] = (df["close_next"] / df["Close"]) - 1.0
    threshold = float(args.target_threshold_bps) / 10000.0
    if threshold > 0:
        df["target"] = np.where(
            df["ret_fwd_1d"] > threshold,
            1,
            np.where(df["ret_fwd_1d"] < -threshold, 0, np.nan),
        )
    else:
        df["target"] = (df["ret_fwd_1d"] > 0).astype(int)
    df = df.dropna(subset=["close_next"]).copy()
    if threshold > 0:
        df = df.dropna(subset=["target"]).copy()
    df["target"] = df["target"].astype(int)

    # Remove startup rows where rolling market features are not defined.
    df = df.dropna(subset=MARKET_COLS).copy()
    lag_cols = [c for c in df.columns if c.endswith("_lag1") or c.endswith("_lag2") or c.endswith("_lag3")]
    df = df.dropna(subset=lag_cols).copy()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved: {OUT_PATH} rows={len(df)}")
    print(f"Date range: {df['Date'].min()} -> {df['Date'].max()}")
    print(f"Target threshold (bps): {args.target_threshold_bps}")


if __name__ == "__main__":
    main()
