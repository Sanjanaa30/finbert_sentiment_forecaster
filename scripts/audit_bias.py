from pathlib import Path

import pandas as pd


FORECAST_PATH = Path("data/processed/forecast_dataset.csv")
HEADLINES_PATH = Path("data/raw/analyst_ratings_processed.csv")

SENT_COLS = [
    "mean_sentiment",
    "sentiment_dispersion",
    "headline_volume",
    "sentiment_ma_7",
    "sentiment_ma_14",
]


def audit_forecast_dataset() -> None:
    if not FORECAST_PATH.exists():
        raise FileNotFoundError(f"Missing {FORECAST_PATH}. Run build_forecast_dataset.py first.")

    df = pd.read_csv(FORECAST_PATH)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).copy()
    df["year"] = df["Date"].dt.year

    print("=== Forecast Dataset Audit ===")
    print(f"Rows: {len(df):,}")
    print(f"Date range: {df['Date'].min().date()} -> {df['Date'].max().date()}")

    # 1) Target balance overall + by year
    if "target" in df.columns:
        up_rate = float(df["target"].mean())
        print(f"\n1) Target Balance")
        print(f"Overall up-rate (target=1): {up_rate:.4f}")
        yr = df.groupby("year")["target"].mean().rename("up_rate").reset_index()
        print("By year:")
        for _, r in yr.iterrows():
            print(f"  {int(r['year'])}: {float(r['up_rate']):.4f}")
    else:
        print("\n1) Target Balance: target column missing")

    # 2) Sentiment zero coverage
    print("\n2) Sentiment Zero-Rate")
    for col in SENT_COLS:
        if col in df.columns:
            z = float((df[col] == 0).mean())
            print(f"  {col}: {z:.4f}")
    if all(c in df.columns for c in SENT_COLS):
        any_zero = (df[SENT_COLS] == 0).all(axis=1)
        print(f"  all sentiment cols zero on day: {float(any_zero.mean()):.4f}")


def audit_ticker_concentration() -> None:
    if not HEADLINES_PATH.exists():
        print(f"\n=== Ticker Concentration ===\nMissing {HEADLINES_PATH} (skipped)")
        return

    # load only needed cols
    hd = pd.read_csv(HEADLINES_PATH, usecols=["stock", "date"])
    hd = hd.dropna(subset=["stock"]).copy()

    vc = hd["stock"].value_counts()
    total = int(vc.sum())
    top10 = vc.head(10)
    top10_share = float(top10.sum() / max(total, 1))

    print("\n=== Ticker Concentration ===")
    print(f"Total headlines with ticker: {total:,}")
    print(f"Unique tickers: {vc.shape[0]:,}")
    print(f"Top-10 ticker share: {top10_share:.4f}")
    print("Top 10 tickers:")
    for ticker, cnt in top10.items():
        print(f"  {ticker}: {int(cnt):,}")


def main() -> None:
    audit_forecast_dataset()
    audit_ticker_concentration()


if __name__ == "__main__":
    main()
