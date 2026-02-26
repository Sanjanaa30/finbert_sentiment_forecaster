from pathlib import Path

import pandas as pd


OUT = Path("data/processed/spy_stooq.csv")
STOOQ_URL = "https://stooq.com/q/d/l/?s=spy.us&i=d"


def main() -> None:
    df = pd.read_csv(STOOQ_URL, parse_dates=["Date"])
    df = df.sort_values("Date")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Saved: {OUT} rows={len(df)}")


if __name__ == "__main__":
    main()
