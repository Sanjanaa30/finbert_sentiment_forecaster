from pathlib import Path

import pandas as pd

OUT = Path("data/processed/spx_stooq.csv")
STOOQ_URL = "https://stooq.com/q/d/l/?s=%5ESPX&i=d"


def main() -> None:
    # Stooq returns CSV with Date,Open,High,Low,Close,Volume.
    df = pd.read_csv(STOOQ_URL, parse_dates=["Date"])
    df = df.sort_values("Date").set_index("Date")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT)
    print(f"Saved: {OUT}  rows={len(df)}")


if __name__ == "__main__":
    main()
