import argparse
import os
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


HEADLINES_PATH = Path("data/raw/analyst_ratings_processed.csv")
MODEL_DIR = Path("models/finbert_sentiment")
TEMP_PATH = Path("artifacts/temperature.json")
OUT_PATH = Path("data/processed/daily_sentiment_index.csv")

TEXT_COL = "title"
DATE_COL = "date"
BATCH_SIZE = 64
MAX_LENGTH = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build daily sentiment index from headlines")
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip this many rows before loading sample rows (for paging through data).",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=None,
        help="Load only first N rows from headlines CSV for faster test runs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Inference batch size.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=MAX_LENGTH,
        help="Tokenizer max sequence length.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=(os.cpu_count() or 4),
        help="Number of CPU threads for PyTorch.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append by merging with existing daily CSV (deduplicates dates correctly).",
    )
    return parser.parse_args()


def load_temperature(path: Path) -> float:
    if not path.exists():
        print(f"[warn] {path} not found. Using temperature=1.0")
        return 1.0

    payload = pd.read_json(path, typ="series")
    t = float(payload.get("temperature", 1.0))
    if t <= 0:
        raise ValueError(f"Temperature must be > 0. Got {t}")
    return t


def find_sentiment_indices(model) -> tuple[int, int]:
    id2label = model.config.id2label
    if not id2label:
        raise ValueError("Model config does not contain id2label mapping.")

    normalized = {int(k): str(v).lower() for k, v in id2label.items()}

    def pick_index(candidates: tuple[str, ...]) -> int | None:
        for idx, name in normalized.items():
            if any(token in name for token in candidates):
                return idx
        return None

    pos_idx = pick_index(("positive", "pos"))
    neg_idx = pick_index(("negative", "neg"))

    if pos_idx is None or neg_idx is None:
        raise ValueError(
            f"Could not detect positive/negative indices from id2label={normalized}"
        )

    return pos_idx, neg_idx


def to_daily_moments(daily_df: pd.DataFrame) -> pd.DataFrame:
    n = daily_df["headline_volume"].astype(float)
    mean = daily_df["mean_sentiment"].astype(float)
    std = daily_df["sentiment_dispersion"].fillna(0.0).astype(float)

    # Recover additive sufficient statistics so chunks can be merged safely.
    total = mean * n
    total_sq = (std.pow(2) * (n - 1).clip(lower=0.0)) + (mean.pow(2) * n)

    return pd.DataFrame(
        {
            DATE_COL: daily_df[DATE_COL],
            "_n": n,
            "_sum": total,
            "_sum_sq": total_sq,
        }
    )


def from_daily_moments(moments_df: pd.DataFrame) -> pd.DataFrame:
    n = moments_df["_n"].astype(float)
    total = moments_df["_sum"].astype(float)
    total_sq = moments_df["_sum_sq"].astype(float)

    mean = total / n
    var = pd.Series(0.0, index=moments_df.index, dtype=float)
    mask = n > 1
    var.loc[mask] = ((total_sq[mask] - (total[mask].pow(2) / n[mask])) / (n[mask] - 1)).clip(lower=0.0)
    std = var.pow(0.5)

    return pd.DataFrame(
        {
            DATE_COL: moments_df[DATE_COL],
            "mean_sentiment": mean,
            "sentiment_dispersion": std,
            "headline_volume": n.round().astype(int),
        }
    )


def main() -> None:
    args = parse_args()

    if not HEADLINES_PATH.exists():
        raise FileNotFoundError(f"Missing file: {HEADLINES_PATH}")
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Missing model directory: {MODEL_DIR}")

    torch.set_num_threads(max(1, int(args.threads)))

    print("[1/6] Loading Kaggle headlines columns (title, date)...")
    skiprows = None
    if args.offset > 0:
        # Skip data rows only; keep header row.
        skiprows = range(1, args.offset + 1)

    df = pd.read_csv(
        HEADLINES_PATH,
        usecols=[TEXT_COL, DATE_COL],
        skiprows=skiprows,
        nrows=args.sample_rows,
    )
    df = df.dropna(subset=[TEXT_COL, DATE_COL]).copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce", utc=True).dt.date
    df = df.dropna(subset=[DATE_COL]).copy()

    print(f"      offset={args.offset:,} rows after cleanup: {len(df):,}")

    print("[2/6] Loading fine-tuned FinBERT model + tokenizer...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()

    temp = load_temperature(TEMP_PATH)
    print(f"      temperature: {temp:.6f}")

    pos_idx, neg_idx = find_sentiment_indices(model)
    print(f"      label indices -> positive: {pos_idx}, negative: {neg_idx}")

    print("[3/6] Running batched inference and scoring headlines...")
    print(f"      batch_size={args.batch_size}, max_length={args.max_length}, threads={torch.get_num_threads()}")

    texts = df[TEXT_COL].astype(str).tolist()
    scores: list[float] = []

    with torch.no_grad():
        for start in range(0, len(texts), args.batch_size):
            end = min(start + args.batch_size, len(texts))
            batch = texts[start:end]

            encoded = tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}

            logits = model(**encoded).logits
            probs = torch.softmax(logits / temp, dim=-1)

            batch_scores = (probs[:, pos_idx] - probs[:, neg_idx]).detach().cpu().tolist()
            scores.extend(batch_scores)

            if start % (args.batch_size * 500) == 0:
                print(f"      processed {end:,}/{len(texts):,} headlines")

    df["sentiment_score"] = scores

    print("[4/6] Aggregating to daily sentiment features...")
    daily = (
        df.groupby(DATE_COL, as_index=False)
        .agg(
            mean_sentiment=("sentiment_score", "mean"),
            sentiment_dispersion=("sentiment_score", "std"),
            headline_volume=("sentiment_score", "count"),
        )
        .sort_values(DATE_COL)
    )

    daily["sentiment_dispersion"] = daily["sentiment_dispersion"].fillna(0.0)

    if args.append and OUT_PATH.exists():
        print("[5/6] Appending by merging with existing daily CSV...")
        existing = pd.read_csv(
            OUT_PATH,
            usecols=[DATE_COL, "mean_sentiment", "sentiment_dispersion", "headline_volume"],
        )
        existing[DATE_COL] = pd.to_datetime(existing[DATE_COL], errors="coerce").dt.date
        existing = existing.dropna(subset=[DATE_COL]).copy()

        merged_moments = (
            pd.concat([to_daily_moments(existing), to_daily_moments(daily)], ignore_index=True)
            .groupby(DATE_COL, as_index=False)[["_n", "_sum", "_sum_sq"]]
            .sum()
        )
        daily = from_daily_moments(merged_moments).sort_values(DATE_COL)
        print(f"      merged existing+new into {len(daily):,} daily rows")
    else:
        print("[5/6] No append merge requested (or no existing file).")

    print("[6/6] Adding rolling averages (7-day, 14-day)...")
    daily["sentiment_ma_7"] = daily["mean_sentiment"].rolling(window=7, min_periods=1).mean()
    daily["sentiment_ma_14"] = daily["mean_sentiment"].rolling(window=14, min_periods=1).mean()

    print("[7/7] Saving output CSV...")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(OUT_PATH, index=False)

    print(f"Saved -> {OUT_PATH}")
    print(f"rows={len(daily):,} cols={list(daily.columns)}")


if __name__ == "__main__":
    main()
