from __future__ import annotations

import argparse
from pathlib import Path

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.live_news import build_live_snapshot


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "finbert_sentiment"
DEFAULT_OUTPUT = ROOT / "artifacts" / "live_sentiment_snapshot.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a GDELT-backed live sentiment snapshot.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Where to save the snapshot JSON.",
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default="manual-live-snapshot",
        help="Model version label to store in the output snapshot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("[live] loading tokenizer and fine-tuned model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).eval()

    print("[live] fetching GDELT recent news windows and scoring headlines...")
    snapshot = build_live_snapshot(
        tokenizer=tokenizer,
        model=model,
        model_version=args.model_version,
        output_path=args.output,
    )

    print(f"[live] saved snapshot to {args.output}")
    print(
        "[live] 1d summary -> "
        f"feed={snapshot.get('feed_type')} "
        f"sentiment={snapshot.get('sentiment_index')} "
        f"headlines={snapshot.get('headlines_analyzed')}"
    )


if __name__ == "__main__":
    main()
