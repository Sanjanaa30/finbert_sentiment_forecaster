import argparse
import json
import pickle
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SENTIMENT_CSV = ROOT / "data" / "processed" / "daily_sentiment_index.csv"
SENTIMENT_PKL = ROOT / "artifacts" / "latest_sentiment_index.pkl"
STATE_JSON = ROOT / "artifacts" / "score_progress.json"


def run_cmd(args: list[str]) -> None:
    print(f"[cmd] {' '.join(args)}")
    subprocess.run(args, cwd=ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 6 scoring step")
    parser.add_argument(
        "--run-scoring",
        action="store_true",
        help="Run expensive headline scoring. If omitted, reuses existing daily sentiment index.",
    )
    parser.add_argument("--skip-if-exists", action="store_true", help="Skip scoring if sentiment CSV already exists.")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--sample-rows", type=int, default=None)
    parser.add_argument(
        "--auto-offset",
        action="store_true",
        help="Use and update an internal offset pointer so each run processes the next chunk.",
    )
    parser.add_argument(
        "--step-rows",
        type=int,
        default=64,
        help="Rows per run when --auto-offset is enabled and --sample-rows is not provided.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--append", action="store_true")
    return parser.parse_args()


def load_next_offset() -> int:
    if not STATE_JSON.exists():
        return 0
    payload = json.loads(STATE_JSON.read_text(encoding="utf-8"))
    return int(payload.get("next_offset", 0))


def save_next_offset(next_offset: int) -> None:
    STATE_JSON.parent.mkdir(parents=True, exist_ok=True)
    STATE_JSON.write_text(json.dumps({"next_offset": int(next_offset)}, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()

    should_run = args.run_scoring
    if args.skip_if_exists and (not args.run_scoring) and SENTIMENT_CSV.exists():
        should_run = False

    offset = int(args.offset)
    sample_rows = args.sample_rows
    if args.auto_offset:
        offset = load_next_offset()
        if sample_rows is None:
            sample_rows = int(args.step_rows)
        print(f"[score] auto-offset enabled -> offset={offset}, sample_rows={sample_rows}")

    if should_run:
        print("[score] running sentiment index build...")
        cmd = [
            sys.executable,
            "scripts/build_sentiment_index.py",
            "--offset",
            str(offset),
            "--batch-size",
            str(args.batch_size),
            "--max-length",
            str(args.max_length),
            "--threads",
            str(args.threads),
        ]
        if sample_rows is not None:
            cmd += ["--sample-rows", str(sample_rows)]
        if args.append:
            cmd += ["--append"]
        run_cmd(cmd)
        if args.auto_offset and sample_rows is not None:
            save_next_offset(offset + int(sample_rows))
            print(f"[score] updated next_offset -> {offset + int(sample_rows)}")
    else:
        print("[score] reusing existing daily_sentiment_index.csv")

    if not SENTIMENT_CSV.exists():
        raise FileNotFoundError(f"Missing {SENTIMENT_CSV}")

    print("[score] saving latest sentiment artifact as pickle...")
    df = pd.read_csv(SENTIMENT_CSV)
    SENTIMENT_PKL.parent.mkdir(parents=True, exist_ok=True)
    with open(SENTIMENT_PKL, "wb") as f:
        pickle.dump(df, f)
    print(f"[score] saved {SENTIMENT_PKL}")


if __name__ == "__main__":
    main()
