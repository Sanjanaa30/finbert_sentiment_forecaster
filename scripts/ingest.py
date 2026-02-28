import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW_HEADLINES = ROOT / "data" / "raw" / "analyst_ratings_processed.csv"
PHRASEBANK = ROOT / "data" / "processed" / "phrasebank_allagree.csv"


def run_cmd(args: list[str]) -> None:
    print(f"[cmd] {' '.join(args)}")
    subprocess.run(args, cwd=ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 6 ingest step")
    parser.add_argument(
        "--fetch-spy",
        action="store_true",
        help="Fetch/update SPY data from Stooq.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("[ingest] checking required source files...")
    missing = [p for p in [RAW_HEADLINES, PHRASEBANK] if not p.exists()]
    if missing:
        names = ", ".join(str(p) for p in missing)
        raise FileNotFoundError(f"Missing required data files: {names}")

    if args.fetch_spy:
        print("[ingest] fetching SPY data...")
        run_cmd([sys.executable, "scripts/fetch_spy_stooq.py"])
    else:
        print("[ingest] fetch-spy disabled, keeping existing SPY file.")

    print("[ingest] done.")


if __name__ == "__main__":
    main()
