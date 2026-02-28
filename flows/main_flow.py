import argparse
import subprocess
import sys
from pathlib import Path

from prefect import flow, task


ROOT = Path(__file__).resolve().parents[1]


def run_cmd(args: list[str]) -> None:
    print(f"[cmd] {' '.join(args)}")
    subprocess.run(args, cwd=ROOT, check=True)


@task(retries=2, retry_delay_seconds=10)
def ingest_task(fetch_spy: bool) -> None:
    cmd = [sys.executable, "scripts/ingest.py"]
    if fetch_spy:
        cmd.append("--fetch-spy")
    run_cmd(cmd)


@task(retries=1, retry_delay_seconds=10)
def score_task(
    run_scoring: bool,
    skip_if_exists: bool,
    auto_offset: bool,
    step_rows: int,
    offset: int,
    sample_rows: int | None,
    batch_size: int,
    max_length: int,
    threads: int,
    append: bool,
) -> None:
    cmd = [sys.executable, "scripts/score.py"]
    if run_scoring:
        cmd.append("--run-scoring")
    if skip_if_exists:
        cmd.append("--skip-if-exists")
    if auto_offset:
        cmd.append("--auto-offset")
        cmd += ["--step-rows", str(step_rows)]
    cmd += ["--offset", str(offset), "--batch-size", str(batch_size), "--max-length", str(max_length), "--threads", str(threads)]
    if sample_rows is not None:
        cmd += ["--sample-rows", str(sample_rows)]
    if append:
        cmd.append("--append")
    run_cmd(cmd)


@task(retries=1, retry_delay_seconds=10)
def train_task(target_threshold_bps: float, horizons: str, thresholds_bps: str) -> None:
    run_cmd(
        [
            sys.executable,
            "scripts/train.py",
            "--target-threshold-bps",
            str(target_threshold_bps),
            "--horizons",
            horizons,
            "--thresholds-bps",
            thresholds_bps,
        ]
    )


@flow(name="finbert_sentiment_forecaster_batch")
def main_flow(
    fetch_spy: bool = True,
    run_scoring: bool = False,
    skip_if_exists: bool = True,
    auto_offset: bool = False,
    step_rows: int = 64,
    offset: int = 0,
    sample_rows: int | None = None,
    batch_size: int = 64,
    max_length: int = 64,
    threads: int = 8,
    append: bool = False,
    target_threshold_bps: float = 0.0,
    horizons: str = "10,15,20",
    thresholds_bps: str = "0,5,10,15",
) -> None:
    ingest_task(fetch_spy=fetch_spy)
    score_task(
        run_scoring=run_scoring,
        skip_if_exists=skip_if_exists,
        auto_offset=auto_offset,
        step_rows=step_rows,
        offset=offset,
        sample_rows=sample_rows,
        batch_size=batch_size,
        max_length=max_length,
        threads=threads,
        append=append,
    )
    train_task(
        target_threshold_bps=target_threshold_bps,
        horizons=horizons,
        thresholds_bps=thresholds_bps,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Prefect batch pipeline for finbert_sentiment_forecaster.")
    parser.add_argument("--fetch-spy", action="store_true")
    parser.add_argument("--run-scoring", action="store_true", help="Enable expensive headline scoring step.")
    parser.add_argument("--no-skip-if-exists", action="store_true", help="Force scoring even if sentiment file exists.")
    parser.add_argument("--auto-offset", action="store_true", help="Automatically process next chunk each run.")
    parser.add_argument("--step-rows", type=int, default=64, help="Chunk size for --auto-offset.")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--sample-rows", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--target-threshold-bps", type=float, default=0.0)
    parser.add_argument("--horizons", type=str, default="10,15,20")
    parser.add_argument("--thresholds-bps", type=str, default="0,5,10,15")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main_flow(
        fetch_spy=args.fetch_spy,
        run_scoring=args.run_scoring,
        skip_if_exists=not args.no_skip_if_exists,
        auto_offset=args.auto_offset,
        step_rows=args.step_rows,
        offset=args.offset,
        sample_rows=args.sample_rows,
        batch_size=args.batch_size,
        max_length=args.max_length,
        threads=args.threads,
        append=args.append,
        target_threshold_bps=args.target_threshold_bps,
        horizons=args.horizons,
        thresholds_bps=args.thresholds_bps,
    )
