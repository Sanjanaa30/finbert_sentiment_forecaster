import sys
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prefect import flow, task
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.live_news import run_pipeline
from src.database import init_db

MODEL_DIR = ROOT / "models" / "finbert_sentiment"
MODEL_VERSION = "live-snapshot-flow"
ET = ZoneInfo("America/New_York")

# Market-hours schedule (ET) — Mon–Fri only
SCHEDULED_HOURS = [9, 12, 15, 18]  # 9 AM, 12 PM, 3 PM, 6 PM


@task(retries=3, retry_delay_seconds=120, log_prints=True)
def run_snapshot() -> None:
    print("Initialising database...")
    init_db()

    print("Loading FinBERT model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).eval()

    print("Running pipeline (fetch → translate → score → store)...")
    result = run_pipeline(tokenizer=tokenizer, model=model, model_version=MODEL_VERSION)

    si = result.get("sentiment_index")
    mood = result.get("market_mood")
    count = result.get("headlines_analyzed")
    sources = result.get("sources_used", [])
    errors = result.get("source_errors", [])

    print(f"Done — sentiment: {si:.3f if si else 'N/A'}, mood: {mood}, headlines: {count}")
    print(f"Sources used: {sources}")
    if errors:
        print(f"Source errors: {errors}")


@flow(name="live_sentiment_snapshot", log_prints=True)
def live_snapshot_flow() -> None:
    run_snapshot()


def _next_scheduled_run(now: datetime) -> datetime:
    """Return the next scheduled run time after `now` (ET, Mon–Fri only)."""
    candidate = now.replace(minute=0, second=0, microsecond=0)

    for _ in range(8):  # look ahead up to 8 time slots
        for hour in SCHEDULED_HOURS:
            run_time = candidate.replace(hour=hour)
            if run_time > now and run_time.weekday() < 5:
                return run_time
        # move to next day at midnight and try again
        candidate = (candidate + timedelta(days=1)).replace(hour=0)

    # fallback: next Monday 9 AM
    days_ahead = (7 - now.weekday()) % 7 or 7
    return (now + timedelta(days=days_ahead)).replace(
        hour=SCHEDULED_HOURS[0], minute=0, second=0, microsecond=0
    )


if __name__ == "__main__":
    import time

    print("Running live snapshot immediately on startup...")
    live_snapshot_flow()

    while True:
        now = datetime.now(ET)
        next_run = _next_scheduled_run(now)
        sleep_seconds = (next_run - now).total_seconds()
        print(
            f"Next run at {next_run.strftime('%Y-%m-%d %H:%M %Z')} "
            f"({sleep_seconds / 3600:.1f}h away)"
        )
        time.sleep(sleep_seconds)
        live_snapshot_flow()
