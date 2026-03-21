"""
SQLite database layer for the FinBERT sentiment pipeline.

Tables:
    raw_headlines      — every headline fetched from any source, before scoring
    scored_headlines   — FinBERT output for each raw headline
    daily_sentiment    — per-day aggregates used by the dashboard
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path


DB_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "sentiment.db"


def _connect(path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def get_conn(path: Path = DB_PATH):
    conn = _connect(path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns that were introduced after the initial schema."""
    migrations = [
        ("daily_sentiment", "top_positive_headline_id", "INTEGER REFERENCES raw_headlines(id)"),
        ("daily_sentiment", "top_negative_headline_id", "INTEGER REFERENCES raw_headlines(id)"),
        ("scored_headlines", "relevance_terms", "TEXT"),
    ]
    existing: dict[str, set[str]] = {}
    for table, col, col_def in migrations:
        if table not in existing:
            rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
            existing[table] = {r["name"] for r in rows}
        if col not in existing[table]:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_def}")
            existing[table].add(col)


def init_db(path: Path = DB_PATH) -> None:
    """Create all tables if they don't exist, then apply any pending migrations."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with get_conn(path) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS raw_headlines (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                headline          TEXT    NOT NULL,
                original_headline TEXT,
                source            TEXT,
                url               TEXT,
                published_at      TEXT,
                fetched_at        TEXT    NOT NULL,
                language          TEXT,
                was_translated    INTEGER NOT NULL DEFAULT 0,
                feed_type         TEXT,
                query_bucket      TEXT,
                UNIQUE(headline)
            );

            CREATE TABLE IF NOT EXISTS scored_headlines (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                headline_id   INTEGER NOT NULL REFERENCES raw_headlines(id),
                label         TEXT    NOT NULL,
                display_label TEXT    NOT NULL,
                score         REAL    NOT NULL,
                abs_score     REAL    NOT NULL,
                p_positive    REAL    NOT NULL,
                p_neutral     REAL    NOT NULL,
                p_negative    REAL    NOT NULL,
                relevance_terms TEXT,
                UNIQUE(headline_id)
            );

            CREATE TABLE IF NOT EXISTS daily_sentiment (
                date                     TEXT PRIMARY KEY,
                mean_sentiment           REAL    NOT NULL,
                headline_volume          INTEGER NOT NULL,
                positive_count           INTEGER NOT NULL DEFAULT 0,
                neutral_count            INTEGER NOT NULL DEFAULT 0,
                negative_count           INTEGER NOT NULL DEFAULT 0,
                positive_share           REAL    NOT NULL DEFAULT 0,
                neutral_share            REAL    NOT NULL DEFAULT 0,
                negative_share           REAL    NOT NULL DEFAULT 0,
                market_mood              TEXT    NOT NULL,
                top_positive_headline_id INTEGER REFERENCES raw_headlines(id),
                top_negative_headline_id INTEGER REFERENCES raw_headlines(id),
                updated_at               TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS pipeline_state (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_raw_published_at
                ON raw_headlines(published_at);
            CREATE INDEX IF NOT EXISTS idx_raw_feed_type
                ON raw_headlines(feed_type);
            CREATE INDEX IF NOT EXISTS idx_scored_headline_id
                ON scored_headlines(headline_id);
            CREATE INDEX IF NOT EXISTS idx_daily_date
                ON daily_sentiment(date);
        """)
        _migrate(conn)


# ---------------------------------------------------------------------------
# Pipeline state
# ---------------------------------------------------------------------------

def get_last_run_at() -> datetime | None:
    """Return the last successful pipeline run time, or None if never run."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT value FROM pipeline_state WHERE key = 'last_run_at'"
        ).fetchone()
    if row:
        return datetime.fromisoformat(row["value"])
    return None


def set_last_run_at(dt: datetime) -> None:
    """Record the time of a successful pipeline run."""
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO pipeline_state (key, value) VALUES ('last_run_at', ?)
               ON CONFLICT(key) DO UPDATE SET value = excluded.value""",
            (dt.isoformat(),),
        )


def insert_raw_headlines(records: list[dict]) -> list[int]:
    """
    Insert raw headlines, skip duplicates (same headline + published_at).
    Returns list of inserted row IDs (None for skipped duplicates).
    """
    fetched_at = datetime.now(timezone.utc).isoformat()
    ids = []
    with get_conn() as conn:
        for record in records:
            try:
                cur = conn.execute(
                    """
                    INSERT INTO raw_headlines
                        (headline, original_headline, source, url, published_at,
                         fetched_at, language, was_translated, feed_type, query_bucket)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.get("headline"),
                        record.get("original_headline"),
                        record.get("source"),
                        record.get("link"),
                        record.get("published_at") or fetched_at,
                        fetched_at,
                        record.get("language"),
                        int(record.get("was_translated", False)),
                        record.get("feed_type"),
                        record.get("query_bucket"),
                    ),
                )
                ids.append(cur.lastrowid)
            except sqlite3.IntegrityError:
                ids.append(None)
    return ids


def insert_scored_headlines(scored_records: list[dict], raw_ids: list[int]) -> None:
    """Store FinBERT scores, skipping headlines that failed insertion."""
    with get_conn() as conn:
        for record, raw_id in zip(scored_records, raw_ids):
            if raw_id is None:
                continue
            try:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO scored_headlines
                        (headline_id, label, display_label, score, abs_score,
                         p_positive, p_neutral, p_negative, relevance_terms)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        raw_id,
                        record.get("label"),
                        record.get("display_label"),
                        record.get("score"),
                        record.get("abs_score"),
                        record.get("positive_prob"),
                        record.get("neutral_prob"),
                        record.get("negative_prob"),
                        ",".join(record.get("relevance_terms") or []),
                    ),
                )
            except sqlite3.IntegrityError:
                pass


def upsert_daily_sentiment(
    day: date,
    records: list[dict],
    raw_ids: list[int | None] | None = None,
) -> None:
    """Aggregate scored records for a day and upsert into daily_sentiment."""
    if not records:
        return

    scores = [r["score"] for r in records]
    labels = [r["label"] for r in records]
    total = len(records)
    pos = labels.count("positive")
    neu = labels.count("neutral")
    neg = labels.count("negative")
    mean = sum(scores) / total
    mood = _market_mood(mean)

    # Determine top positive/negative headline IDs from raw_ids if provided
    top_pos_id = None
    top_neg_id = None
    if raw_ids:
        paired = [(r, rid) for r, rid in zip(records, raw_ids) if rid is not None]
        if paired:
            top_pos_id = max(paired, key=lambda x: x[0]["score"])[1]
            top_neg_id = min(paired, key=lambda x: x[0]["score"])[1]

    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO daily_sentiment
                (date, mean_sentiment, headline_volume,
                 positive_count, neutral_count, negative_count,
                 positive_share, neutral_share, negative_share,
                 market_mood,
                 top_positive_headline_id, top_negative_headline_id,
                 updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(date) DO UPDATE SET
                mean_sentiment           = excluded.mean_sentiment,
                headline_volume          = excluded.headline_volume,
                positive_count           = excluded.positive_count,
                neutral_count            = excluded.neutral_count,
                negative_count           = excluded.negative_count,
                positive_share           = excluded.positive_share,
                neutral_share            = excluded.neutral_share,
                negative_share           = excluded.negative_share,
                market_mood              = excluded.market_mood,
                top_positive_headline_id = excluded.top_positive_headline_id,
                top_negative_headline_id = excluded.top_negative_headline_id,
                updated_at               = excluded.updated_at
            """,
            (
                day.isoformat(),
                mean,
                total,
                pos,
                neu,
                neg,
                pos / total,
                neu / total,
                neg / total,
                mood,
                top_pos_id,
                top_neg_id,
                datetime.now(timezone.utc).isoformat(),
            ),
        )


def fetch_scored_for_window(start_dt: datetime, end_dt: datetime) -> list[dict]:
    """Return scored headlines joined with raw for a given UTC time window."""
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT
                r.id AS raw_id,
                r.headline, r.original_headline, r.source,
                r.url AS link,
                r.published_at, r.language, r.was_translated,
                r.feed_type, r.query_bucket,
                s.label, s.display_label, s.score, s.abs_score,
                s.p_positive, s.p_neutral, s.p_negative,
                s.relevance_terms
            FROM scored_headlines s
            JOIN raw_headlines r ON r.id = s.headline_id
            WHERE COALESCE(r.published_at, r.fetched_at) >= ?
              AND COALESCE(r.published_at, r.fetched_at) <= ?
            ORDER BY COALESCE(r.published_at, r.fetched_at) DESC
            """,
            (start_dt.isoformat(), end_dt.isoformat()),
        ).fetchall()
    return [dict(r) for r in rows]


def count_distinct_days(start_dt: datetime, end_dt: datetime) -> int:
    """Count how many distinct calendar days have scored headlines in the window."""
    with get_conn() as conn:
        result = conn.execute(
            """
            SELECT COUNT(DISTINCT DATE(COALESCE(r.published_at, r.fetched_at))) AS day_count
            FROM scored_headlines s
            JOIN raw_headlines r ON r.id = s.headline_id
            WHERE COALESCE(r.published_at, r.fetched_at) >= ?
              AND COALESCE(r.published_at, r.fetched_at) <= ?
            """,
            (start_dt.isoformat(), end_dt.isoformat()),
        ).fetchone()
    return int(result["day_count"]) if result else 0


def rebuild_daily_sentiment() -> int:
    """
    Rebuild daily_sentiment rows for every day that has scored headlines
    but is missing a daily_sentiment entry. Returns number of days rebuilt.
    """
    with get_conn() as conn:
        missing_days = conn.execute(
            """
            SELECT DISTINCT DATE(COALESCE(r.published_at, r.fetched_at)) AS day
            FROM scored_headlines s
            JOIN raw_headlines r ON r.id = s.headline_id
            WHERE DATE(COALESCE(r.published_at, r.fetched_at)) NOT IN (
                SELECT date FROM daily_sentiment
            )
            ORDER BY day
            """
        ).fetchall()

    rebuilt = 0
    for row in missing_days:
        day_str = row["day"]
        if not day_str:
            continue
        from datetime import datetime as _dt, timezone as _tz
        day_start = _dt.fromisoformat(f"{day_str}T00:00:00+00:00")
        day_end   = _dt.fromisoformat(f"{day_str}T23:59:59+00:00")
        records = fetch_scored_for_window(day_start, day_end)
        if records:
            upsert_daily_sentiment(date.fromisoformat(day_str), records)
            rebuilt += 1
    return rebuilt


def fetch_daily_sentiment_range(start_date: date, end_date: date) -> list[dict]:
    """Return daily_sentiment rows between two dates inclusive."""
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT * FROM daily_sentiment
            WHERE date >= ? AND date <= ?
            ORDER BY date ASC
            """,
            (start_date.isoformat(), end_date.isoformat()),
        ).fetchall()
    return [dict(r) for r in rows]


def fetch_today_top_headlines(limit: int = 5) -> dict[str, list[dict]]:
    """Return top positive, neutral, negative headlines for today."""
    from zoneinfo import ZoneInfo
    ET = ZoneInfo("America/New_York")
    now = datetime.now(ET)
    start = datetime.combine(now.date(), datetime.min.time(), tzinfo=ET).astimezone(timezone.utc)
    end = now.astimezone(timezone.utc)

    rows = fetch_scored_for_window(start, end)
    positive = sorted([r for r in rows if r["label"] == "positive"], key=lambda x: x["score"], reverse=True)
    neutral  = sorted([r for r in rows if r["label"] == "neutral"],  key=lambda x: x["abs_score"])
    negative = sorted([r for r in rows if r["label"] == "negative"], key=lambda x: x["score"])
    return {
        "positive": positive[:limit],
        "neutral":  neutral[:limit],
        "negative": negative[:limit],
    }


def _market_mood(score: float) -> str:
    if score >= 0.40:
        return "Strongly Positive"
    if score >= 0.15:
        return "Slightly Positive"
    if score > -0.15:
        return "Neutral"
    if score > -0.40:
        return "Slightly Negative"
    return "Strongly Negative"
