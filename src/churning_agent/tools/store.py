"""SQLite store for classification decisions."""
import sqlite3
from datetime import datetime, timezone

from churning_agent._paths import DATA_DIR
from ._sql import run_select

_DB_PATH = DATA_DIR / "classifications.db"


def _conn() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS classifications (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            url           TEXT NOT NULL,
            title         TEXT NOT NULL,
            label         TEXT NOT NULL,
            reasoning     TEXT NOT NULL,
            estimated_value REAL,
            classified_at TEXT NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_url ON classifications(url)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_label ON classifications(label)")
    conn.commit()
    return conn


def record(url: str, title: str, label: str, reasoning: str, estimated_value: float | None) -> None:
    """Insert one classification row."""
    with _conn() as conn:
        conn.execute(
            "INSERT INTO classifications (url, title, label, reasoning, estimated_value, classified_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (url, title, label, reasoning, estimated_value,
             datetime.now(timezone.utc).isoformat()),
        )


def query_classifications(sql: str) -> dict:
    """
    Run a read-only SQL query against the classifications database and return the results.

    The classifications table schema:
        id              INTEGER  -- autoincrement primary key
        url             TEXT     -- post URL
        title           TEXT     -- post title
        label           TEXT     -- IRRELEVANT, MONEYMAKER, DISCOUNT_MONEYMAKER, WORTHLESS, or UNCERTAIN
        reasoning       TEXT     -- classifier reasoning
        estimated_value REAL     -- estimated dollar value (set for MONEYMAKER/DISCOUNT_MONEYMAKER/WORTHLESS, else NULL)
        classified_at   TEXT     -- ISO 8601 UTC timestamp

    Args:
        sql: A SELECT query to run. Only SELECT statements are permitted.

    Returns:
        Dict with 'columns' (list of column names) and 'rows' (list of row dicts).
        On error, returns {'error': '<message>'}.
    """
    return run_select(_conn(), sql)
