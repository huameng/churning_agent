"""SQLite store for classification decisions."""
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

_DB_PATH = Path(__file__).parent.parent / "data" / "classifications.db"


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
        label           TEXT     -- IRRELEVANT, MONEYMAKER, or WORTHLESS
        reasoning       TEXT     -- classifier reasoning
        estimated_value REAL     -- estimated dollar value (MONEYMAKERs only, else NULL)
        classified_at   TEXT     -- ISO 8601 UTC timestamp

    Args:
        sql: A SELECT query to run. Only SELECT statements are permitted.

    Returns:
        Dict with 'columns' (list of column names) and 'rows' (list of row dicts).
        On error, returns {'error': '<message>'}.
    """
    sql = sql.strip()
    if not sql.upper().startswith("SELECT"):
        return {"error": "Only SELECT queries are permitted."}
    try:
        conn = _conn()
        cursor = conn.execute(sql)
        columns = [d[0] for d in cursor.description]
        rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
        return {"columns": columns, "rows": rows, "count": len(rows)}
    except sqlite3.Error as e:
        return {"error": str(e)}
