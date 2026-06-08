"""
Log of portal offers we've already surfaced, so the agent stops re-reporting the
same offers run after run. Keyed by (site, offer_key); the agent records each
offer it evaluates and reports only the ones that are new.

Separate from store.py (which logs DoC post classifications) — different shape,
different lifecycle.
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

_DB_PATH = Path(__file__).parent.parent / "data" / "seen_offers.db"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _conn() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS seen_offers (
            site            TEXT NOT NULL,
            offer_key       TEXT NOT NULL,
            merchant        TEXT,
            reward          TEXT,
            label           TEXT,
            estimated_value REAL,
            first_seen      TEXT NOT NULL,
            last_seen       TEXT NOT NULL,
            times_seen      INTEGER NOT NULL DEFAULT 1,
            PRIMARY KEY (site, offer_key)
        )
    """)
    return conn


def note_offer(
    site: str,
    offer_key: str,
    merchant: str = "",
    reward: str = "",
    label: str = "",
    estimated_value: float | None = None,
) -> dict:
    """
    Record that we've seen an offer. Returns whether it's new.

    `offer_key` must be stable across runs — prefer the provider's offer id
    (e.g. Swagbucks offerID from the /offer/click URL); fall back to
    "merchant|reward" when no id is available.

    Returns dict: {new: bool, times_seen: int, first_seen: ISO8601}.
    """
    now = _now()
    with _conn() as conn:
        row = conn.execute(
            "SELECT first_seen, times_seen FROM seen_offers WHERE site=? AND offer_key=?",
            (site, offer_key),
        ).fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO seen_offers (site, offer_key, merchant, reward, label, "
                "estimated_value, first_seen, last_seen, times_seen) VALUES (?,?,?,?,?,?,?,?,1)",
                (site, offer_key, merchant, reward, label, estimated_value, now, now),
            )
            return {"new": True, "times_seen": 1, "first_seen": now}
        conn.execute(
            "UPDATE seen_offers SET last_seen=?, times_seen=times_seen+1, "
            "merchant=?, reward=?, label=?, estimated_value=? WHERE site=? AND offer_key=?",
            (now, merchant, reward, label, estimated_value, site, offer_key),
        )
        return {"new": False, "times_seen": row["times_seen"] + 1, "first_seen": row["first_seen"]}


def is_seen(site: str, offer_key: str) -> bool:
    """Read-only check: have we recorded this offer before?"""
    with _conn() as conn:
        return conn.execute(
            "SELECT 1 FROM seen_offers WHERE site=? AND offer_key=?", (site, offer_key)
        ).fetchone() is not None


def query_seen_offers(sql: str) -> dict:
    """
    Run a read-only SELECT against the seen_offers table.

    Schema: seen_offers(site, offer_key, merchant, reward, label,
    estimated_value, first_seen, last_seen, times_seen).
    Returns {columns, rows, count} or {error}.
    """
    sql = sql.strip()
    if not sql.upper().startswith("SELECT"):
        return {"error": "Only SELECT queries are permitted."}
    try:
        cursor = _conn().execute(sql)
        columns = [d[0] for d in cursor.description]
        rows = [dict(zip(columns, r)) for r in cursor.fetchall()]
        return {"columns": columns, "rows": rows, "count": len(rows)}
    except sqlite3.Error as e:
        return {"error": str(e)}
