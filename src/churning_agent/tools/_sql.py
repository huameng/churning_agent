"""Shared read-only SQL helper for the per-store query tools.

store.query_classifications and offer_log.query_seen_offers both expose a
SELECT-only query surface to the agent; this is the one implementation they
share. reports.run_query is separate — it attaches both stores and renders a
table for direct display rather than returning rows to the model.
"""
import sqlite3


def run_select(conn: sqlite3.Connection, sql: str) -> dict:
    """Run a single read-only SELECT and return {columns, rows, count}, or
    {error} on a rejected/failed query. Closes `conn` before returning."""
    sql = sql.strip()
    if not sql.upper().startswith("SELECT"):
        conn.close()
        return {"error": "Only SELECT queries are permitted."}
    try:
        cursor = conn.execute(sql)
        columns = [d[0] for d in cursor.description]
        rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
        return {"columns": columns, "rows": rows, "count": len(rows)}
    except sqlite3.Error as e:
        return {"error": str(e)}
    finally:
        conn.close()
