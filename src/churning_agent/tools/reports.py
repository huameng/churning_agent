"""
Deterministic SQL reporting for the root agent.

The model writes the SQL (full flexibility), but the rows are rendered to a table
and returned with `skip_summarization` set — so ADK treats the tool output as the
final response and never sends the rows back through the model. That's the perf
fix: a `SELECT *` over classifications is ~23K tokens, and previously the model
re-ingested and re-emitted all of it. Now result size doesn't affect latency.

Both stores are attached into one connection so a single query can span them
(JOIN/UNION across DoC posts and portal offers).
"""
import logging
import sqlite3

from google.adk.tools import ToolContext

from . import offer_log, store

logger = logging.getLogger(__name__)


def _cell(col: str, value) -> str:
    if value is None:
        return ""
    if col == "estimated_value" and isinstance(value, (int, float)):
        return f"${value:,.0f}"
    return str(value).replace("|", "\\|")


def _table(columns: list[str], rows: list[dict]) -> str:
    """Render rows as a markdown table."""
    if not rows:
        return "_no rows_"
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = "\n".join("| " + " | ".join(_cell(c, r.get(c)) for c in columns) + " |" for r in rows)
    return "\n".join([header, sep, body])


def _open() -> sqlite3.Connection:
    """A connection with both stores attached: doc.classifications + portals.seen_offers."""
    for ensure in (store._conn, offer_log._conn):   # make sure both files + tables exist
        c = ensure()
        c.commit()
        c.close()
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("ATTACH DATABASE ? AS doc", (str(store._DB_PATH),))
    conn.execute("ATTACH DATABASE ? AS portals", (str(offer_log._DB_PATH),))
    return conn


def run_query(sql: str, tool_context: ToolContext = None) -> str:
    """
    Run a read-only SQL SELECT over the recorded results and render the rows as a
    table shown DIRECTLY to the user. The rows are NOT sent back through the model,
    so this stays fast no matter how many there are. Use it for any "show me / list"
    request about already-found opportunities.

    Two tables are queryable in one statement (you may JOIN or UNION them):
      doc.classifications(id, url, title, label, reasoning, estimated_value, classified_at)
        Doctor of Credit posts. Moneymakers: label IN ('MONEYMAKER', 'DISCOUNT_MONEYMAKER').
      portals.seen_offers(site, offer_key, merchant, reward, label, estimated_value,
                          first_seen, last_seen, times_seen)
        TopCashback / Swagbucks offers. Moneymakers: label = 'ACCEPT'.

    Notes:
      - Only a single SELECT statement is allowed.
      - doc.classifications is append-only (a row per run); GROUP BY url to dedupe,
        e.g. SELECT title, MAX(estimated_value) AS estimated_value, url ... GROUP BY url.

    Args:
        sql: the SELECT to run (may reference doc.classifications and portals.seen_offers).
    """
    if tool_context is not None:
        tool_context.actions.skip_summarization = True   # render directly; don't re-summarise

    sql = sql.strip().rstrip(";").strip()
    if not sql.upper().startswith("SELECT"):
        return "Only a single SELECT statement is allowed."
    try:
        conn = _open()
        cur = conn.execute(sql)
        columns = [d[0] for d in cur.description]
        rows = [dict(zip(columns, r)) for r in cur.fetchall()]
        conn.close()
    except (sqlite3.Error, sqlite3.Warning) as e:
        return f"query error: {e}"

    logger.info("report: query returned %d row(s)", len(rows))
    return f"**{len(rows)} row(s)**\n\n{_table(columns, rows)}"
