"""
Tests for run_query: the model writes arbitrary SELECTs (across both attached
stores), results render to a table, and skip_summarization is set so the rows
never go back through the model.
"""

import pytest

from churning_agent.tools import offer_log, reports, store


class _Actions:
    skip_summarization = False


class _ToolContext:
    def __init__(self):
        self.actions = _Actions()


@pytest.fixture
def stores(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "_DB_PATH", tmp_path / "classifications.db")
    monkeypatch.setattr(offer_log, "_DB_PATH", tmp_path / "seen_offers.db")
    store.record("http://x", "Chase $300", "MONEYMAKER", "good bonus", 300.0)
    store.record("http://y", "Boring post", "IRRELEVANT", "nope", None)
    offer_log.note_offer("swagbucks", "1", merchant="SoFi", label="ACCEPT", estimated_value=365.0)


def test_cell_formats_value_and_none():
    assert reports._cell("estimated_value", 300.0) == "$300"
    assert reports._cell("estimated_value", None) == ""
    assert reports._cell("title", "A | B") == "A \\| B"   # pipe escaped for markdown


def test_run_query_renders_and_skips_summarization(stores):
    ctx = _ToolContext()
    out = reports.run_query(
        "SELECT title, estimated_value FROM doc.classifications "
        "WHERE label = 'MONEYMAKER' ORDER BY estimated_value DESC",
        tool_context=ctx,
    )
    assert ctx.actions.skip_summarization is True       # rendered directly, not summarised
    assert "1 row(s)" in out
    assert "Chase $300" in out and "$300" in out
    assert "Boring post" not in out


def test_run_query_can_union_both_stores(stores):
    out = reports.run_query(
        "SELECT title AS item, estimated_value FROM doc.classifications WHERE label='MONEYMAKER' "
        "UNION ALL "
        "SELECT merchant AS item, estimated_value FROM portals.seen_offers WHERE label='ACCEPT' "
        "ORDER BY estimated_value DESC"
    )
    assert "SoFi" in out and "Chase $300" in out        # spans both attached databases
    assert "2 row(s)" in out


def test_run_query_rejects_non_select(stores):
    assert "Only a single SELECT" in reports.run_query("DELETE FROM doc.classifications")


def test_run_query_reports_sql_error(stores):
    assert "query error" in reports.run_query("SELECT nope FROM doc.classifications")
