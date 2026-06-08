"""
seen_offers log tests. Uses a temp DB via monkeypatching the module path so we
don't touch the real data/ store.
"""

import importlib

import pytest


@pytest.fixture
def log(tmp_path, monkeypatch):
    from churning_agent.tools import offer_log
    monkeypatch.setattr(offer_log, "_DB_PATH", tmp_path / "seen.db")
    return offer_log


def test_first_sighting_is_new(log):
    r = log.note_offer("swagbucks", "615336", merchant="SoFi", reward="36500 SB",
                       label="ACCEPT", estimated_value=365.0)
    assert r["new"] is True
    assert r["times_seen"] == 1


def test_repeat_sighting_is_not_new(log):
    log.note_offer("swagbucks", "615336", reward="36500 SB")
    r = log.note_offer("swagbucks", "615336", reward="36500 SB")
    assert r["new"] is False
    assert r["times_seen"] == 2
    assert r["first_seen"]  # preserved from first sighting


def test_same_key_different_site_is_distinct(log):
    assert log.note_offer("swagbucks", "abc")["new"] is True
    assert log.note_offer("topcashback", "abc")["new"] is True


def test_query_returns_recorded_offers(log):
    log.note_offer("swagbucks", "615336", merchant="SoFi", label="ACCEPT", estimated_value=365.0)
    log.note_offer("swagbucks", "999", merchant="Junk", label="SKIP", estimated_value=0.5)
    res = log.query_seen_offers("SELECT merchant FROM seen_offers WHERE label='ACCEPT'")
    assert res["count"] == 1
    assert res["rows"][0]["merchant"] == "SoFi"


def test_query_rejects_non_select(log):
    assert "error" in log.query_seen_offers("DELETE FROM seen_offers")
