"""
Pool/lane tests: independent lanes get their own tab in one shared context, so
site agents can drive the browser concurrently. Uses a temp profile + headless.
"""

from pathlib import Path

import pytest

from churning_agent.tools import browser

_FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
async def pool(tmp_path, monkeypatch):
    monkeypatch.setattr(browser, "_PROFILE_DIR", tmp_path / "profile")
    monkeypatch.setattr(browser, "_POOL_HEADLESS", True)
    yield browser
    await browser.close_session()


async def test_lanes_get_distinct_pages_in_one_context(pool):
    a = await pool.get_session("topcashback")
    b = await pool.get_session("swagbucks")
    assert a is not b
    assert a.page is not b.page                 # separate tabs
    assert a.page.context is b.page.context     # same shared context (shared login)


async def test_same_lane_returns_same_session(pool):
    a1 = await pool.get_session("topcashback")
    a2 = await pool.get_session("topcashback")
    assert a1 is a2


async def test_lanes_navigate_independently(pool):
    a = await pool.get_session("topcashback")
    b = await pool.get_session("swagbucks")
    # permissive guard so we can use example pages
    a._allow_url = lambda u: True
    b._allow_url = lambda u: True
    await a.navigate((_FIXTURES / "offers.html").as_uri())
    await b.navigate((_FIXTURES / "login.html").as_uri())
    assert a.page.url.endswith("offers.html")
    assert b.page.url.endswith("login.html")    # independent navigation, no collision
