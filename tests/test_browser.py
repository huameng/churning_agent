"""
BrowserSession tests run against local file:// fixtures — no network.

The domain guard defaults to sites.is_allowed (which rejects file://), so these
tests inject a permissive allow_url to exercise the mechanics, plus one test
that the real guard blocks off-whitelist pages.
"""

from pathlib import Path

import pytest

from churning_agent.tools.browser import BrowserSession

_FIXTURES = Path(__file__).parent / "fixtures"


def _url(name: str) -> str:
    return (_FIXTURES / name).as_uri()


@pytest.fixture
async def session():
    s = BrowserSession(
        user_data_dir=_FIXTURES / "_profile",
        headless=True,
        allow_url=lambda url: True,  # permissive: exercise mechanics on file://
    )
    await s.start()
    yield s
    await s.stop()


async def test_observe_lists_visible_interactive_elements(session):
    await session.navigate(_url("offers.html"))
    obs = await session.observe()
    names = [e.name for e in obs.elements]
    assert "Log out" in names
    assert names.count("Activate") == 2          # third Activate is display:none
    assert obs.title == "Online Cashback - TopCashback"


async def test_fill_and_read_value(session):
    await session.navigate(_url("login.html"))
    obs = await session.observe()
    email_ref = next(e.ref for e in obs.elements if e.role == "email")
    result = await session.fill(email_ref, "me@example.com")
    assert result.success
    filled = next(e for e in result.observation.elements if e.ref == email_ref)
    assert filled.value == "me@example.com"


async def test_click_unknown_ref_fails_gracefully(session):
    await session.navigate(_url("offers.html"))
    result = await session.click(9999)
    assert not result.success
    assert "Re-observe" in result.error


async def test_observe_sees_into_iframes(session):
    """Offer walls render in iframes; observe() must surface their elements too."""
    await session.navigate(_url("with_iframe.html"))
    obs = await session.observe()
    names = [e.name for e in obs.elements]
    assert "Top Button" in names          # top document
    assert "Claim 2000 SB" in names       # inside the iframe
    assert "Framed Link" in names


async def test_click_element_inside_iframe(session):
    await session.navigate(_url("with_iframe.html"))
    obs = await session.observe()
    framed_ref = next(e.ref for e in obs.elements if e.name == "Claim 2000 SB")
    result = await session.click(framed_ref)        # must route to the iframe's frame
    assert result.success


async def test_observe_handles_real_topcashback_html(session):
    """Smoke test: observe() survives a real (heavy) TopCashback page.

    Loaded via set_content with subresources aborted so it works offline.
    """
    html = (_FIXTURES / "real" / "topcashback_home.html").read_text(encoding="utf-8")

    async def _abort(route):
        await route.abort()

    await session.page.route("**/*", _abort)
    await session.page.set_content(html, wait_until="domcontentloaded")
    obs = await session.observe()
    assert len(obs.elements) > 50                     # real page has many links/buttons
    assert any("log in" in e.name.lower() for e in obs.elements)


async def test_navigate_blocked_for_non_whitelisted():
    # Real guard: file:// is not whitelisted, navigation must refuse.
    s = BrowserSession(user_data_dir=_FIXTURES / "_profile", headless=True)
    await s.start()
    try:
        result = await s.navigate(_url("offers.html"))
        assert not result.success
        assert "not a whitelisted site" in result.error
    finally:
        await s.stop()
