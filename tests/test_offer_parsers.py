"""
Swagbucks offer parser test — runs the extraction JS against a synthetic
unified_card fixture (offline, no login). Validates reward/title parsing and
dedupe by title.
"""

from pathlib import Path

import pytest

from churning_agent.tools.browser import BrowserSession
from churning_agent.tools.offer_parsers import parse_offers, parse_swagbucks_api

_FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
async def session():
    s = BrowserSession(user_data_dir=_FIXTURES / "_profile", headless=True, allow_url=lambda u: True)
    await s.start()
    yield s
    await s.stop()


async def test_parses_swagbucks_cards(session):
    await session.navigate((_FIXTURES / "swagbucks_offers.html").as_uri())
    offers = await parse_offers(session, "swagbucks")
    # two distinct cards (the duplicate Raid card is deduped by title)
    assert len(offers) == 2
    by_title = {o.title: o.reward_text for o in offers}
    assert by_title["Raid Shadow Legends Desktop"] == "up to 23,130 SB"
    assert by_title["SoFi Relay - Get Rewarded"] == "400 SB"
    # reward unit is inferred from the text, so the DOM path is valued too
    sofi = next(o for o in offers if o.title == "SoFi Relay - Get Rewarded")
    assert sofi.reward.amount == 400 and sofi.reward.unit == "SB"


def test_parse_swagbucks_api_full_list():
    """The API payload is the authoritative full list — parse all items with
    stable keys, reward from totalPoints, and game flag."""
    payload = {
        "totalItems": 3,
        "isLastPage": True,
        "content": [
            {"id": 1673018, "productName": "Shortical", "totalPoints": 5425,
             "useEarnUpTo": True, "isGame": True, "shortDescription": "Watch dramas"},
            {"id": 999, "productName": "SoFi Checking", "totalPoints": 36500,
             "useEarnUpTo": False, "isGame": False, "shortDescription": "Open account"},
            {"id": 0, "productName": "", "totalPoints": 10},  # junk: dropped (no key/title)
        ],
    }
    offers = parse_swagbucks_api(payload)
    assert len(offers) == 2
    sofi = next(o for o in offers if o.title == "SoFi Checking")
    assert sofi.key == "999"
    assert sofi.reward_text == "36,500 SB"
    assert sofi.reward.amount == 36500 and sofi.reward.unit == "SB"
    assert sofi.is_game is False
    shortical = next(o for o in offers if o.title == "Shortical")
    assert shortical.reward_text == "up to 5,425 SB"   # useEarnUpTo -> "up to"


def test_parse_swagbucks_api_captures_requirements_and_events():
    """The classifier needs to know what it takes to earn the reward — capture
    thingsToKnow and the per-goal events breakdown into `detail`."""
    payload = {"content": [{
        "id": 999, "productName": "SoFi Checking", "totalPoints": 36500,
        "thingsToKnow": ["Must be a new user", "Make a $400 direct deposit within 45 days"],
        "events": [
            {"name": "Bank Account Created", "flatPoints": 1500, "payable": True},
            {"name": "Money Direct Deposit", "flatPoints": 35000, "payable": True},
            {"name": "Internal non-payable", "flatPoints": 0, "payable": False},
        ],
    }]}
    o = parse_swagbucks_api(payload)[0]
    assert len(o.reward_breakdown) == 2                # non-payable dropped
    assert o.reward_breakdown[1].name == "Money Direct Deposit"
    assert o.reward_breakdown[1].sb == 35000
    assert "$400 direct deposit" in o.detail
    assert "Money Direct Deposit: 35000 SB" in o.detail


async def test_unknown_site_has_no_parser(session):
    await session.navigate((_FIXTURES / "offers.html").as_uri())
    assert await parse_offers(session, "topcashback") == []
