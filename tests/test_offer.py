"""Unit tests for the typed Offer model — reward parsing and USD valuation."""

from churning_agent.tools.offer import Offer, RewardValue


def test_reward_value_to_usd():
    assert RewardValue(amount=2000, unit="SB").to_usd() == 20.0
    assert RewardValue(amount=10, unit="GBP").to_usd() == 12.7
    assert RewardValue(amount=8, unit="%").to_usd() is None   # can't value % without a price


def test_reward_parse_units():
    assert RewardValue.parse("up to 23,130 SB") == RewardValue(amount=23130, unit="SB")
    assert RewardValue.parse("£25 bonus") == RewardValue(amount=25, unit="GBP")
    assert RewardValue.parse("$30 back") == RewardValue(amount=30, unit="USD")
    assert RewardValue.parse("8% cashback") == RewardValue(amount=8, unit="%")
    assert RewardValue.parse("free shipping") is None


def test_offer_value_usd():
    sb = Offer(site="swagbucks", key="1", title="X", reward_text="2000 SB",
               reward=RewardValue(amount=2000, unit="SB"))
    assert sb.value_usd() == 20.0
    # An unparseable reward leaves the offer unvalued rather than crashing the sort.
    bare = Offer(site="topcashback", key="2", title="Y", reward_text="see merchant")
    assert bare.value_usd() is None
