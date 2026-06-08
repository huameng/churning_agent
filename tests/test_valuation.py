from churning_agent.tools import valuation


def test_usd_is_identity():
    assert valuation.to_usd(100, "USD") == 100.0


def test_points_convert():
    assert valuation.to_usd(2000, "SB") == 20.0       # 2000 Swagbucks @ $0.01
    assert valuation.to_usd(10, "GBP") == 12.7


def test_case_insensitive_unit():
    assert valuation.to_usd(2000, "sb") == 20.0


def test_unknown_unit_returns_none():
    assert valuation.to_usd(100, "DOGECOIN") is None


def test_prompt_str_lists_units():
    s = valuation.to_prompt_str()
    assert "SB" in s and "USD" in s
