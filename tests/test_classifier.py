"""
Post-classifier tests. The decision rules (`_decide`) are pure, so most of the
behaviour is tested directly against hand-built Eligibility facts; classify()
is tested end to end with the LLM client mocked. Live extraction quality is
covered by eval/eval_classifier.py.
"""

from datetime import date

import pytest

from churning_agent.tools import classifier
from churning_agent.tools.classifier import Classification, Eligibility, _decide, classify
from churning_agent.tools.profile import ChurningCooldown, Preferences, UserProfile

TODAY = date(2026, 6, 15)


def _profile(**overrides) -> UserProfile:
    prefs = overrides.pop("preferences", {})
    fields = dict(
        state="FL",
        zip_code="33613",
        existing_accounts=["bank of america"],
        existing_credit_cards=["chase sapphire preferred", "amex gold"],
        preferences=Preferences(**prefs),
    )
    fields.update(overrides)
    return UserProfile(**fields)


def _bonus(**facts) -> Eligibility:
    """A $400 nationwide bank bonus unless overridden."""
    return Eligibility(**{"offer_type": "new_account_bonus", "brand": "Chase",
                          "issuer": "Chase", "net_profit": 400.0, **facts})


def _label(elig: Eligibility, profile: UserProfile | None = None) -> str:
    return _decide(elig, profile or _profile(), today=TODAY).label


# ── hard gates ────────────────────────────────────────────────────────────────

def test_eligible_bank_bonus_is_moneymaker():
    result = _decide(_bonus(), _profile(), today=TODAY)
    assert result.label == "MONEYMAKER"
    assert result.estimated_value == 400.0
    assert result.extracted is not None


def test_state_restriction():
    assert _label(_bonus(nationwide=False, eligible_states=["MN", "WI"])) == "IRRELEVANT"
    assert _label(_bonus(nationwide=False, eligible_states=["fl"])) == "MONEYMAKER"


def test_excluded_bank_is_irrelevant():
    profile = _profile(preferences={"excluded_banks": ["chase"]})
    assert _label(_bonus(), profile) == "IRRELEVANT"


def test_excluded_bank_matches_card_by_issuer():
    profile = _profile(preferences={"excluded_banks": ["wells fargo"]})
    card = Eligibility(offer_type="new_card_signup", brand="Active Cash",
                       issuer="Wells Fargo", net_profit=200.0)
    assert _label(card, profile) == "IRRELEVANT"


def test_holding_a_card_does_not_block_a_checking_bonus():
    # The user holds a Chase card but has no Chase bank account.
    assert _label(_bonus()) == "MONEYMAKER"


def test_existing_account_blocks_new_account_bonus():
    profile = _profile(existing_accounts=["chase checking"])
    assert _label(_bonus(), profile) == "IRRELEVANT"


def test_existing_account_does_not_block_card_signup():
    profile = _profile(existing_accounts=["chase checking"], existing_credit_cards=[])
    card = Eligibility(offer_type="new_card_signup", brand="Chase Sapphire Preferred",
                       issuer="Chase", net_profit=750.0)
    assert _label(card, profile) == "MONEYMAKER"


def test_card_signup_blocked_only_for_the_same_product():
    held = Eligibility(offer_type="new_card_signup", brand="Chase Sapphire Preferred",
                       issuer="Chase", net_profit=750.0)
    other = Eligibility(offer_type="new_card_signup", brand="Chase IHG Premier",
                        issuer="Chase", net_profit=1400.0)
    assert _label(held) == "IRRELEVANT"
    assert _label(other) == "MONEYMAKER"


def test_amex_alias_matches_held_card():
    offer = Eligibility(offer_type="spend_offer_existing_card", brand="American Express",
                        discount_pct=100.0, dollar_savings=50.0)
    assert _label(offer) == "MONEYMAKER"


def test_spend_offer_requires_holding_the_card():
    offer = Eligibility(offer_type="spend_offer_existing_card", brand="Citi",
                        discount_pct=100.0, dollar_savings=50.0)
    assert _label(offer) == "IRRELEVANT"


def test_brokerage_requires_willingness():
    elig = _bonus(brand="Fidelity", issuer="Fidelity", requires_brokerage_account=True)
    assert _label(elig) == "MONEYMAKER"
    unwilling = _profile(preferences={"willing_to_open_brokerage": False})
    assert _label(elig, unwilling) == "IRRELEVANT"


def test_direct_deposit_limit():
    assert _label(_bonus(direct_deposits_required=2)) == "MONEYMAKER"
    assert _label(_bonus(direct_deposits_required=3)) == "IRRELEVANT"


@pytest.mark.parametrize("facts, pref", [
    ({"requires_business_account": True}, {}),
    ({"requires_branch_visit": True}, {"willing_to_visit_branch": False}),
    ({"requires_credit_pull": True}, {"willing_to_do_credit_pull": False}),
    ({"offer_type": "referral"}, {}),
    ({"offer_type": "informational"}, {}),
])
def test_other_hard_gates(facts, pref):
    assert _label(_bonus(**facts), _profile(preferences=pref)) == "IRRELEVANT"


# ── churning cooldowns ───────────────────────────────────────────────────────

def _cooled(last: date) -> UserProfile:
    return _profile(churning_cooldowns=[ChurningCooldown(bank="chase", last_bonus_date=last)])


def test_cooldown_inside_stated_lookback_is_irrelevant():
    assert _label(_bonus(lookback_months=24), _cooled(date(2025, 6, 1))) == "IRRELEVANT"


def test_cooldown_outside_stated_lookback_passes():
    assert _label(_bonus(lookback_months=12), _cooled(date(2025, 1, 1))) == "MONEYMAKER"


def test_recent_cooldown_without_stated_lookback_is_uncertain():
    result = _decide(_bonus(), _cooled(date(2025, 6, 1)), today=TODAY)
    assert result.label == "UNCERTAIN"
    assert "2025-06-01" in result.question


def test_old_cooldown_without_stated_lookback_passes():
    assert _label(_bonus(), _cooled(date(2023, 1, 1))) == "MONEYMAKER"


def test_cooldown_for_another_bank_is_ignored():
    profile = _profile(churning_cooldowns=[ChurningCooldown(bank="citi", last_bonus_date=date(2026, 5, 1))])
    assert _label(_bonus(), profile) == "MONEYMAKER"


# ── value thresholds ─────────────────────────────────────────────────────────

def test_cash_bonus_below_threshold_is_worthless():
    assert _label(_bonus(net_profit=50.0)) == "WORTHLESS"


def test_discount_needs_both_thresholds():
    good = Eligibility(offer_type="percentage_discount", discount_pct=60.0, dollar_savings=30.0)
    small = Eligibility(offer_type="percentage_discount", discount_pct=60.0, dollar_savings=10.0)
    assert _label(good) == "DISCOUNT_MONEYMAKER"
    assert _label(small) == "WORTHLESS"


def test_spend_cashback_judged_on_percentage():
    rebate = Eligibility(offer_type="portal_cashback", discount_pct=12.5, dollar_savings=250.0)
    flat = Eligibility(offer_type="portal_cashback", dollar_savings=150.0)
    assert _label(rebate) == "WORTHLESS"
    assert _label(flat) == "MONEYMAKER"


# ── classify() end to end, LLM mocked ────────────────────────────────────────

class _FakeModels:
    def __init__(self, text):
        self._text = text
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return type("R", (), {"text": self._text})()


class _FakeClient:
    def __init__(self, text):
        self.models = _FakeModels(text)


def test_classify_extracts_then_decides(monkeypatch):
    client = _FakeClient(_bonus().model_dump_json())
    monkeypatch.setattr(classifier, "load_profile", _profile)
    monkeypatch.setattr(classifier, "_get_client", lambda: client)

    result = classify("Chase $400 Checking Bonus", "Open a checking account...")

    assert isinstance(result, Classification)
    assert result.label == "MONEYMAKER"
    assert result.estimated_value == 400.0
    prompt = client.models.calls[0]["contents"]
    assert "Chase $400 Checking Bonus" in prompt
    assert "Open a checking account" in prompt
