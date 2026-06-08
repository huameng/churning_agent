"""
Unit tests for offer_classifier's pure parts (prompt construction). The actual
LLM classification quality is exercised by eval/eval_offers.py (live calls),
mirroring how the post classifier is evaluated.
"""

from churning_agent.tools.offer_classifier import PortalOffer, _build_prompt
from churning_agent.tools.profile import UserProfile


def _profile() -> UserProfile:
    return UserProfile(state="FL", zip_code="33613")


def test_prompt_includes_offer_and_valuation_table():
    offer = PortalOffer(merchant="Nike", reward="2000 SB", description="new members only")
    prompt = _build_prompt(offer, _profile())
    assert "Nike" in prompt
    assert "2000 SB" in prompt
    assert "new members only" in prompt
    assert "1 SB = $0.01" in prompt          # valuation table injected
    assert "Minimum profit threshold" in prompt  # profile injected


def test_prompt_handles_missing_description():
    offer = PortalOffer(merchant="Expedia", reward="£25 bonus")
    prompt = _build_prompt(offer, _profile())
    assert "(none)" in prompt
