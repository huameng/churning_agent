"""
Portal-offer classifier: decides whether a cashback/points offer is worth
acting on. The site analog of classifier.py.

`classify_offer` is the testable core. `_build_prompt` is split out so prompt
construction can be unit-tested without an LLM call. ACCEPT/SKIP/UNCERTAIN
mirrors the post classifier's labels and reuses the same UNCERTAIN -> ask the
human -> update_profile -> re-classify loop.

These offers are free to activate and never spend money, so there is no
purchase path to reason about — only whether the offer is worth the user's
attention given their profile.
"""

import logging
from typing import Literal

from google.genai import types
from pydantic import BaseModel

from . import valuation
from .classifier import _get_client
from .profile import UserProfile, load_profile
from churning_agent import prompts
from churning_agent.llm import retry_transient

logger = logging.getLogger(__name__)

# Model id and system prompt live in config/prompts/offer_classifier.yaml.
_PROMPT = "offer_classifier"


class PortalOffer(BaseModel):
    merchant: str
    reward: str                  # raw reward text, e.g. "8% cashback", "2000 SB", "£25 bonus"
    description: str = ""
    url: str = ""


class OfferDecision(BaseModel):
    label: Literal["ACCEPT", "SKIP", "UNCERTAIN"]
    reasoning: str
    question: str | None = None            # populated only when UNCERTAIN
    estimated_value: float | None = None   # USD


def _build_prompt(offer: PortalOffer, profile: UserProfile) -> str:
    return f"""User Profile:
{profile.to_prompt_str()}

{valuation.to_prompt_str()}

Offer:
  Merchant: {offer.merchant}
  Reward: {offer.reward}
  Details: {offer.description or '(none)'}

Classify this offer."""


@retry_transient
def classify_offer(offer: PortalOffer, profile: UserProfile | None = None) -> OfferDecision:
    """Classify a single portal offer. Testable core — no ADK dependency."""
    profile = profile or load_profile()
    cfg = prompts.load(_PROMPT)
    response = _get_client().models.generate_content(
        model=cfg.model,
        contents=_build_prompt(offer, profile),
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=OfferDecision,
            system_instruction=cfg.system,
        ),
    )
    return OfferDecision.model_validate_json(response.text)


def classify_portal_offer(merchant: str, reward: str, description: str = "", url: str = "") -> dict:
    """
    ADK tool: classify one cashback/points offer for the user.

    Args:
        merchant: Store/brand name (e.g. "Nike").
        reward: Raw reward text (e.g. "8% cashback", "2000 SB", "£25 bonus").
        description: Any extra offer detail/terms.
        url: Offer URL, if known.

    Returns:
        Dict with label (ACCEPT/SKIP/UNCERTAIN), reasoning, question (UNCERTAIN only),
        estimated_value (USD), merchant.
    """
    logger.info("portal: classifying %s (%s)", merchant, reward)
    offer = PortalOffer(merchant=merchant, reward=reward, description=description, url=url)
    result = classify_offer(offer)
    val = f" (~${result.estimated_value:.0f})" if result.estimated_value else ""
    logger.info("portal: %s -> %s%s", merchant, result.label, val)
    return {
        "merchant": merchant,
        "label": result.label,
        "reasoning": result.reasoning,
        "question": result.question,
        "estimated_value": result.estimated_value,
    }
