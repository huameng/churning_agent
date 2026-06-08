"""
Post classifier: assigns IRRELEVANT, MONEYMAKER, or WORTHLESS to DoC posts.

The `classify` function is the testable core — call it directly without ADK.
The `fetch_and_classify` function is the ADK tool wrapper.
"""

from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .profile import load_profile
from .scraper import fetch_offer_section
from . import store
from churning_agent._paths import PROJECT_ROOT

load_dotenv(PROJECT_ROOT / ".env")

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client()
    return _client


class Classification(BaseModel):
    label: Literal["IRRELEVANT", "MONEYMAKER", "DISCOUNT_MONEYMAKER", "WORTHLESS", "UNCERTAIN"]
    reasoning: str
    question: str | None = None       # populated only when label=UNCERTAIN
    estimated_value: float | None = None  # dollars, set for MONEYMAKER, DISCOUNT_MONEYMAKER, and WORTHLESS


_SYSTEM_PROMPT = """You classify Doctor of Credit blog posts for a user interested in making money, or getting free stuff.

Labels:
- IRRELEVANT: The post is not actionable for this user. This is the highest priority label and should be chosen over other correct labels.
Examples: 
* geographic restrictions the user doesn't meet,
* requires a card or account the user already has (for a new-account bonus),
* post is informational/educational/a discussion thread/expired deal,
* requires something the user isn't willing to do.
- MONEYMAKER: The post describes an offer the user can profitably act on with a direct cash or cash-equivalent return.
Do not include referral bonuses, since those require action from other humans. 
Examples: 
* bank checking/savings bonuses the user is eligible for,
* brokerage bonuses,
* signup bonuses clearly above the profit threshold.
- DISCOUNT_MONEYMAKER: A substantial discount on something worth buying — must meet BOTH thresholds from the user profile (minimum % off AND minimum $ savings).
Examples:
* something worth $100 for $10.
* 60% off a $50 item.
* a $20 item for free, or $1.
- WORTHLESS: The user is technically eligible but it's not worth pursuing. 
Examples: 
* net profit below the minimum threshold,
* minor discount or gift card deals that don't meet the DISCOUNT_MONEYMAKER thresholds,
* too much friction for the reward,
* very long lock-up period.
- UNCERTAIN: The profile does not contain enough information to classify confidently. Use ONLY when a specific unknown fact would change the label — not as a hedge on borderline cases. 

When making a decision for which label to apply, use this workflow:
* First, check which states are in the title of the post. If the user is NOT in one of those states, classify as IRRELEVANT.
* Next, check if the post is something other than a direct offer. If it is, classify as IRRELEVANT.
* Next, check if the post is for a credit card, bank account, or other service which the user does not have. If it is, classify as IRRELEVANT.
* Once those are done, evaluate how much profit the user can make from the offer, and what the cost of that profit is.
* If the offer is high enough profit, and no cost, classify as MONEYMAKER.
* If the offer is high enough profit, with meaningful cost, classify as DISCOUNT_MONEYMAKER.
* Otherwise, classify as WORTHLESS.
* If the offer is confusing in some way due to not following any of these patterns, classify as UNCERTAIN and provide the questions which, when answered, would allow you to confidently classify it.
When label is UNCERTAIN, populate `question` with a single, specific yes/no or short-answer question for the human that would resolve the uncertainty. Do not ask vague questions.

Return JSON with:
- label: one of IRRELEVANT, MONEYMAKER, WORTHLESS, UNCERTAIN
- reasoning: 1-2 sentences explaining the classification
- question: the specific question to ask the human (UNCERTAIN only, otherwise null)
- estimated_value: estimated net dollar value (cash profit for MONEYMAKER, dollar savings for DISCOUNT_MONEYMAKER, or dollar value for WORTHLESS). Null for IRRELEVANT and UNCERTAIN.
"""


@retry(
    retry=retry_if_exception_type(genai_errors.ServerError),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    stop=stop_after_attempt(5),
    reraise=True,
)
def classify(title: str, content: str) -> Classification:
    """
    Classify a DoC post. This is the testable core — no ADK dependency.

    Args:
        title: Post title
        content: Offer section text (pre-extracted; passed as-is to the model)

    Returns:
        Classification with label, reasoning, and estimated_value.
    """
    profile = load_profile()

    prompt = f"""User Profile:
{profile.to_prompt_str()}

Post Title: {title}

Post Content:
{content}

Classify this post."""

    response = _get_client().models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=Classification,
            system_instruction=_SYSTEM_PROMPT,
        ),
    )

    return Classification.model_validate_json(response.text)


async def fetch_and_classify(title: str, url: str) -> dict:
    """
    Fetch the 'The Offer' section of a DoC post and classify it.
    Uses a local cache so repeat calls for the same URL don't re-fetch.

    Args:
        title: The post title
        url: The post URL

    Returns:
        Dict with label, reasoning, estimated_value (dollars for MONEYMAKER and WORTHLESS, null for IRRELEVANT),
        question (UNCERTAIN only), title, url.
    """
    content = await fetch_offer_section(url, title)
    result = classify(title, content)
    if result.label != "UNCERTAIN":
        store.record(url, title, result.label, result.reasoning, result.estimated_value)
    return {
        "url": url,
        "title": title,
        "label": result.label,
        "reasoning": result.reasoning,
        "question": result.question,
        "estimated_value": result.estimated_value,
    }
