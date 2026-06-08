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
from churning_agent import prompts
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


# Model id and system prompt live in config/prompts/post_classifier.yaml.
_PROMPT = "post_classifier"


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

    cfg = prompts.load(_PROMPT)
    response = _get_client().models.generate_content(
        model=cfg.model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=Classification,
            system_instruction=cfg.system,
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
