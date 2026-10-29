"""
Post classifier: assigns IRRELEVANT, MONEYMAKER, DISCOUNT_MONEYMAKER, WORTHLESS,
or UNCERTAIN to DoC posts.

The `classify` function is the testable core — call it directly without ADK.
The `fetch_and_classify` function is the ADK tool wrapper.
"""

import asyncio
import logging
from datetime import date
from typing import Literal

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

from .profile import ChurningCooldown, UserProfile, load_profile
from .scraper import fetch_offer_section
from . import store
from churning_agent import prompts
from churning_agent.llm import retry_transient
from churning_agent._paths import PROJECT_ROOT

load_dotenv(PROJECT_ROOT / ".env")

logger = logging.getLogger(__name__)

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client()
    return _client


class Eligibility(BaseModel):
    """Structured offer facts extracted by the LLM. Deliberately profile-free:
    these describe the *offer*, and `_decide` compares them to the user profile
    deterministically. See config/prompts/eligibility_extractor.yaml."""
    offer_type: Literal[
        "new_account_bonus", "new_card_signup", "spend_offer_existing_card",
        "portal_cashback", "percentage_discount", "referral", "informational", "other",
    ]
    nationwide: bool = True
    eligible_states: list[str] = []
    requires_business_account: bool = False
    requires_branch_visit: bool = False
    requires_credit_pull: bool = False
    requires_brokerage_account: bool = False
    direct_deposits_required: int | None = None
    lookback_months: int | None = None    # "no bonus/account in the past N months"
    issuer: str | None = None             # bank/issuer behind `brand`, e.g. "Chase"
    brand: str | None = None
    discount_pct: float | None = None
    dollar_savings: float | None = None
    net_profit: float | None = None
    reasoning: str = ""


class Classification(BaseModel):
    label: Literal["IRRELEVANT", "MONEYMAKER", "DISCOUNT_MONEYMAKER", "WORTHLESS", "UNCERTAIN"]
    reasoning: str
    question: str | None = None       # populated only when label=UNCERTAIN
    estimated_value: float | None = None  # dollars, set for MONEYMAKER, DISCOUNT_MONEYMAKER, and WORTHLESS
    extracted: Eligibility | None = None  # the facts the label was decided from (for evals)


# Model id and system prompt live in config/prompts/eligibility_extractor.yaml.
_EXTRACTOR_PROMPT = "eligibility_extractor"


def _extract_eligibility(title: str, content: str) -> Eligibility:
    """The single LLM call: extract offer facts (no judgement, no profile)."""
    cfg = prompts.load(_EXTRACTOR_PROMPT)
    prompt = f"""Post Title: {title}

Post Content:
{content}

Extract the eligibility facts."""
    response = _get_client().models.generate_content(
        model=cfg.model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=Eligibility,
            system_instruction=cfg.system,
        ),
    )
    return Eligibility.model_validate_json(response.text)


# Issuer name variants the extractor and the profile spell differently.
_BRAND_ALIASES = {
    "american express": "amex",
    "bank of america": "bofa",
    "us bank": "u.s. bank",
}


def _norm_brand(s: str) -> str:
    s = s.lower()
    for variant, canonical in _BRAND_ALIASES.items():
        s = s.replace(variant, canonical)
    return s


def _brand_matches(brand: str | None, names: list[str]) -> bool:
    """Does any of `names` (holdings, excluded banks, ...) match `brand`? Matches on
    substring either way, or on all brand tokens (>2 chars) appearing in a name
    — so "Chase IHG Premier" does not match a plain "chase sapphire" holding, but
    "Chase" does. Issuer aliases (Amex/American Express) are normalized first."""
    if not brand:
        return False
    b = _norm_brand(brand)
    tokens = [t for t in b.replace("-", " ").replace(".", " ").split() if len(t) > 2]
    for name in names:
        h = _norm_brand(name).strip()
        if not h:
            continue
        if b in h or h in b:
            return True
        if tokens and all(t in h for t in tokens):
            return True
    return False


# Lookback assumed when a post doesn't state one: most bank bonuses exclude
# anyone who got a bonus within the past 12–24 months.
_DEFAULT_LOOKBACK_MONTHS = 24


def _months_between(earlier: date, later: date) -> int:
    return (later.year - earlier.year) * 12 + (later.month - earlier.month)


def _recent_cooldown(bank: str | None, profile: UserProfile) -> ChurningCooldown | None:
    """The most recent churning cooldown recorded for `bank`, if any."""
    matches = [c for c in profile.churning_cooldowns if _brand_matches(bank, [c.bank])]
    return max(matches, key=lambda c: c.last_bonus_date, default=None)


def _decide(elig: Eligibility, profile: UserProfile, today: date | None = None) -> Classification:
    """Deterministic gating: turn extracted offer facts + the user profile into a
    label. Hard constraints (state, business, excluded bank, already held,
    willingness, direct deposits, churning cooldown) yield IRRELEVANT; a
    cooldown the post's terms can't resolve yields UNCERTAIN with a question;
    surviving offers get a value-based label."""
    p = profile.preferences
    note = elig.reasoning
    today = today or date.today()
    bank = elig.issuer or elig.brand

    def out(label: str, value: float | None, why: str, question: str | None = None) -> Classification:
        reason = f"{why}." + (f" {note}" if note else "")
        return Classification(label=label, reasoning=reason[:500], estimated_value=value,
                              question=question, extracted=elig)

    # --- hard IRRELEVANT gates ---
    if _brand_matches(bank, p.excluded_banks):
        return out("IRRELEVANT", None, f"{bank} is on the user's excluded-banks list")

    if not elig.nationwide and elig.eligible_states:
        states = {s.strip().upper() for s in elig.eligible_states}
        if profile.state.upper() not in states:
            return out("IRRELEVANT", None,
                       f"Limited to {sorted(states)}; user is in {profile.state.upper()}")

    if elig.requires_business_account and not profile.has_business:
        return out("IRRELEVANT", None, "Requires a business account; user has no business")

    # Bank accounts and cards are checked separately: holding a Chase card does
    # not rule out a Chase checking bonus, and vice versa.
    if (elig.offer_type == "spend_offer_existing_card"
            and not _brand_matches(elig.brand, profile.existing_credit_cards)):
        return out("IRRELEVANT", None,
                   f"Spend offer requires holding {elig.brand}, which the user does not")

    if (elig.offer_type == "new_account_bonus"
            and _brand_matches(elig.brand, profile.existing_accounts)):
        return out("IRRELEVANT", None,
                   f"New-account bonus for {elig.brand}, which the user already has an account with")

    if (elig.offer_type == "new_card_signup"
            and _brand_matches(elig.brand, profile.existing_credit_cards)):
        return out("IRRELEVANT", None,
                   f"Sign-up bonus for {elig.brand}, which the user already holds")

    if elig.requires_brokerage_account and not p.willing_to_open_brokerage:
        return out("IRRELEVANT", None, "Requires opening a brokerage account; user is unwilling")

    if (elig.direct_deposits_required is not None
            and elig.direct_deposits_required > p.max_direct_deposit_required):
        return out("IRRELEVANT", None,
                   f"Requires {elig.direct_deposits_required} direct deposits; "
                   f"user allows at most {p.max_direct_deposit_required}")

    if elig.requires_branch_visit and not p.willing_to_visit_branch:
        return out("IRRELEVANT", None, "Requires a branch visit; user is unwilling")

    if elig.requires_credit_pull and not p.willing_to_do_credit_pull:
        return out("IRRELEVANT", None, "Requires a hard credit pull; user is unwilling")

    if elig.offer_type == "referral":
        return out("IRRELEVANT", None, "Referral bonus (requires another person); excluded")

    if elig.offer_type == "informational":
        return out("IRRELEVANT", None, "Informational / non-actionable post")

    # --- churning cooldown (bank bonuses): IRRELEVANT inside a stated lookback,
    # UNCERTAIN when the post doesn't say and the last bonus was recent ---
    if elig.offer_type == "new_account_bonus":
        cooldown = _recent_cooldown(bank, profile)
        if cooldown is not None:
            months = _months_between(cooldown.last_bonus_date, today)
            if elig.lookback_months is not None:
                if months < elig.lookback_months:
                    return out("IRRELEVANT", None,
                               f"Last {bank} bonus was {cooldown.last_bonus_date}, inside the "
                               f"{elig.lookback_months}-month lookback")
            elif months < _DEFAULT_LOOKBACK_MONTHS:
                return out(
                    "UNCERTAIN", None,
                    f"Last {bank} bonus was {cooldown.last_bonus_date} ({months} months ago) "
                    f"and the post doesn't state a lookback window",
                    question=(f"You last got a {bank} bonus on {cooldown.last_bonus_date}. "
                              f"Are you eligible for a new {bank} bonus yet?"),
                )

    # --- value-based labels for offers that pass every gate ---
    # A retail discount is judged purely on the discount thresholds.
    if elig.offer_type == "percentage_discount":
        pct = elig.discount_pct or 0.0
        savings = elig.dollar_savings or 0.0
        if pct >= p.min_discount_pct and savings >= p.min_discount_savings:
            return out("DISCOUNT_MONEYMAKER", savings,
                       f"{pct:.0f}% off and ${savings:.0f} savings meets the discount thresholds")
        return out("WORTHLESS", savings,
                   f"Discount {pct:.0f}% / ${savings:.0f} is below thresholds "
                   f"(need >={p.min_discount_pct:.0f}% AND >=${p.min_discount_savings:.0f})")

    # Spend-gated cashback / statement credits are effectively a discount on the
    # required spend: a 10–15% rebate isn't worth going out of your way for, even
    # if the dollar cap clears the cash threshold. Judge them on the *percentage*
    # (reusing the discount thresholds), not the cap — but a flat sign-up cashback
    # with no spend percentage is just cash, judged on dollars.
    if elig.offer_type in ("spend_offer_existing_card", "portal_cashback"):
        pct = elig.discount_pct or 0.0
        savings = elig.dollar_savings or elig.net_profit or 0.0
        if pct <= 0:
            if savings >= p.min_profit_threshold:
                return out("MONEYMAKER", savings, f"Flat ${savings:.0f} cashback bonus")
            return out("WORTHLESS", savings,
                       f"Flat ${savings:.0f} below the ${p.min_profit_threshold:.0f} threshold")
        if pct >= p.min_discount_pct and savings >= p.min_discount_savings:
            return out("MONEYMAKER", savings,
                       f"{pct:.0f}% back (${savings:.0f}) is substantial cashback")
        return out("WORTHLESS", savings,
                   f"{pct:.0f}% back / ${savings:.0f} on required spend isn't worth pursuing "
                   f"(need >={p.min_discount_pct:.0f}% AND >=${p.min_discount_savings:.0f})")

    # Direct cash offers (bank/brokerage/card bonuses) are judged on net profit.
    profit = elig.net_profit or 0.0
    if profit >= p.min_profit_threshold:
        return out("MONEYMAKER", profit,
                   f"Direct return ~${profit:.0f} meets the ${p.min_profit_threshold:.0f} threshold")
    return out("WORTHLESS", profit,
               f"Net return ~${profit:.0f} is below the ${p.min_profit_threshold:.0f} threshold")


@retry_transient
def classify(title: str, content: str) -> Classification:
    """
    Classify a DoC post. This is the testable core — no ADK dependency.

    Two stages: an LLM extracts structured offer facts (`Eligibility`), then
    `_decide` applies the user profile's hard constraints and value thresholds
    deterministically. Keeping the eligibility judgement out of the LLM stops it
    from contradicting facts it just read (e.g. ignoring a "[MS Only]" title).

    Args:
        title: Post title
        content: Offer section text (pre-extracted; passed as-is to the model)

    Returns:
        Classification with label, reasoning, and estimated_value.
    """
    elig = _extract_eligibility(title, content)
    return _decide(elig, load_profile())


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
    logger.info("doc: classifying %s", title)
    content = await fetch_offer_section(url, title)
    # classify() makes a blocking LLM call; keep it off the event loop so
    # concurrently running site agents aren't stalled.
    result = await asyncio.to_thread(classify, title, content)
    val = f" (~${result.estimated_value:.0f})" if result.estimated_value is not None else ""
    logger.info("doc: %s -> %s%s", title, result.label, val)
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
