"""
Typed offer model shared across the portal pipeline.

Parsers (offer_parsers) emit `Offer`; portal_tools enriches it (usd, seen) and
caches it; the offer classifier reads its reward / requirements / breakdown. One
type end-to-end instead of loosely-shaped dicts, so adding a site or a reward
unit is a localized change.
"""
import re

from pydantic import BaseModel

from . import valuation


class RewardGoal(BaseModel):
    """One payable goal within a tiered offer (e.g. 'Direct Deposit = 35,000 SB')."""
    name: str
    sb: int | None = None


class RewardValue(BaseModel):
    """A reward amount in some unit. `unit` keys into config/valuation.yaml."""
    amount: float
    unit: str

    def to_usd(self) -> float | None:
        return valuation.to_usd(self.amount, self.unit)

    @classmethod
    def parse(cls, text: str) -> "RewardValue | None":
        """Best-effort parse of a raw reward string into amount + unit. Returns
        None when no amount is recognisable. '%' rewards parse but don't convert
        to USD (no item price), which is the honest behaviour."""
        t = text.strip()
        for pattern, unit in (
            (r"([\d,]+(?:\.\d+)?)\s*SB", "SB"),
            (r"£\s*([\d,]+(?:\.\d+)?)", "GBP"),
            (r"\$\s*([\d,]+(?:\.\d+)?)", "USD"),
            (r"([\d,]+(?:\.\d+)?)\s*%", "%"),
        ):
            m = re.search(pattern, t, re.I)
            if m:
                return cls(amount=float(m.group(1).replace(",", "")), unit=unit)
        return None


class Offer(BaseModel):
    site: str
    key: str                       # stable id for de-dup (provider offer id, or "merchant|reward")
    title: str
    reward_text: str               # raw headline, e.g. "up to 23,130 SB", "8% cashback"
    reward: RewardValue | None = None
    is_game: bool = False
    requirements: list[str] = []   # "things to know" / qualifying conditions
    reward_breakdown: list[RewardGoal] = []
    detail: str = ""               # human-readable detail for the classifier
    # Enrichment added by portal_tools when listing:
    usd: float | None = None
    seen: bool = False

    def value_usd(self) -> float | None:
        return self.reward.to_usd() if self.reward else None
