"""
Offer-classifier evaluation (live Gemini calls). Hand-crafted edge cases.

Run from the churning_agent directory:
    uv run python -m churning_agent.eval.eval_offers
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rich.console import Console
from rich.table import Table

from churning_agent.tools.offer_classifier import PortalOffer, classify_offer
from churning_agent.tools.profile import UserProfile

console = Console()

# state/zip only; profile threshold default is $100 unless overridden
_PROFILE = UserProfile(state="FL", zip_code="33613")

# Default profile threshold is $100. Expectations reflect that; edit any you
# disagree with (same philosophy as eval_classifier.py).
SAMPLES: list[dict] = [
    {"merchant": "Dell", "reward": "$150 bonus on a laptop purchase", "expected": "UNCERTAIN",
     "notes": "Above threshold, but merchant-fit unknown on a bare profile -> escalates"},
    {"merchant": "Obscure Vape Shop", "reward": "0.5% cashback", "expected": "SKIP",
     "notes": "Trivial rate, niche merchant"},
    {"merchant": "Expedia", "reward": "£40 bonus on first booking", "expected": "SKIP",
     "notes": "£40 ~ $50, below the $100 threshold"},
    {"merchant": "Local Florist", "reward": "1% cashback", "expected": "SKIP",
     "notes": "Below threshold"},
]


def main() -> int:
    table = Table(title="Offer Classifier Eval", show_lines=True)
    for col in ("Merchant", "Reward", "Expected", "Got", "Value", "Match", "Reasoning"):
        table.add_column(col, max_width=30 if col == "Reasoning" else None)

    passed = failed = 0
    for s in SAMPLES:
        offer = PortalOffer(merchant=s["merchant"], reward=s["reward"])
        result = classify_offer(offer, _PROFILE)
        expected = s.get("expected")
        match = "-"
        if expected:
            if result.label == expected:
                passed += 1
                match = "[green]PASS[/green]"
            else:
                failed += 1
                match = "[red]FAIL[/red]"
        value = f"${result.estimated_value:.0f}" if result.estimated_value else "-"
        table.add_row(s["merchant"], s["reward"], expected or "-", result.label,
                      value, match, result.reasoning)

    console.print(table)
    console.print(f"\n[bold]{passed} passed, {failed} failed[/bold]")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
