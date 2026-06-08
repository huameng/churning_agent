"""
Offer-classifier evaluation harness — the portal-offer analog of eval_classifier.

Reuses the same metric core (evaluate_samples + per-label F1) but drives the
offer classifier instead of the post classifier. Cases live in
eval/offer_cases.json, each:

    {"title": "<merchant>", "merchant": "...", "reward": "2000 SB",
     "description": "...", "expected": "ACCEPT" | "SKIP" | "UNCERTAIN"}

Run from churning_agent/:
    uv run python -m churning_agent.eval.eval_offers
"""

import json
import sys

from rich.console import Console

from churning_agent.tools.offer_classifier import PortalOffer, classify_offer
from churning_agent.eval.eval_classifier import evaluate_samples, _print_metrics
from churning_agent.eval.improve_classifier import macro_f1
from churning_agent._paths import PROJECT_ROOT

console = Console()

_CASES_PATH = PROJECT_ROOT / "eval" / "offer_cases.json"


def _classify(sample: dict):
    """classify_fn for evaluate_samples: build a PortalOffer from a case."""
    return classify_offer(PortalOffer(
        merchant=sample.get("merchant") or sample.get("title", ""),
        reward=sample["reward"],
        description=sample.get("description", ""),
    ))


def main() -> int:
    if not _CASES_PATH.exists():
        console.print(
            f"[yellow]No offer cases at eval/{_CASES_PATH.name}. "
            "Add cases there to evaluate the offer classifier.[/yellow]"
        )
        return 0

    cases = json.loads(_CASES_PATH.read_text(encoding="utf-8"))
    console.print(f"Evaluating {len(cases)} offer case(s)...")
    metrics, failures = evaluate_samples(cases, classify_fn=_classify)

    console.print(f"\n[bold]Macro-F1:[/bold] {macro_f1(metrics):.3f}  ({len(failures)} failure(s))")
    _print_metrics(metrics)

    if failures:
        console.print("\n[bold]Failures:[/bold]")
        for f in failures:
            console.print(f"  [red]{f['expected']} → {f['got']}[/red]  {f['title']}")
            console.print(f"    [dim]{f['model_reasoning']}[/dim]")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
