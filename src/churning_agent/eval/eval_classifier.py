"""
Classifier evaluation harness.

Cases come from two sources:
  1. eval/cases.json — generated from cached posts via eval/generate_cases.py.
     Each entry has `expected` set to the label the classifier produced at generation
     time, making this a regression suite. Edit any `expected` value you disagree with.
  2. SAMPLES below — hand-crafted edge cases that test specific behaviours.

Run from churning_agent/:
    uv run python -m churning_agent.eval.eval_classifier
    uv run python -m churning_agent.eval.eval_classifier --strict   # exit 1 on first wrong label
    uv run python -m churning_agent.eval.eval_classifier --manual-only  # skip cases.json
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.table import Table

from churning_agent.tools.classifier import classify

console = Console()

from churning_agent._paths import PROJECT_ROOT

_EVAL_DIR = PROJECT_ROOT / "eval"
_CASES_PATH = _EVAL_DIR / "cases.json"
_RESULTS_DIR = _EVAL_DIR / "results"

SAMPLES: list[dict] = []


def run_eval(samples: list[dict], stop_on_unexpected: bool = False) -> int:
    table = Table(title="Classifier Evaluation", show_lines=True)
    table.add_column("Title", max_width=38)
    table.add_column("Expected", style="cyan", no_wrap=True)
    table.add_column("Got", style="bold", no_wrap=True)
    table.add_column("Value")
    table.add_column("Match", no_wrap=True)
    table.add_column("Reasoning", max_width=45)

    passed = failed = skipped = 0
    run_at = datetime.now(timezone.utc).isoformat()
    case_results = []

    for sample in samples:
        console.print(f"[dim]Classifying: {sample['title'][:70]}[/dim]")

        result = classify(sample["title"], sample["content"])
        expected = sample.get("expected")

        if expected is None:
            skipped += 1
            match_str = "[dim]-[/dim]"
            outcome = "skipped"
        elif result.label == expected:
            passed += 1
            match_str = "[green]PASS[/green]"
            outcome = "pass"
        else:
            failed += 1
            match_str = "[red]FAIL[/red]"
            outcome = "fail"

        value_str = f"${result.estimated_value:.0f}" if result.estimated_value is not None else "-"

        table.add_row(
            sample["title"][:38],
            expected or "-",
            result.label,
            value_str,
            match_str,
            result.reasoning,
        )

        case_results.append({
            "title": sample["title"],
            "url": sample.get("url"),
            "expected": expected,
            "got": result.label,
            "outcome": outcome,
            "estimated_value": result.estimated_value,
            "model_reasoning": result.reasoning,
            "human_reasoning": sample.get("human_reasoning"),
        })

        if expected and result.label != expected and stop_on_unexpected:
            console.print(table)
            console.print(f"\n[red]FAILED[/red]: {sample['title']}")
            console.print(f"  Expected [cyan]{expected}[/cyan], got [bold]{result.label}[/bold]")
            console.print(f"  Model reasoning: {result.reasoning}")
            if sample.get("human_reasoning"):
                console.print(f"  Human reasoning: {sample['human_reasoning']}")
            confusion = _confusion_matrix(case_results)
            _write_results(run_at, case_results, passed, failed, skipped, confusion, _compute_metrics(confusion))
            return 1

    console.print(table)
    console.print(
        f"\n[bold]Results:[/bold] {passed} passed, {failed} failed, {skipped} without expected label"
        f" (out of {len(samples)} samples)"
    )
    if skipped:
        console.print("[dim]Set 'expected' on samples above to track regressions.[/dim]")

    confusion = _confusion_matrix(case_results)
    metrics = _compute_metrics(confusion)
    _print_confusion_matrix(confusion)
    _print_metrics(metrics)

    path = _write_results(run_at, case_results, passed, failed, skipped, confusion, metrics)
    console.print(f"[dim]Results saved to {path}[/dim]")

    failures = [c for c in case_results if c["outcome"] == "fail"]
    if failures:
        # Enrich failures with content from the original samples for easier review
        content_by_url = {s.get("url"): s.get("content", "") for s in samples}
        for f in failures:
            f["content"] = content_by_url.get(f["url"], "")
        failures_path = _write_failures(failures)
        console.print(f"[dim]{len(failures)} failure(s) written to {failures_path}[/dim]")

    return 1 if failed else 0


def _confusion_matrix(cases: list[dict]) -> dict:
    """Build {expected_label: {got_label: count}} for cases with an expected label."""
    labels = ["IRRELEVANT", "MONEYMAKER", "DISCOUNT_MONEYMAKER", "WORTHLESS", "UNCERTAIN"]
    matrix = {exp: {got: 0 for got in labels} for exp in labels}
    for case in cases:
        if case["outcome"] == "skipped":
            continue
        exp = case["expected"]
        got = case["got"]
        if exp not in matrix:
            matrix[exp] = {got: 0 for got in labels}
        if got not in matrix[exp]:
            matrix[exp][got] = 0
        matrix[exp][got] += 1
    # Drop rows where expected label had zero cases
    return {exp: row for exp, row in matrix.items() if sum(row.values()) > 0}


def _print_confusion_matrix(confusion: dict) -> None:
    if not confusion:
        return
    labels = ["IRRELEVANT", "MONEYMAKER", "DISCOUNT_MONEYMAKER", "WORTHLESS", "UNCERTAIN"]
    present = [l for l in labels if l in confusion]

    matrix_table = Table(title="Confusion Matrix (rows=expected, cols=predicted)", show_lines=True)
    matrix_table.add_column("Expected \\ Got", style="cyan")
    for col in present:
        matrix_table.add_column(col[:10], justify="center")

    for exp in present:
        row = []
        for got in present:
            count = confusion[exp].get(got, 0)
            if exp == got:
                row.append(f"[green]{count}[/green]" if count > 0 else "[dim]0[/dim]")
            else:
                row.append(f"[red]{count}[/red]" if count > 0 else "[dim]0[/dim]")
        matrix_table.add_row(exp, *row)

    console.print()
    console.print(matrix_table)


def _compute_metrics(confusion: dict) -> dict:
    """Compute per-label precision, recall, and F1 from the confusion matrix."""
    labels = list(confusion.keys())
    all_labels = set(labels)
    for row in confusion.values():
        all_labels |= set(row.keys())

    metrics = {}
    for label in sorted(all_labels):
        tp = confusion.get(label, {}).get(label, 0)
        fp = sum(confusion.get(exp, {}).get(label, 0) for exp in all_labels if exp != label)
        fn = sum(confusion.get(label, {}).get(got, 0) for got in all_labels if got != label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        metrics[label] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}

    return metrics


def _print_metrics(metrics: dict) -> None:
    if not metrics:
        return
    t = Table(title="Per-Label Metrics", show_lines=True)
    t.add_column("Label", style="cyan")
    t.add_column("Precision", justify="right")
    t.add_column("Recall", justify="right")
    t.add_column("F1", justify="right")
    t.add_column("Support", justify="right")

    for label, m in sorted(metrics.items()):
        f1_str = f"{m['f1']:.2f}"
        f1_colored = f"[green]{f1_str}[/green]" if m["f1"] >= 0.8 else (
                     f"[yellow]{f1_str}[/yellow]" if m["f1"] >= 0.5 else f"[red]{f1_str}[/red]")
        t.add_row(
            label,
            f"{m['precision']:.2f}",
            f"{m['recall']:.2f}",
            f1_colored,
            str(m["support"]),
        )
    console.print()
    console.print(t)


def _write_failures(failures: list[dict]) -> Path:
    """Overwrite eval/failures.json with the latest set of failed cases, ready for correction."""
    path = Path(__file__).parent / "failures.json"
    annotated = [
        {
            "title": f["title"],
            "url": f["url"],
            "content": f["content"],
            "expected": f["expected"],
            "got": f["got"],
            "model_reasoning": f["model_reasoning"],
            "human_reasoning": f.get("human_reasoning"),
        }
        for f in failures
    ]
    path.write_text(json.dumps(annotated, indent=2), encoding="utf-8")
    return path


def _write_results(run_at: str, cases: list[dict], passed: int, failed: int, skipped: int, confusion: dict, metrics: dict) -> Path:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    filename = run_at.replace(":", "").replace("-", "").replace("T", "_")[:15] + ".json"
    path = _RESULTS_DIR / filename
    path.write_text(
        json.dumps({
            "run_at": run_at,
            "summary": {"passed": passed, "failed": failed, "skipped": skipped, "total": passed + failed + skipped},
            "confusion_matrix": confusion,
            "metrics": metrics,
            "cases": cases,
        }, indent=2),
        encoding="utf-8",
    )
    return path


def _post_classify(sample: dict):
    """Default classify_fn for evaluate_samples: the DoC post classifier."""
    return classify(sample["title"], sample["content"])


def evaluate_samples(samples: list[dict], classify_fn=_post_classify) -> tuple[dict, list[dict]]:
    """
    Run a classifier over samples without any printing, returning
    (metrics_dict, failures_list). `classify_fn(sample)` returns an object with
    .label/.reasoning/.estimated_value, so this harness works for any classifier
    (the post classifier by default, the offer classifier via eval_offers).
    Used by the improvement loop.
    """
    case_results = []
    for sample in samples:
        result = classify_fn(sample)
        expected = sample.get("expected")
        outcome = (
            "skip" if expected is None
            else "pass" if result.label == expected
            else "fail"
        )
        case_results.append({
            "title": sample["title"],
            "url": sample.get("url"),
            "expected": expected,
            "got": result.label,
            "outcome": outcome,
            "model_reasoning": result.reasoning,
            "human_reasoning": sample.get("human_reasoning"),
        })

    confusion = _confusion_matrix(case_results)
    metrics = _compute_metrics(confusion)
    failures = [c for c in case_results if c["outcome"] == "fail"]
    return metrics, failures


def _load_generated_cases() -> list[dict]:
    if not _CASES_PATH.exists():
        return []
    return json.loads(_CASES_PATH.read_text(encoding="utf-8"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the post classifier")
    parser.add_argument("--strict", action="store_true", help="Exit 1 on first unexpected label")
    parser.add_argument("--manual-only", action="store_true", help="Skip cases.json, run only SAMPLES")
    args = parser.parse_args()

    all_samples = list(SAMPLES)
    if not args.manual_only:
        generated = _load_generated_cases()
        if generated:
            console.print(f"[dim]Loaded {len(generated)} cases from {_CASES_PATH.name}[/dim]")
        else:
            console.print(f"[yellow]No cases.json found — run eval/generate_cases.py to create it.[/yellow]")
        all_samples = generated + all_samples

    sys.exit(run_eval(all_samples, stop_on_unexpected=args.strict))
