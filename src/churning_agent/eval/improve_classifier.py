"""
Agentic prompt improvement loop for the DoC classifier.

DO NOT RUN without understanding what it does: this script modifies
churning_agent/tools/classifier.py in place. Commit or stash your
changes first so you can diff and revert.

Strategy
--------
Cases are split 80/20 (stratified by label) into a train set and a
holdout set. The loop optimises only against the train set and uses
the holdout as an independent guard against overfitting.

Each iteration:
  1. Run eval on the train split — get baseline macro-F1 and failures.
  2. Call an LLM with the current _SYSTEM_PROMPT and a failure summary
     (titles + model reasonings only — no case content, no holdout data).
  3. The LLM proposes one targeted change to the label definitions.
  4. Apply the change in-memory, re-run eval on both splits.
  5. Accept the change only if:
       - Train macro-F1 improved, AND
       - Holdout macro-F1 did not drop below MIN_HOLDOUT_F1
     Otherwise revert.
  6. Write the accepted prompt back to classifier.py.
  7. Append a record to eval/improvement_history.json.
  8. Stop if no failures remain, F1 plateaus, or max iterations reached.

Anti-overfit measures
---------------------
- The LLM proposing changes never sees the holdout set or case content.
- It sees only failure titles and the model's own reasonings.
- Its output must be a revised _SYSTEM_PROMPT (label definitions only,
  no specific case examples allowed).
- Improvement on BOTH splits is required to accept a change.

Run from churning_agent/:
    uv run python -m churning_agent.eval.improve_classifier
    uv run python -m churning_agent.eval.improve_classifier --iterations 3
"""

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from rich.console import Console

from churning_agent._paths import PROJECT_ROOT
load_dotenv(PROJECT_ROOT / ".env")

import churning_agent.tools.classifier as _clf_module
from churning_agent.eval.eval_classifier import evaluate_samples

console = Console()

_EVAL_DIR = PROJECT_ROOT / "eval"
_CASES_PATH = _EVAL_DIR / "cases.json"
_HISTORY_PATH = _EVAL_DIR / "improvement_history.json"
_CLASSIFIER_PATH = Path(__file__).parent.parent / "tools" / "classifier.py"

MAX_ITERATIONS = 5
TRAIN_FRAC = 0.8
SPLIT_SEED = 42
MIN_HOLDOUT_F1 = 0.82   # Reject any change that drops holdout macro-F1 below this
PLATEAU_THRESHOLD = 0.002  # Stop if improvement is smaller than this


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def macro_f1(metrics: dict) -> float:
    if not metrics:
        return 0.0
    return sum(m["f1"] for m in metrics.values()) / len(metrics)


def stratified_split(cases: list[dict], frac: float = TRAIN_FRAC, seed: int = SPLIT_SEED):
    """Split cases into (train, holdout) with equal label distribution."""
    import random
    by_label: dict[str, list] = defaultdict(list)
    for c in cases:
        by_label[c["expected"]].append(c)

    train, holdout = [], []
    rng = random.Random(seed)
    for label_cases in by_label.values():
        shuffled = label_cases.copy()
        rng.shuffle(shuffled)
        n_train = max(1, int(len(shuffled) * frac))
        train.extend(shuffled[:n_train])
        holdout.extend(shuffled[n_train:])
    return train, holdout


def _extract_system_prompt(source: str) -> str:
    """Extract the _SYSTEM_PROMPT string value from classifier.py source."""
    m = re.search(r'_SYSTEM_PROMPT\s*=\s*"""(.*?)"""', source, re.DOTALL)
    if not m:
        raise ValueError("Could not find _SYSTEM_PROMPT in classifier.py")
    return m.group(1)


def _write_system_prompt(new_prompt: str) -> None:
    """Replace _SYSTEM_PROMPT in classifier.py on disk and in the live module."""
    source = _CLASSIFIER_PATH.read_text(encoding="utf-8")
    new_source = re.sub(
        r'(_SYSTEM_PROMPT\s*=\s*""").*?(""")',
        lambda m: m.group(1) + new_prompt + m.group(2),
        source,
        flags=re.DOTALL,
    )
    _CLASSIFIER_PATH.write_text(new_source, encoding="utf-8")
    _clf_module._SYSTEM_PROMPT = new_prompt


def _propose_prompt_change(current_prompt: str, failures: list[dict]) -> str:
    """
    Ask Gemini to propose an improved _SYSTEM_PROMPT based on failure patterns.
    Only sees titles and model reasonings — never case content or holdout data.
    """
    def _failure_line(f: dict) -> str:
        line = (
            f"- Expected {f['expected']}, got {f['got']}: \"{f['title']}\"\n"
            f"  Model's reasoning: {f['model_reasoning']}"
        )
        if f.get("human_reasoning"):
            line += f"\n  Human explanation: {f['human_reasoning']}"
        return line

    failure_lines = "\n".join(_failure_line(f) for f in failures)

    prompt = f"""You are improving a classifier system prompt. The classifier assigns one of four
labels (IRRELEVANT, MONEYMAKER, WORTHLESS, UNCERTAIN) to Doctor of Credit blog posts.

Current system prompt:
---
{current_prompt}
---

The classifier made the following mistakes on the training set.
You are shown only the post title and the model's own reasoning — not the full content.

Failures:
{failure_lines}

Task: Propose ONE targeted improvement to the label definitions or classification criteria
that would fix the most failures above. Your change must:
- Modify the label definitions or decision criteria only
- NOT add specific post titles or content as examples
- NOT change the JSON output format description
- Be minimal — change as few words as possible while fixing the pattern

Return ONLY the complete updated system prompt text (everything between the triple quotes),
with no other explanation. Do not wrap it in markdown code fences."""

    client = genai.Client()
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=8000),
        ),
    )
    return response.text.strip()


def _load_history() -> list[dict]:
    if not _HISTORY_PATH.exists():
        return []
    return json.loads(_HISTORY_PATH.read_text(encoding="utf-8"))


def _save_history(history: list[dict]) -> None:
    _HISTORY_PATH.write_text(json.dumps(history, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def improve(max_iterations: int = MAX_ITERATIONS) -> None:
    cases = json.loads(_CASES_PATH.read_text(encoding="utf-8"))
    train_cases, holdout_cases = stratified_split(cases)
    console.print(
        f"Split: {len(train_cases)} train / {len(holdout_cases)} holdout "
        f"(stratified, seed={SPLIT_SEED})"
    )

    history = _load_history()
    source = _CLASSIFIER_PATH.read_text(encoding="utf-8")
    current_prompt = _extract_system_prompt(source)

    # Baseline on both splits
    console.print("\n[bold]Baseline[/bold]")
    train_metrics, _ = evaluate_samples(train_cases)
    holdout_metrics, _ = evaluate_samples(holdout_cases)
    baseline_train_f1 = macro_f1(train_metrics)
    baseline_holdout_f1 = macro_f1(holdout_metrics)
    console.print(f"  Train macro-F1:   {baseline_train_f1:.3f}")
    console.print(f"  Holdout macro-F1: {baseline_holdout_f1:.3f}")

    for iteration in range(1, max_iterations + 1):
        console.print(f"\n[bold]Iteration {iteration}[/bold]")

        train_metrics, train_failures = evaluate_samples(train_cases)
        current_train_f1 = macro_f1(train_metrics)

        if not train_failures:
            console.print("  No failures on train set — stopping.")
            break

        console.print(f"  Train macro-F1: {current_train_f1:.3f} | {len(train_failures)} failure(s)")
        for f in train_failures:
            console.print(f"  [dim]  {f['expected']} -> {f['got']}: {f['title'][:60]}[/dim]")

        console.print("  Proposing prompt change...")
        try:
            new_prompt = _propose_prompt_change(current_prompt, train_failures)
        except Exception as e:
            console.print(f"  [red]LLM call failed: {e}[/red]")
            break

        # Test new prompt in-memory
        _clf_module._SYSTEM_PROMPT = new_prompt
        new_train_metrics, _ = evaluate_samples(train_cases)
        new_holdout_metrics, _ = evaluate_samples(holdout_cases)
        new_train_f1 = macro_f1(new_train_metrics)
        new_holdout_f1 = macro_f1(new_holdout_metrics)

        improvement = new_train_f1 - current_train_f1
        holdout_ok = new_holdout_f1 >= MIN_HOLDOUT_F1

        record = {
            "iteration": iteration,
            "timestamp": datetime.utcnow().isoformat(),
            "train_f1_before": round(current_train_f1, 4),
            "train_f1_after": round(new_train_f1, 4),
            "holdout_f1_after": round(new_holdout_f1, 4),
            "accepted": False,
            "reason": "",
            "prompt_before": current_prompt,
            "prompt_after": new_prompt,
        }

        if improvement > PLATEAU_THRESHOLD and holdout_ok:
            record["accepted"] = True
            record["reason"] = f"Train +{improvement:.3f}, holdout {new_holdout_f1:.3f} >= floor"
            console.print(
                f"  [green]ACCEPTED[/green] Train F1: {current_train_f1:.3f} -> {new_train_f1:.3f} "
                f"| Holdout: {new_holdout_f1:.3f}"
            )
            _write_system_prompt(new_prompt)
            current_prompt = new_prompt

        elif not holdout_ok:
            record["reason"] = f"Holdout {new_holdout_f1:.3f} < floor {MIN_HOLDOUT_F1}"
            console.print(
                f"  [red]REJECTED (overfit risk)[/red] Holdout F1 {new_holdout_f1:.3f} < {MIN_HOLDOUT_F1}"
            )
            _clf_module._SYSTEM_PROMPT = current_prompt  # revert in memory

        elif improvement <= PLATEAU_THRESHOLD:
            record["reason"] = f"Plateau: improvement {improvement:.4f} <= threshold {PLATEAU_THRESHOLD}"
            console.print(f"  [yellow]REJECTED (plateau)[/yellow] Improvement {improvement:.4f} too small")
            _clf_module._SYSTEM_PROMPT = current_prompt  # revert in memory

        history.append(record)
        _save_history(history)

        if improvement <= PLATEAU_THRESHOLD:
            console.print("  Plateaued — stopping early.")
            break

    # Final report
    console.print("\n[bold]Final[/bold]")
    final_train_metrics, _ = evaluate_samples(train_cases)
    final_holdout_metrics, _ = evaluate_samples(holdout_cases)
    final_train_f1 = macro_f1(final_train_metrics)
    final_holdout_f1 = macro_f1(final_holdout_metrics)
    console.print(f"  Train macro-F1:   {baseline_train_f1:.3f} -> {final_train_f1:.3f}")
    console.print(f"  Holdout macro-F1: {baseline_holdout_f1:.3f} -> {final_holdout_f1:.3f}")
    console.print(f"  History saved to {_HISTORY_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agentic classifier prompt improvement loop")
    parser.add_argument("--iterations", type=int, default=MAX_ITERATIONS)
    args = parser.parse_args()
    improve(max_iterations=args.iterations)
