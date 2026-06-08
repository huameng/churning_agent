"""
Self-improvement subagent for the classifier.

Exposed as AgentTool to the root agent so the user can say
"run evals and improve yourself".
"""

import json
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.adk.agents import LlmAgent
from google.genai import types

import churning_agent.tools.classifier as _clf_module
from churning_agent.eval.eval_classifier import evaluate_samples
from churning_agent.eval.improve_classifier import (
    MIN_HOLDOUT_F1,
    SPLIT_SEED,
    TRAIN_FRAC,
    _CASES_PATH,
    _HISTORY_PATH,
    _write_system_prompt,
    macro_f1,
    stratified_split,
    _load_history,
    _save_history,
)

load_dotenv(Path(__file__).parent / ".env")

_MODEL = "gemini-2.5-flash"


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def run_eval() -> dict:
    """
    Run the classifier against all eval cases.
    Returns macro F1, per-label F1, total cases, failure count, and a list of failures
    (title, expected, got, model_reasoning, human_reasoning).
    """
    cases = json.loads(_CASES_PATH.read_text(encoding="utf-8"))
    metrics, failures = evaluate_samples(cases)
    return {
        "macro_f1": round(macro_f1(metrics), 4),
        "total_cases": len(cases),
        "failure_count": len(failures),
        "per_label_f1": {label: round(m["f1"], 4) for label, m in metrics.items()},
        "failures": [
            {
                "title": f["title"],
                "expected": f["expected"],
                "got": f["got"],
                "model_reasoning": f["model_reasoning"],
                "human_reasoning": f.get("human_reasoning"),
            }
            for f in failures
        ],
    }


def run_eval_split(split: str) -> dict:
    """
    Run eval on a named split of the eval cases.

    Args:
        split: "train" or "holdout" (80/20 stratified split, seed=42)

    Returns macro F1, per-label F1, case count, and failures for that split.
    """
    all_cases = json.loads(_CASES_PATH.read_text(encoding="utf-8"))
    train, holdout = stratified_split(all_cases, frac=TRAIN_FRAC, seed=SPLIT_SEED)
    cases = train if split == "train" else holdout

    metrics, failures = evaluate_samples(cases)
    return {
        "split": split,
        "n_cases": len(cases),
        "macro_f1": round(macro_f1(metrics), 4),
        "per_label_f1": {label: round(m["f1"], 4) for label, m in metrics.items()},
        "failure_count": len(failures),
        "failures": [
            {
                "title": f["title"],
                "expected": f["expected"],
                "got": f["got"],
                "model_reasoning": f["model_reasoning"],
                "human_reasoning": f.get("human_reasoning"),
            }
            for f in failures
        ],
    }


def get_classifier_prompt() -> str:
    """Return the current classifier system prompt."""
    return _clf_module._SYSTEM_PROMPT


def set_classifier_prompt(new_prompt: str) -> str:
    """
    Replace the classifier system prompt in memory only — does NOT write to disk.
    Use this to test a proposed change before committing.

    Args:
        new_prompt: The full replacement prompt text.
    """
    _clf_module._SYSTEM_PROMPT = new_prompt
    return "Prompt updated in memory. Run eval to test it, then call save_classifier_prompt to persist."


def save_classifier_prompt() -> str:
    """
    Persist the current in-memory classifier prompt to classifier.py on disk.
    Call this only after verifying the new prompt improves eval metrics.
    """
    _write_system_prompt(_clf_module._SYSTEM_PROMPT)
    return "Prompt saved to classifier.py."


def propose_prompt_improvement(failures: list) -> dict:
    """
    Ask Gemini to propose a targeted improvement to the classifier prompt based on failures.
    Only sees titles and model/human reasonings — never the full post content.

    Args:
        failures: List of failure dicts with keys:
            title, expected, got, model_reasoning, human_reasoning (may be null).
            Pass the failures list directly from run_eval_split.

    Returns a dict with:
        - new_prompt: the full replacement prompt text
        - what_changed: a short description of the specific change made
        - why: explanation of the failure pattern that motivated the change
    """

    def _fmt(f: dict) -> str:
        line = (
            f"- Expected {f['expected']}, got {f['got']}: \"{f['title']}\"\n"
            f"  Model's reasoning: {f['model_reasoning']}"
        )
        if f.get("human_reasoning"):
            line += f"\n  Human explanation: {f['human_reasoning']}"
        return line

    failure_lines = "\n".join(_fmt(f) for f in failures)
    current_prompt = _clf_module._SYSTEM_PROMPT

    meta_prompt = f"""You are improving a classifier system prompt. The classifier assigns one of five
labels (IRRELEVANT, MONEYMAKER, DISCOUNT_MONEYMAKER, WORTHLESS, UNCERTAIN) to Doctor of Credit blog posts.

Current system prompt:
---
{current_prompt}
---

The classifier made the following mistakes on the training set.
You are shown only the post title and the model's own reasoning — not the full content.
Human explanations are provided where available and are high-signal.

Failures:
{failure_lines}

Task: Propose ONE targeted improvement to the label definitions or classification criteria
that would fix the most failures above. Your change must:
- Modify the label definitions or decision criteria only
- NOT add specific post titles or content as examples
- NOT change the JSON output format description
- Be minimal — change as few words as possible while fixing the pattern

Respond with a JSON object with three fields:
- "new_prompt": the complete updated system prompt text (everything that would go between the triple quotes)
- "what_changed": one sentence describing exactly what you changed (e.g. "Added 'card-not-held offers' to IRRELEVANT examples")
- "why": one or two sentences explaining the failure pattern you observed and why this change addresses it"""

    client = genai.Client()
    response = client.models.generate_content(
        model=_MODEL,
        contents=meta_prompt,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=8000),
            response_mime_type="application/json",
        ),
    )
    result = json.loads(response.text)
    return {
        "new_prompt": result["new_prompt"],
        "what_changed": result.get("what_changed", ""),
        "why": result.get("why", ""),
    }


def record_improvement_attempt(
    iteration: int,
    train_f1_before: float,
    train_f1_after: float,
    holdout_f1_after: float,
    accepted: bool,
    reason: str,
    prompt_before: str,
    prompt_after: str,
) -> str:
    """
    Append a record of an improvement attempt to improvement_history.json.
    Call this after each accept or reject decision.

    Args:
        iteration: Which iteration number this is (1-based)
        train_f1_before: Train macro-F1 before the change
        train_f1_after: Train macro-F1 after the change
        holdout_f1_after: Holdout macro-F1 after the change
        accepted: Whether the change was accepted
        reason: Short explanation of the accept/reject decision
        prompt_before: The prompt text before the change
        prompt_after: The proposed prompt text
    """
    from datetime import datetime
    history = _load_history()
    history.append({
        "iteration": iteration,
        "timestamp": datetime.utcnow().isoformat(),
        "train_f1_before": round(train_f1_before, 4),
        "train_f1_after": round(train_f1_after, 4),
        "holdout_f1_after": round(holdout_f1_after, 4),
        "accepted": accepted,
        "reason": reason,
        "prompt_before": prompt_before,
        "prompt_after": prompt_after,
    })
    _save_history(history)
    return f"Recorded iteration {iteration} ({'accepted' if accepted else 'rejected'}) to improvement_history.json."


def get_improvement_history() -> dict:
    """
    Return the full improvement history — all past attempts, accepted or rejected.
    Use this to answer questions like 'did the improvement agent run?' or
    'what changed last time?'

    Returns a dict with:
        - total_attempts: how many iterations have been recorded
        - accepted: how many were accepted
        - entries: list of all records (newest first), each with
            iteration, timestamp, train_f1_before, train_f1_after,
            holdout_f1_after, accepted, reason
          (prompt text is omitted to keep the response concise)
    """
    history = _load_history()
    entries = [
        {
            "iteration": h["iteration"],
            "timestamp": h["timestamp"],
            "train_f1_before": h["train_f1_before"],
            "train_f1_after": h["train_f1_after"],
            "holdout_f1_after": h["holdout_f1_after"],
            "accepted": h["accepted"],
            "reason": h["reason"],
        }
        for h in reversed(history)
    ]
    return {
        "total_attempts": len(history),
        "accepted": sum(1 for h in history if h["accepted"]),
        "history_path": str(_HISTORY_PATH),
        "entries": entries,
    }


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

improvement_agent = LlmAgent(
    model=_MODEL,
    name="improvement_agent",
    description="Runs classifier evals and iteratively improves the classifier system prompt.",
    instruction=f"""You improve the DoC classifier by running evals and proposing targeted prompt changes.

When asked about past runs or improvement history, call get_improvement_history() and summarise the results — how many attempts, how many accepted, and what the F1 trajectory looked like.

When asked to run evals and improve (up to 5 iterations), follow this loop and narrate every step clearly to the user:

**Step 1 — Baseline**
Call run_eval_split("train") and run_eval_split("holdout") in parallel.
Then tell the user:
  "📊 Baseline — Train macro-F1: X.XXX (N failures) | Holdout macro-F1: X.XXX"
  List each failure as: "  • [Expected → Got] Title"
Save the original prompt via get_classifier_prompt().

**Step 2 — Check for failures**
If train failures = 0, tell the user the classifier is already perfect on the training set and stop.

**Step 3 — Propose a change**
Call propose_prompt_improvement with the full failures list from the train split.
Then tell the user:
  "💡 Proposed change: <what_changed from the result>"
  "   Why: <why from the result>"
  Show a brief before/after of the specific lines that changed in the prompt.

**Step 4 — Test the change**
Call set_classifier_prompt with new_prompt from the proposal result.
Call run_eval_split("train") and run_eval_split("holdout") in parallel.
Then tell the user:
  "🧪 Testing — Train macro-F1: X.XXX → X.XXX (+/-0.XXX) | Holdout: X.XXX"

**Step 5 — Accept or reject**
ACCEPT if: train macro-F1 improved by > 0.002 AND holdout macro-F1 >= {MIN_HOLDOUT_F1}
REJECT otherwise.

On ACCEPT:
  Call save_classifier_prompt().
  Call record_improvement_attempt with accepted=True.
  Tell the user: "✅ Accepted — prompt saved."

On REJECT (overfit):
  Call set_classifier_prompt(original_prompt) to revert.
  Call record_improvement_attempt with accepted=False.
  Tell the user: "❌ Rejected — holdout F1 dropped to X.XXX (floor is {MIN_HOLDOUT_F1}), reverting."

On REJECT (plateau):
  Call set_classifier_prompt(original_prompt) to revert.
  Call record_improvement_attempt with accepted=False.
  Tell the user: "⏹ Stopped — improvement too small (+X.XXX), plateau reached."

**Step 6 — Loop or stop**
- If accepted: loop back to Step 1 for the next iteration (update "original prompt" to the saved one).
- If rejected for any reason: stop.
- Stop after 5 accepted iterations regardless.

**Final report** (always give this at the end)
  "📈 Done — X iteration(s) accepted."
  "   Train:   X.XXX → X.XXX"
  "   Holdout: X.XXX → X.XXX"
  Bullet list of every accepted change with its what_changed description.
""",
    tools=[
        run_eval,
        run_eval_split,
        get_classifier_prompt,
        set_classifier_prompt,
        get_improvement_history,
        save_classifier_prompt,
        propose_prompt_improvement,
        record_improvement_attempt,
    ],
)
