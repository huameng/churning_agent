"""
Self-improvement subagent for the classifier.

Exposed as AgentTool to the root agent so the user can say
"run evals and improve yourself".
"""

import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.adk.agents import LlmAgent
from google.genai import types

from churning_agent import prompts
from churning_agent.llm import retry_transient, retrying_model
from churning_agent.eval.eval_classifier import evaluate_samples, _RESULTS_DIR
from churning_agent.eval.improve_classifier import (
    MIN_HOLDOUT_F1,
    SPLIT_SEED,
    TRAIN_FRAC,
    _CASES_PATH,
    _HISTORY_PATH,
    macro_f1,
    stratified_split,
    _load_history,
    _save_history,
)

from churning_agent._paths import PROJECT_ROOT
load_dotenv(PROJECT_ROOT / ".env")

logger = logging.getLogger(__name__)

# The classifier whose prompt this agent tunes (config/prompts/post_classifier.yaml).
_CLASSIFIER_PROMPT = "post_classifier"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _write_split_result(split: str, n_cases: int, metrics: dict, failures: list) -> Path:
    """Write a timestamped result file to eval/results/ so the user has a record."""
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_at = datetime.utcnow().isoformat()
    filename = run_at.replace(":", "").replace("-", "").replace("T", "_")[:15] + f"_{split}.json"
    path = _RESULTS_DIR / filename
    path.write_text(
        json.dumps({
            "run_at": run_at,
            "split": split,
            "n_cases": n_cases,
            "macro_f1": round(macro_f1(metrics), 4),
            "per_label_f1": {label: round(m["f1"], 4) for label, m in metrics.items()},
            "failure_count": len(failures),
            "failures": failures,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def _enrich_failures(failures: list, cases: list) -> list:
    """Attach a content snippet to each failure so the proposer sees the offer text."""
    content_by_url = {c["url"]: c.get("content", "") for c in cases}
    return [
        {
            "title": f["title"],
            "url": f.get("url"),
            "expected": f["expected"],
            "got": f["got"],
            "model_reasoning": f["model_reasoning"],
            "human_reasoning": f.get("human_reasoning"),
            "content_snippet": content_by_url.get(f.get("url"), "")[:300],
        }
        for f in failures
    ]


def _eval(cases: list, split: str) -> dict:
    """Evaluate `cases`, write a timestamped result file, and return the summary."""
    logger.info("improver: evaluating %d case(s) [%s]", len(cases), split)
    metrics, failures = evaluate_samples(cases)
    enriched = _enrich_failures(failures, cases)
    logger.info("improver: [%s] macro-F1=%.3f, %d failure(s)",
                split, macro_f1(metrics), len(failures))
    path = _write_split_result(split, len(cases), metrics, enriched)
    return {
        "split": split,
        "n_cases": len(cases),
        "macro_f1": round(macro_f1(metrics), 4),
        "per_label_f1": {label: round(m["f1"], 4) for label, m in metrics.items()},
        "failure_count": len(failures),
        "result_file": str(path),
        "failures": enriched,
    }


def run_eval() -> dict:
    """
    Run the classifier against ALL eval cases (no split).
    Writes a timestamped result file to eval/results/. Returns macro F1,
    per-label F1, case count, result file path, and failures.
    """
    all_cases = json.loads(_CASES_PATH.read_text(encoding="utf-8"))
    return _eval(all_cases, "all")


def run_eval_split(split: str) -> dict:
    """
    Run eval on a named split of the eval cases.
    Writes a timestamped result file to eval/results/.

    Args:
        split: "train" or "holdout" (80/20 stratified split, seed=42)

    Returns macro F1, per-label F1, case count, result file path, and failures.
    Each failure includes a content_snippet (first 300 chars of the offer section).
    """
    all_cases = json.loads(_CASES_PATH.read_text(encoding="utf-8"))
    train, holdout = stratified_split(all_cases, frac=TRAIN_FRAC, seed=SPLIT_SEED)
    return _eval(train if split == "train" else holdout, split)


def get_classifier_prompt() -> str:
    """Return the current classifier system prompt."""
    return prompts.load(_CLASSIFIER_PROMPT).system


def set_classifier_prompt(new_prompt: str) -> str:
    """
    Replace the classifier system prompt in memory only — does NOT write to disk.
    Use this to test a proposed change before committing.

    Args:
        new_prompt: The full replacement prompt text.
    """
    prompts.set_system(_CLASSIFIER_PROMPT, new_prompt)
    return "Prompt updated in memory. Run eval to test it, then call save_classifier_prompt to persist."


def save_classifier_prompt() -> str:
    """
    Persist the current in-memory classifier prompt to its YAML on disk.
    Call this only after verifying the new prompt improves eval metrics.
    """
    prompts.save_system(_CLASSIFIER_PROMPT, prompts.load(_CLASSIFIER_PROMPT).system)
    return "Prompt saved to config/prompts/post_classifier.yaml."


@retry_transient
def _propose(meta_prompt: str, model: str, thinking_budget: int | None) -> str:
    """The improver's model call, retried through transient transport/server errors."""
    response = genai.Client().models.generate_content(
        model=model,
        contents=meta_prompt,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
            response_mime_type="application/json",
        ),
    )
    return response.text


def propose_prompt_improvement(failures: list) -> dict:
    """
    Ask Gemini to propose an improvement to the classifier prompt based on failures.
    Receives titles, model/human reasonings, AND a content snippet for each failure.

    Args:
        failures: List of failure dicts from run_eval_split, including content_snippet.

    Returns a dict with:
        - new_prompt: the full replacement prompt text
        - what_changed: a short description of the specific change made
        - why: explanation of the failure pattern that motivated the change
    """
    # Group failures by (expected → got) to surface patterns
    by_type: dict[str, list] = defaultdict(list)
    for f in failures:
        key = f"{f['expected']} → {f['got']}"
        by_type[key].append(f)

    unannotated = [f for f in failures if not f.get("human_reasoning")]

    sections = []
    for key, cases in sorted(by_type.items(), key=lambda x: -len(x[1])):
        sections.append(f"\n### {key} ({len(cases)} failure(s)):")
        for f in cases:
            entry = f'- Title: "{f["title"]}"'
            if f.get("content_snippet"):
                entry += f'\n  Offer text: {f["content_snippet"]}'
            entry += f'\n  Model reasoning: {f["model_reasoning"]}'
            if f.get("human_reasoning"):
                entry += f'\n  Human note: {f["human_reasoning"]}'
            sections.append(entry)

    failure_block = "\n".join(sections)
    current_prompt = prompts.load(_CLASSIFIER_PROMPT).system

    annotation_note = ""
    if unannotated:
        annotation_note = (
            f"\nNote: {len(unannotated)} of {len(failures)} failures have no human annotation. "
            "Infer the correct rule from the offer text and model reasoning.\n"
        )

    cfg = prompts.load("prompt_improver")
    meta_prompt = f"""You are improving a classifier system prompt. The classifier assigns one of five
labels (IRRELEVANT, MONEYMAKER, DISCOUNT_MONEYMAKER, WORTHLESS, UNCERTAIN) to Doctor of Credit blog posts.

Current system prompt:
---
{current_prompt}
---

The classifier made the following mistakes on the training set, grouped by mistake type.
Each entry includes the post title, a snippet of the actual offer text, the model's reasoning,
and a human annotation where available (human annotations are high-signal — prioritise them).
{annotation_note}
Failures:
{failure_block}

{cfg.system}"""

    result = json.loads(_propose(meta_prompt, cfg.model, cfg.thinking_budget))
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
    Call this after every accept or reject decision — never skip this.

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
    Use this to answer 'did the improvement agent run?' or 'what changed last time?'
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

_AGENT = prompts.load("improvement_agent")
improvement_agent = LlmAgent(
    model=retrying_model(_AGENT.model),
    name="improvement_agent",
    description="Runs classifier evals and iteratively improves the classifier system prompt.",
    instruction=_AGENT.system.replace("{min_holdout_f1}", str(MIN_HOLDOUT_F1)),
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
