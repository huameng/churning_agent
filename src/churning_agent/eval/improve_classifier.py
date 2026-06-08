"""
Shared helpers for classifier evaluation and self-improvement.

The improvement loop itself lives in churning_agent.improvement_agent (an
LLM-driven ADK agent the user talks to). This module holds only the pure pieces
that the agent and the eval CLIs both reuse: the train/holdout split, the
macro-F1 metric, and the improvement-history log. The classifier prompt is read
and written through churning_agent.prompts (config/prompts/post_classifier.yaml)
— nothing here edits source code.
"""

import json
import random
from collections import defaultdict

from churning_agent._paths import PROJECT_ROOT

_EVAL_DIR = PROJECT_ROOT / "eval"
_CASES_PATH = _EVAL_DIR / "cases.json"
_HISTORY_PATH = _EVAL_DIR / "improvement_history.json"

TRAIN_FRAC = 0.8
SPLIT_SEED = 42
MIN_HOLDOUT_F1 = 0.82   # Reject any change that drops holdout macro-F1 below this


def macro_f1(metrics: dict) -> float:
    if not metrics:
        return 0.0
    return sum(m["f1"] for m in metrics.values()) / len(metrics)


def stratified_split(cases: list[dict], frac: float = TRAIN_FRAC, seed: int = SPLIT_SEED):
    """Split cases into (train, holdout) with equal label distribution."""
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


def _load_history() -> list[dict]:
    if not _HISTORY_PATH.exists():
        return []
    return json.loads(_HISTORY_PATH.read_text(encoding="utf-8"))


def _save_history(history: list[dict]) -> None:
    _HISTORY_PATH.write_text(json.dumps(history, indent=2), encoding="utf-8")
