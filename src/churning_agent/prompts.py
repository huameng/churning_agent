"""
Prompt + model registry.

Every LLM-driven component (the classifiers, the agents, the prompt-improver)
keeps its model id and system prompt together in one YAML file under
config/prompts/<name>.yaml:

    model: gemini-3.1-flash-lite
    system: |
      <the system prompt / agent instruction>
    thinking_budget: 8000   # optional, for the improver's meta-call

Edit the YAML to change a prompt or swap a model — no code change needed. The
self-improvement loop edits prompts the same way (save_system), so it no longer
rewrites Python source.
"""
from pathlib import Path

import yaml
from pydantic import BaseModel

from churning_agent._paths import CONFIG_DIR

_DIR = CONFIG_DIR / "prompts"


class Prompt(BaseModel):
    name: str
    model: str | None = None    # None for prompt fragments composed into another prompt
    system: str
    thinking_budget: int | None = None


_cache: dict[str, Prompt] = {}


def load(name: str) -> Prompt:
    """Load (and cache) a prompt. In-memory edits via set_system persist for the
    life of the process so callers always read the current text."""
    if name not in _cache:
        data = yaml.safe_load((_DIR / f"{name}.yaml").read_text(encoding="utf-8"))
        _cache[name] = Prompt(name=name, **data)
    return _cache[name]


def set_system(name: str, system: str) -> None:
    """Override a prompt's system text in memory only — used to test a candidate
    prompt before deciding whether to persist it."""
    load(name).system = system


def save_system(name: str, system: str) -> None:
    """Persist a new system text to the prompt's YAML and update the in-memory copy."""
    system = "\n".join(line.rstrip() for line in system.splitlines())
    path = _DIR / f"{name}.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["system"] = system
    path.write_text(_dump(data), encoding="utf-8")
    load(name).system = system


def reset_cache() -> None:
    _cache.clear()


# Emit multi-line strings as literal blocks so prompt YAML stays human-editable
# after a machine rewrite (rather than one giant quoted line).
class _PromptDumper(yaml.SafeDumper):
    pass


def _str_representer(dumper, data):
    style = "|" if "\n" in data else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)


_PromptDumper.add_representer(str, _str_representer)


def _dump(data: dict) -> str:
    return yaml.dump(data, Dumper=_PromptDumper, sort_keys=False, allow_unicode=True, width=100)
