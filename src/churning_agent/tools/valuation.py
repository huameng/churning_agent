"""
Convert reward units (points, foreign currency) to USD so offers with
non-cash rewards can be valued and compared on one scale.

Rates live in config/valuation.yaml — edit them as your own valuation changes.
"""

from pathlib import Path

import yaml

from churning_agent._paths import CONFIG_DIR

_CONFIG_PATH = CONFIG_DIR / "valuation.yaml"
_units: dict[str, float] | None = None


def _load() -> dict[str, float]:
    global _units
    if _units is None:
        data = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))
        _units = {k.upper(): float(v) for k, v in data["units"].items()}
    return _units


def to_usd(amount: float, unit: str) -> float | None:
    """Convert `amount` of `unit` to USD. Returns None for unknown units."""
    rate = _load().get(unit.upper())
    return None if rate is None else round(amount * rate, 2)


def to_prompt_str() -> str:
    """Conversion table for inclusion in a classifier prompt."""
    rows = [f"  1 {unit} = ${rate:g}" for unit, rate in _load().items()]
    return "Reward unit conversions (to USD):\n" + "\n".join(rows)
