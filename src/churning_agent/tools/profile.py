import threading
from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel

from churning_agent._paths import CONFIG_DIR

_CONFIG_PATH = CONFIG_DIR / "user_profile.yaml"

# update_profile does a read-modify-write of the YAML, and concurrent site lanes
# (topcashback, swagbucks, ...) can each call it; serialize so edits don't clobber.
_write_lock = threading.Lock()


class ChurningCooldown(BaseModel):
    bank: str
    last_bonus_date: date


class Preferences(BaseModel):
    min_profit_threshold: float = 100.0
    max_direct_deposit_required: int = 2
    willing_to_open_brokerage: bool = True
    willing_to_do_credit_pull: bool = True
    willing_to_visit_branch: bool = True
    excluded_banks: list[str] = []
    additional_context: list[str] = []
    min_discount_pct: float = 50.0     # minimum % off to qualify as DISCOUNT_MONEYMAKER
    min_discount_savings: float = 25.0  # minimum $ savings to qualify as DISCOUNT_MONEYMAKER


class UserProfile(BaseModel):
    state: str
    zip_code: str
    existing_accounts: list[str] = []
    existing_credit_cards: list[str] = []
    churning_cooldowns: list[ChurningCooldown] = []
    preferences: Preferences = Preferences()

    def to_prompt_str(self) -> str:
        p = self.preferences
        lines = [
            f"Location: {self.state} (zip {self.zip_code})",
            f"Existing bank accounts: {', '.join(self.existing_accounts) or 'none'}",
            f"Credit cards held: {', '.join(self.existing_credit_cards) or 'none'}",
            f"Minimum profit threshold: ${p.min_profit_threshold:.0f}",
            f"Willing to open brokerage account: {p.willing_to_open_brokerage}",
            f"Willing to do hard credit pull: {p.willing_to_do_credit_pull}",
            f"Willing to visit a branch: {p.willing_to_visit_branch}",
        ]
        lines.append(
            f"Discount thresholds (DISCOUNT_MONEYMAKER): "
            f">={p.min_discount_pct:.0f}% off AND >=${p.min_discount_savings:.0f} savings"
        )
        if p.excluded_banks:
            lines.append(f"Banks to skip entirely: {', '.join(p.excluded_banks)}")
        if self.churning_cooldowns:
            cooldowns = [f"{c.bank} (last bonus: {c.last_bonus_date})" for c in self.churning_cooldowns]
            lines.append(f"Recent churning history (may be in cooldown): {', '.join(cooldowns)}")
        if p.additional_context:
            lines.append("Additional context:")
            for note in p.additional_context:
                lines.append(f"  - {note}")
        return "\n".join(lines)


# Cache keyed by config path, so loading a different profile (e.g. a test
# fixture) doesn't return the default that was cached first.
_profiles: dict[str, UserProfile] = {}


def _load_from(path: Path) -> UserProfile:
    with open(path) as f:
        data = yaml.safe_load(f)

    personal = data.get("personal", {})
    banking = data.get("banking", {})

    return UserProfile(
        state=personal["state"],
        zip_code=str(personal["zip_code"]),
        existing_accounts=[a.lower() for a in banking.get("existing_accounts", [])],
        existing_credit_cards=[c.lower() for c in banking.get("existing_credit_cards", [])],
        churning_cooldowns=[
            ChurningCooldown(bank=c["bank"].lower(), last_bonus_date=c["last_bonus_date"])
            for c in banking.get("churning_cooldowns", [])
        ],
        preferences=Preferences(**(banking.get("preferences", {}))),
    )


def load_profile(config_path: Path | None = None) -> UserProfile:
    path = config_path or _CONFIG_PATH
    key = str(path)
    if key not in _profiles:
        _profiles[key] = _load_from(path)
    return _profiles[key]


def _reset_cache() -> None:
    _profiles.clear()


def update_profile(field: str, value: str) -> str:
    """
    Update the user profile with information learned from human feedback, then
    reset the classifier cache so the next classification picks up the change.

    Supported fields:
      existing_accounts      — bank name to add (e.g. "chase checking")
      existing_credit_cards  — card name to add (e.g. "citi double cash")
      excluded_banks         — bank name to skip entirely (e.g. "bank of america")
      additional_context     — one factual note to append (last resort; prefer specific fields)
      willing_to_visit_branch   — "true" or "false"
      willing_to_open_brokerage — "true" or "false"
      willing_to_do_credit_pull — "true" or "false"
      min_profit_threshold   — dollar amount as a number (e.g. "50")
      min_discount_pct       — minimum % off for DISCOUNT_MONEYMAKER (e.g. "50")
      min_discount_savings   — minimum $ savings for DISCOUNT_MONEYMAKER (e.g. "25")

    Returns a confirmation string describing what changed.
    """
    path = _CONFIG_PATH
    value = value.strip()

    # Serialize the whole read-modify-write: concurrent lanes editing the same
    # YAML would otherwise lose each other's edits (last writer wins).
    with _write_lock:
        with open(path) as f:
            data = yaml.safe_load(f)

        banking = data.setdefault("banking", {})
        prefs = banking.setdefault("preferences", {})

        LIST_FIELDS = {
            "existing_accounts": banking,
            "existing_credit_cards": banking,
            "excluded_banks": prefs,
            "additional_context": prefs,
        }
        BOOL_FIELDS = {
            "willing_to_visit_branch": prefs,
            "willing_to_open_brokerage": prefs,
            "willing_to_do_credit_pull": prefs,
        }

        if field in LIST_FIELDS:
            container = LIST_FIELDS[field]
            current = container.get(field, []) or []
            normalized = value.lower() if field != "additional_context" else value
            if normalized not in [x.lower() if field != "additional_context" else x for x in current]:
                current.append(normalized if field != "additional_context" else value)
                container[field] = current
                msg = f"Added '{value}' to {field}."
            else:
                msg = f"'{value}' already present in {field}, no change."
        elif field in BOOL_FIELDS:
            if value.lower() not in ("true", "false"):
                return f"Error: value for {field} must be 'true' or 'false', got '{value}'."
            BOOL_FIELDS[field][field] = value.lower() == "true"
            msg = f"Set {field} to {value.lower() == 'true'}."
        elif field in ("min_profit_threshold", "min_discount_pct", "min_discount_savings"):
            try:
                prefs[field] = float(value)
                msg = f"Set {field} to {float(value)}."
            except ValueError:
                return f"Error: {field} must be a number, got '{value}'."
        else:
            return (
                f"Error: unknown field '{field}'. Supported: "
                "existing_accounts, existing_credit_cards, excluded_banks, additional_context, "
                "willing_to_visit_branch, willing_to_open_brokerage, willing_to_do_credit_pull, "
                "min_profit_threshold, min_discount_pct, min_discount_savings."
            )

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

        _reset_cache()
    return msg
