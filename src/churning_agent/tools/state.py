import json
from datetime import datetime
from pathlib import Path

from churning_agent._paths import DATA_DIR

_STATE_FILE = DATA_DIR / "state.json"


def get_last_seen_url() -> str | None:
    """Return the URL of the last DoC post we processed, or None if never run."""
    if not _STATE_FILE.exists():
        return None
    data = json.loads(_STATE_FILE.read_text())
    return data.get("last_seen_url")


def set_last_seen_url(url: str) -> str:
    """Record the URL of the most recent DoC post we processed."""
    _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _STATE_FILE.write_text(json.dumps({
        "last_seen_url": url,
        "updated_at": datetime.now().isoformat(),
    }, indent=2))
    return f"Saved last seen URL: {url}"
