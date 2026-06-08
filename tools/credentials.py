"""
Per-site credentials, loaded from a gitignored .secrets.env.

Keys are SITE_EMAIL / SITE_PASSWORD (site uppercased), e.g.
    TOPCASHBACK_EMAIL=me@example.com
    TOPCASHBACK_PASSWORD=hunter2

Only sites in the whitelist (sites.REGISTRY) ever get credentials — there is no
mechanism here to drive a site we haven't approved.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

_SECRETS_PATH = Path(__file__).parent.parent / ".secrets.env"
_loaded = False


def _ensure_loaded() -> None:
    global _loaded
    if not _loaded:
        load_dotenv(_SECRETS_PATH)
        _loaded = True


def get_credentials(credential_key: str) -> tuple[str, str] | None:
    """Return (email, password) for a site's credential_key, or None if unset."""
    _ensure_loaded()
    prefix = credential_key.upper()
    email = os.environ.get(f"{prefix}_EMAIL")
    password = os.environ.get(f"{prefix}_PASSWORD")
    if email and password:
        return email, password
    return None
