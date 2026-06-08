"""
Whitelist registry of sites the agent is allowed to drive.

Membership in REGISTRY is the permission: if a site isn't here, the browser
action primitives refuse to act on it (see browser.py domain guard). This is a
whitelist by design — we never add a site we don't want the agent touching
(e.g. banks), so there is no blacklist to maintain.
"""

from urllib.parse import urlparse

from pydantic import BaseModel


class SiteAdapter(BaseModel):
    name: str
    allowed_domains: list[str]          # bare hostnames, e.g. "www.topcashback.com"
    base_url: str
    login_url: str
    offers_url: str
    credential_key: str                 # key into .secrets.env (see credentials.py)
    # Deterministic login recipe (CSS selectors). The agent falls back to
    # generic observe/click reasoning when these don't match the live page.
    username_selector: str
    password_selector: str
    submit_selector: str
    logged_in_selector: str             # present only once logged in; used to verify


TOPCASHBACK = SiteAdapter(
    name="topcashback",
    allowed_domains=["www.topcashback.com", "topcashback.com"],
    base_url="https://www.topcashback.com",
    login_url="https://www.topcashback.com/logon/",
    # Homepage lists featured merchants + categories; rates live on each
    # merchant page (/{slug}/). There is no single activate-list page.
    offers_url="https://www.topcashback.com/",
    credential_key="topcashback",
    username_selector="#txtEmail",
    password_selector="#loginPasswordInput",
    submit_selector="#Loginbtn",
    logged_in_selector="a[href*='logout']",
)


# NOTE: selectors/URLs below are first guesses, to be corrected by a live
# discovery pass (discover.py). Swagbucks is busier than TopCashback with many
# offer types (Discover/paid offers, surveys, Shop cashback, daily goals).
SWAGBUCKS = SiteAdapter(
    name="swagbucks",
    allowed_domains=["www.swagbucks.com", "swagbucks.com"],
    base_url="https://www.swagbucks.com",
    login_url="https://www.swagbucks.com/p/login",
    # /discover-new/featured is the full featured paid-offer list (the
    # MONEYMAKERs). Other sections: /surveys, /games-new, /shop, /invite.
    offers_url="https://www.swagbucks.com/discover-new/featured",
    credential_key="swagbucks",
    username_selector="input[name='email']",
    password_selector="input[name='password']",
    submit_selector="button:has-text('Log In')",
    logged_in_selector="a[href*='/account']",
)


REGISTRY: dict[str, SiteAdapter] = {
    TOPCASHBACK.name: TOPCASHBACK,
    SWAGBUCKS.name: SWAGBUCKS,
}


def _hostname(url: str) -> str:
    return (urlparse(url).hostname or "").lower()


def is_allowed(url: str) -> bool:
    """True if url's host belongs to a whitelisted site."""
    host = _hostname(url)
    return any(host in a.allowed_domains for a in REGISTRY.values())


def adapter_for_url(url: str) -> SiteAdapter | None:
    """Return the adapter that owns this url's host, or None if not whitelisted."""
    host = _hostname(url)
    for adapter in REGISTRY.values():
        if host in adapter.allowed_domains:
            return adapter
    return None


def get_adapter(name: str) -> SiteAdapter:
    """Look up an adapter by site name. Raises KeyError if unknown."""
    return REGISTRY[name]
