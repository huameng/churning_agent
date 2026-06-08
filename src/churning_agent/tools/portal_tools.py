"""
ADK tool functions for driving a whitelisted rewards portal.

Thin wrappers over the BrowserSession singleton. The domain guard lives in the
session itself; here we add stuck-detection (so the agent escalates instead of
looping) and a deterministic login using the site adapter's recipe.
"""

from .browser import ActionResult, get_session
from .credentials import get_credentials
from .escalation import StuckDetector
from . import offer_log
from . import valuation
from .offer_classifier import PortalOffer, classify_offer
from .offer_parsers import fetch_swagbucks_offers, parse_offers
from .sites import REGISTRY

# Full offer detail from the last list_offers call, keyed by (site, key), so
# assess_offer can classify using requirements + reward breakdown.
_offer_cache: dict[tuple[str, str], dict] = {}

# Per-site stuck detectors, so concurrent site lanes don't interfere.
_stuck: dict[str, StuckDetector] = {}

_STUCK_NOTE = (
    "\n\n[warning] the page has not changed across recent actions — you may be "
    "stuck (dead button, captcha, or wrong element). Consider ask_human or abort_workflow."
)


def _detector(site: str) -> StuckDetector:
    return _stuck.setdefault(site, StuckDetector())


def _render(site: str, result: ActionResult) -> str:
    if not result.success:
        return f"FAILED: {result.error}"
    obs = result.observation
    note = _STUCK_NOTE if _detector(site).record(obs.signature()) else ""
    return obs.summary() + note


# Every browser tool takes `site` — it both selects the browser tab (lane) for
# that site and lets independent site agents run concurrently. Always pass the
# site you are working on.

async def observe_page(site: str) -> str:
    """Observe the current page for `site`: URL, title, numbered interactive elements, text."""
    session = await get_session(site)
    obs = await session.observe()
    note = _STUCK_NOTE if _detector(site).record(obs.signature()) else ""
    return obs.summary() + note


async def go_to(site: str, url: str) -> str:
    """Navigate `site`'s tab to a URL. Refuses if the URL is not a whitelisted site."""
    session = await get_session(site)
    return _render(site, await session.navigate(url))


async def click_element(site: str, ref: int) -> str:
    """Click the interactive element with the given ref (from observe_page) on `site`'s tab."""
    session = await get_session(site)
    return _render(site, await session.click(ref))


async def fill_field(site: str, ref: int, text: str) -> str:
    """Type `text` into the field with the given ref on `site`'s tab."""
    session = await get_session(site)
    return _render(site, await session.fill(ref, text))


async def scroll_page(site: str, direction: str = "down") -> str:
    """Scroll `site`'s tab ('down' or 'up') to load lazily-rendered offers, then re-observe."""
    session = await get_session(site)
    return _render(site, await session.scroll(direction))


async def list_offers(site: str) -> str:
    """
    Navigate to a site's offers page and return the FULL offer list (all offers,
    not just the few visible on screen), valued in USD, sorted by value, and
    tagged NEW vs already-seen. Use this instead of observe_page to enumerate
    offers — especially on Swagbucks, where it reads the offers API for the
    complete list and gives each a stable key for note_offer.
    """
    adapter = REGISTRY.get(site)
    if adapter is None:
        return f"'{site}' is not a whitelisted site. Allowed: {', '.join(REGISTRY)}."
    session = await get_session(site)

    if site == "swagbucks":
        offers = await fetch_swagbucks_offers(session, adapter.offers_url)
    else:
        nav = await session.navigate(adapter.offers_url)
        if not nav.success:
            return f"Could not open offers page: {nav.error}"
        await session.page.wait_for_timeout(6000)
        for _ in range(4):
            await session.scroll("down")
            await session.page.wait_for_timeout(1500)
        offers = [{"key": o["title"], "title": o["title"], "reward": o["reward"],
                   "sb": None, "is_game": False} for o in await parse_offers(session, site)]

    if not offers:
        return ("No offers parsed (layout/API may have changed). "
                "Fall back to observe_page and read offers from the elements/text.")

    for o in offers:
        o["usd"] = valuation.to_usd(o["sb"], "SB") if o.get("sb") else None
        o["seen"] = offer_log.is_seen(site, o["key"])
        _offer_cache[(site, o["key"])] = o      # remember detail for assess_offer
    offers.sort(key=lambda o: (o.get("usd") or 0), reverse=True)

    new_ct = sum(1 for o in offers if not o["seen"])
    lines = [f"{len(offers)} offers on {site} — {new_ct} NEW since last run. "
             f"Pass the [key=...] to note_offer for any you report:"]
    for i, o in enumerate(offers):
        val = f" ~${o['usd']:.0f}" if o.get("usd") else ""
        kind = " [game]" if o.get("is_game") else ""
        tag = "" if o["seen"] else " (NEW)"
        lines.append(f"  {i}. {o['reward']}{val} — {o['title']}{kind}{tag} [key={o['key']}]")
    return "\n".join(lines)


def offer_details(site: str, offer_key: str) -> str:
    """Return the full detail (requirements + per-goal SB breakdown) for one
    offer from the last list_offers call, so you can see what it actually takes
    to earn the reward."""
    o = _offer_cache.get((site, offer_key))
    if o is None:
        return f"No cached offer {offer_key} for {site}. Call list_offers({site}) first."
    return (f"{o['title']} — {o['reward']}"
            + (" [game]" if o.get("is_game") else "")
            + f"\n{o.get('detail') or '(no further detail provided)'}")


def assess_offer(site: str, offer_key: str) -> dict:
    """
    Classify ONE offer using its full detail (requirements + per-goal SB
    breakdown), not just its title — so it accounts for deposits, spend, and
    grind. Use this for Swagbucks offers from list_offers.

    Returns label (ACCEPT/SKIP/UNCERTAIN), reasoning, estimated_value (realistic
    attainable USD), question (UNCERTAIN only), title, reward.
    """
    o = _offer_cache.get((site, offer_key))
    if o is None:
        return {"error": f"No cached offer {offer_key} for {site}. Call list_offers({site}) first."}
    offer = PortalOffer(merchant=o["title"], reward=o["reward"], description=o.get("detail", ""))
    result = classify_offer(offer)
    return {
        "title": o["title"],
        "reward": o["reward"],
        "label": result.label,
        "reasoning": result.reasoning,
        "question": result.question,
        "estimated_value": result.estimated_value,
    }


def note_offer(site: str, offer_key: str, merchant: str = "", reward: str = "",
               label: str = "", estimated_value: float | None = None) -> dict:
    """
    Record an offer you've evaluated so it isn't re-surfaced on future runs.

    Call this for every offer you classify. Use a STABLE offer_key: prefer the
    provider's offer id (e.g. the Swagbucks offerID in an /offer/click URL); if
    none is visible, use "merchant|reward".

    Returns {new: bool, times_seen: int, first_seen}. Only report MONEYMAKERS
    where new=true; for new=false, the user has already seen it — don't re-surface.
    """
    return offer_log.note_offer(site, offer_key, merchant, reward, label, estimated_value)


async def portal_login(site: str) -> str:
    """
    Log in to a whitelisted site using its stored credentials.

    Args:
        site: Site name, e.g. "topcashback".

    Returns:
        A status string. On anything unexpected (captcha, 2FA, changed layout,
        bad credentials) it says so — escalate to the human via ask_human.
    """
    adapter = REGISTRY.get(site)
    if adapter is None:
        return f"'{site}' is not a whitelisted site. Allowed: {', '.join(REGISTRY)}."

    _detector(site).reset()
    session = await get_session(site)

    # Already authenticated via the persistent profile? Skip the login form.
    await session.navigate(adapter.base_url)
    try:
        # The logout link may sit in a collapsed menu, so match attached (not visible).
        await session.page.wait_for_selector(adapter.logged_in_selector, state="attached", timeout=5000)
        return f"Already logged in to {site} (persistent session)."
    except Exception:
        pass

    creds = get_credentials(adapter.credential_key)
    if creds is None:
        return (f"Not logged in and no credentials for {site}. Either run login_once.py to log in "
                f"by hand, or set {adapter.credential_key.upper()}_EMAIL/_PASSWORD in .secrets.env.")

    nav = await session.navigate(adapter.login_url)
    if not nav.success:
        return f"Could not open the login page: {nav.error}"

    page = session.page
    try:
        await page.fill(adapter.username_selector, creds[0])
        await page.fill(adapter.password_selector, creds[1])
        await page.click(adapter.submit_selector)
        await page.wait_for_load_state("domcontentloaded")
    except Exception as e:
        return (f"The login form did not match the expected layout ({e}). The page may have "
                f"changed or is showing a captcha — escalate to the human.")

    try:
        await page.wait_for_selector(adapter.logged_in_selector, state="attached", timeout=8000)
        return f"Logged in to {site}."
    except Exception:
        return ("Login submitted but success could not be confirmed (no logged-in marker found). "
                "Likely a captcha, 2FA prompt, or wrong credentials — escalate to the human.")
