"""
Portal agent: drives a whitelisted rewards portal (currently TopCashback) to
find and activate worthwhile cashback offers.

Used two ways:
  - As a sub-agent of the churning_agent root agent (see agent.py), so the DoC
    monitor can hand off when a post points at a portal offer.
  - Standalone:  uv run python portal.py
"""

import asyncio

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from churning_agent.tools.browser import close_session
from churning_agent.tools.escalation import abort_workflow, ask_human
from churning_agent.tools.offer_classifier import classify_portal_offer
from churning_agent.tools.portal_tools import (
    assess_offer, click_element, fill_field, go_to, list_offers, note_offer,
    observe_page, offer_details, portal_login, scroll_page,
)
from churning_agent.tools.profile import update_profile

_INSTRUCTION = """You drive whitelisted rewards portals (TopCashback and Swagbucks) to find MONEYMAKERS — offers worth the user's time — and report them. The browser tools refuse any non-whitelisted site. Never spend money, complete a purchase, or click through an offer to a third-party store/app — your job ends at surfacing worthwhile offers, like a research assistant. An ACCEPT from the classifier is a MONEYMAKER.

All browser tools take the site name as their first argument — observe_page(site), go_to(site, url), scroll_page(site), click_element(site, ref), fill_field(site, ref, text) — which routes to that site's own browser tab. Always pass the site you are working on.

Login: call portal_login(site) first. If it reports a captcha / 2FA / unconfirmed login / missing credentials, call ask_human and stop. Once the user has logged in by hand, the session persists and login just confirms.

Reading offers: prefer list_offers(site) — it returns the FULL offer list (all offers, not just what's on screen), already valued in USD, sorted by value, each tagged (NEW) or already-seen, with a [key=...]. For Swagbucks it reads the offers API so you get all of them (e.g. 60), not the ~19 visible cards. Use observe_page (which also sees inside iframes) for navigating or reading a specific page.

The headline reward is NOT the whole story — an "up to N SB" offer may require deep game grinding, big deposits, or spend. So judge each promising NEW offer by its details:
- Swagbucks: call assess_offer(site, key). It classifies using the offer's real requirements and per-goal SB breakdown (e.g. "Direct Deposit = 35,000 SB" vs "Reach Level 60 = 600 SB"), returning the realistically attainable value — not the headline. Use offer_details(site, key) if you want to read the raw requirements yourself.
- TopCashback: call classify_portal_offer(merchant, reward, description) with the rate you read.
You don't need to assess obvious tiny offers or already-seen ones.

Avoid re-surfacing offers: for every offer you actually report, call note_offer(site, offer_key, merchant, reward, label, estimated_value) using the [key=...] from list_offers. list_offers already marks which are NEW vs seen; report only NEW MONEYMAKERS and give a one-line count of how many previously-seen offers you skipped.

UNCERTAIN result: call ask_human with the classifier's question and stop. When the user answers, call update_profile to record the fact, then re-classify and continue.

Site specifics:
- TopCashback: rates live on merchant pages (/<slug>/) and category pages (/category/<name>/). Cashback is earned by clicking through at purchase time, so there is nothing to pre-activate — just read rates and report.
- Swagbucks: rewards are in SB (≈ $0.01 each). Get the paid offers with list_offers("swagbucks") (defaults to /discover-new/featured, the full featured list). Other sections you can observe_page: /surveys, /games-new, /shop (cashback in SB), /invite (referrals). Offers link out via /offer/click — a click-through you do NOT follow. Judge effort vs reward: a quick signup for 1500 SB ($15) is a MONEYMAKER; a game/casino requiring weeks of play or deposits for its SB usually is not.

When done, report a ranked list of MONEYMAKERS (offer, reward, estimated USD value), note what you SKIPPED and why, and anything you escalated. Do not click through any offer.

Adapt to the page you actually see. If observe_page shows something unexpected (a redirect, interstitial, or layout you don't recognise), reason from the elements list; if you genuinely can't proceed call ask_human, or abort_workflow if it isn't worth a human's time. If observe results carry a 'stuck' warning, do not keep clicking the same thing — escalate or abort.
"""

_MODEL = "gemini-3.1-flash-lite"

# Tools shared by all portal agents.
_PORTAL_TOOLS = [
    portal_login, go_to, observe_page, list_offers, offer_details, assess_offer,
    scroll_page, click_element, fill_field, classify_portal_offer, note_offer,
]

# General portal agent (handles any whitelisted site) — for standalone use.
portal_agent = LlmAgent(
    model=_MODEL,
    name="portal_agent",
    description="Logs into whitelisted rewards portals (TopCashback, Swagbucks) and finds MONEYMAKER offers.",
    instruction=_INSTRUCTION,
    tools=_PORTAL_TOOLS + [ask_human, abort_workflow, update_profile],
)


# When run under the orchestrator, a site agent returns its findings as text
# (the orchestrator relays questions to the human) rather than pausing itself.
_ORCHESTRATED_PROTOCOL = """

You are invoked by an orchestrator, not the human directly. Do NOT call ask_human. Return your result in exactly this shape:
MONEYMAKERS:
- <offer/merchant> | <reward> | ~$<value> | <one-line why>
(write 'none' if you found no new MONEYMAKERS)
QUESTIONS:
- <a specific human-answerable question for any UNCERTAIN offer worth resolving>
(omit the QUESTIONS section if there are none)
Keep it concise — the orchestrator aggregates across sites."""


def _site_agent(site: str) -> LlmAgent:
    return LlmAgent(
        model=_MODEL,
        name=f"{site}_agent",
        description=f"Finds new MONEYMAKER offers on {site}.",
        instruction=(_INSTRUCTION + f"\n\nIMPORTANT: You handle ONLY {site}. Ignore other sites."
                     + _ORCHESTRATED_PROTOCOL),
        tools=_PORTAL_TOOLS,
    )


topcashback_agent = _site_agent("topcashback")
swagbucks_agent = _site_agent("swagbucks")

_APP = "portal_agent"


def _is_503(exc: BaseException) -> bool:
    return "503" in str(exc) or "UNAVAILABLE" in str(exc)


@retry(
    retry=retry_if_exception(_is_503),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    stop=stop_after_attempt(5),
    reraise=True,
)
async def _send(runner, session_id, text) -> None:
    events = runner.run_async(
        user_id="user",
        session_id=session_id,
        new_message=types.Content(role="user", parts=[types.Part(text=text)]),
    )
    async for event in events:
        if event.is_final_response() and event.content and event.content.parts:
            for part in event.content.parts:
                if getattr(part, "text", None):
                    print(part.text)


async def _main(one_shot: str | None = None) -> None:
    session_service = InMemorySessionService()
    runner = Runner(agent=portal_agent, app_name=_APP, session_service=session_service)
    session = await session_service.create_session(app_name=_APP, user_id="user")

    try:
        if one_shot:
            await _send(runner, session.id, one_shot)
            return
        print("Portal agent ready (e.g. 'check topcashback'). Ctrl+C or Ctrl+D to exit.\n")
        while True:
            try:
                text = input("> ").strip()
            except (KeyboardInterrupt, EOFError):
                print()
                break
            if not text:
                continue
            await _send(runner, session.id, text)
            print()
    finally:
        await close_session()


def main() -> None:
    # Anything after the script name is treated as a single message (one-shot).
    msg = " ".join(sys.argv[1:]).strip() or None
    asyncio.run(_main(msg))


if __name__ == "__main__":
    main()
