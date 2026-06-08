"""
Portal agent: drives a whitelisted rewards portal (currently TopCashback) to
find and activate worthwhile cashback offers.

Used two ways:
  - As a sub-agent of the churning_agent root agent (see agent.py), so the DoC
    monitor can hand off when a post points at a portal offer.
  - Standalone:  uv run python portal.py
"""

import asyncio
import sys

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from churning_agent.tools.browser import close_session
from churning_agent.tools.escalation import abort_workflow, ask_human
from churning_agent.tools.offer_classifier import classify_portal_offer
from churning_agent.tools.offer_log import query_seen_offers
from churning_agent.tools.portal_tools import (
    assess_offer, click_element, fill_field, go_to, list_offers, note_offer,
    observe_page, offer_details, portal_login, scroll_page,
)
from churning_agent.tools.profile import update_profile
from churning_agent import prompts
from churning_agent.llm import retrying_model

# Model + instructions live in config/prompts/ (portal_agent + the orchestrated
# protocol fragment appended to per-site agents).
_PROMPT = prompts.load("portal_agent")
_MODEL = _PROMPT.model
_INSTRUCTION = _PROMPT.system
_ORCHESTRATED_PROTOCOL = prompts.load("portal_orchestrated_protocol").system

# Tools shared by all portal agents.
_PORTAL_TOOLS = [
    portal_login, go_to, observe_page, list_offers, offer_details, assess_offer,
    scroll_page, click_element, fill_field, classify_portal_offer, note_offer,
    query_seen_offers,
]

# General portal agent (handles any whitelisted site) — for standalone use.
portal_agent = LlmAgent(
    model=retrying_model(_MODEL),
    name="portal_agent",
    description="Logs into whitelisted rewards portals (TopCashback, Swagbucks) and finds MONEYMAKER offers.",
    instruction=_INSTRUCTION,
    tools=_PORTAL_TOOLS + [ask_human, abort_workflow, update_profile],
)


# When run under the orchestrator, a site agent returns its findings as text
# (the orchestrator relays questions to the human) rather than pausing itself.
def _site_agent(site: str) -> LlmAgent:
    return LlmAgent(
        model=retrying_model(_MODEL),
        name=f"{site}_agent",
        description=f"Finds new MONEYMAKER offers on {site}.",
        instruction=(_INSTRUCTION + f"\n\nIMPORTANT: You handle ONLY {site}. Ignore other sites."
                     + "\n\n" + _ORCHESTRATED_PROTOCOL),
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
