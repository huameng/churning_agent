"""
Log in to a whitelisted site by hand, once. Opens a visible browser on the
agent's persistent profile so you can solve any captcha / 2FA yourself; the
authenticated cookies then persist and future agent runs skip the login page.

    uv run python login_once.py            # topcashback
    uv run python login_once.py swagbucks  # (when added)
"""

import asyncio
import sys

from churning_agent.tools.browser import BrowserSession
from churning_agent.tools.sites import REGISTRY


async def _prompt(text: str) -> str:
    return await asyncio.get_event_loop().run_in_executor(None, input, text)


async def _main(site: str) -> None:
    adapter = REGISTRY.get(site)
    if adapter is None:
        print(f"'{site}' is not whitelisted. Allowed: {', '.join(REGISTRY)}.")
        return

    # Headed, persistent profile, real domain guard.
    session = BrowserSession(headless=False)
    await session.start()
    await session.navigate(adapter.login_url)

    print(f"\nA browser window is open at {site}'s login page.")
    print("Log in there by hand (solve any captcha). Then come back here and press Enter.")
    await _prompt("> ")

    try:
        await session.page.wait_for_selector(adapter.logged_in_selector, timeout=5000)
        print(f"Confirmed logged in to {site}. The session is saved — agent runs will reuse it.")
    except Exception:
        print(f"Could not confirm login (no '{adapter.logged_in_selector}' found). "
              "If you did log in, the cookies are still saved; the marker selector may need updating.")

    await session.stop()


def main() -> None:
    site = sys.argv[1] if len(sys.argv) > 1 else "topcashback"
    asyncio.run(_main(site))


if __name__ == "__main__":
    main()
