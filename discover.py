"""
Live structure-discovery for a whitelisted site: dumps login-form fields and the
nav/offer link scheme so we can fill in / correct its SiteAdapter. Site-agnostic.

    uv run python discover.py swagbucks
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from churning_agent.tools.browser import close_session, get_session
from churning_agent.tools.sites import REGISTRY


async def _form_fields(page) -> list[dict]:
    return await page.evaluate(
        """() => Array.from(document.querySelectorAll('input,button,[type=submit]'))
            .map(e => ({tag:e.tagName, type:e.type||'', id:e.id||'', name:e.name||'',
                        ph:e.placeholder||'', txt:(e.innerText||e.value||'').slice(0,25)}))
            .filter(e => e.type !== 'hidden')"""
    )


async def _links(page, limit=60) -> list[dict]:
    return await page.evaluate(
        f"""() => Array.from(document.querySelectorAll('a[href]'))
            .map(a => ({{t:(a.innerText||'').trim().replace(/\\s+/g,' ').slice(0,28), h:a.getAttribute('href')}}))
            .filter(x => x.t && (x.h||'').length > 1)
            .slice(0, {limit})"""
    )


async def _signals(page) -> dict:
    return await page.evaluate(
        """() => ({
            loggedIn: !!document.querySelector("a[href*='logout'],a[href*='signout'],a[href*='/profile']"),
            hasIframes: document.querySelectorAll('iframe').length,
        })"""
    )


async def _main(site: str) -> None:
    adapter = REGISTRY.get(site)
    if adapter is None:
        print(f"'{site}' not whitelisted. Allowed: {', '.join(REGISTRY)}.")
        return

    session = await get_session()
    page = session.page

    print(f"=== {site}: BASE ({adapter.base_url}) ===")
    nav = await session.navigate(adapter.base_url)
    if not nav.success:
        print("BLOCKED:", nav.error)
        await close_session()
        return
    print("URL:", page.url, "| TITLE:", await page.title())
    print("SIGNALS:", await _signals(page))
    print("\n--- nav/offer links (base) ---")
    for x in await _links(page):
        print(f"  {x['t']!r:30} -> {x['h']}")

    print(f"\n=== {site}: LOGIN ({adapter.login_url}) ===")
    nav = await session.navigate(adapter.login_url)
    if nav.success:
        await page.wait_for_timeout(4000)  # login form is JS-rendered
        print("URL:", page.url, "| TITLE:", await page.title())
        print(f"frames: {len(page.frames)}")
        for fr in page.frames:
            try:
                fields = await _form_fields(fr)
            except Exception as e:
                fields = f"<err {e}>"
            if fields:
                print(f"  frame url={fr.url[:60]!r}")
                for f in (fields if isinstance(fields, list) else [fields]):
                    print(f"    {f}")
    else:
        print("BLOCKED:", nav.error)

    await close_session()
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(_main(sys.argv[1] if len(sys.argv) > 1 else "swagbucks"))
