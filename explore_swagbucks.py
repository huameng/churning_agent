"""
Authenticated Swagbucks offer-discovery. Logs in (persistent profile), then
probes the main offer sections and dumps how each renders offers + their SB
rewards, so we can teach the agent to parse the many offer types.

    uv run python explore_swagbucks.py
"""

import asyncio

from churning_agent.tools.browser import close_session, get_session
from churning_agent.tools.portal_tools import portal_login

# Known-good offer sections (confirmed against the live site). The high-value
# "Discover" paid offers live behind login at a URL we learn from the nav menu.
_SECTIONS = {
    "surveys": "https://www.swagbucks.com/g/paid-surveys",
    "shop":    "https://www.swagbucks.com/shop",
    "games":   "https://www.swagbucks.com/games",
}


async def _sb_offers(page) -> list[str]:
    """Text snippets that mention an SB reward — the offer signal."""
    return await page.evaluate(
        r"""() => {
            const out = new Set();
            document.querySelectorAll('body *').forEach(e => {
                if (e.childElementCount > 3) return;
                const t = (e.innerText || '').trim().replace(/\s+/g, ' ');
                if (t.length > 4 && t.length < 90 && /\b\d[\d,]*\s*SB\b/.test(t)) out.add(t);
            });
            return Array.from(out).slice(0, 12);
        }"""
    )


async def _logged_in(page) -> dict:
    return await page.evaluate(
        """() => ({
            logout: !!document.querySelector("a[href*='logout']"),
            account: !!document.querySelector("a[href*='/account']"),
            sbBalance: (document.querySelector("#sbValueHeader,.sb-value,[class*='balance']") || {}).innerText || null,
        })"""
    )


async def _main() -> None:
    print("=== LOGIN ===")
    print(await portal_login("swagbucks"))
    session = await get_session()
    page = session.page
    await session.navigate("https://www.swagbucks.com/")
    await page.wait_for_timeout(2000)
    print("URL:", page.url, "| TITLE:", await page.title())
    print("LOGGED-IN SIGNALS:", await _logged_in(page))

    # Dump the nav/offer links so we learn the real (logged-in) section URLs,
    # e.g. Discover / Answer / Offers — the high-value paid offers.
    links = await page.evaluate(
        r"""() => Array.from(document.querySelectorAll('a[href]'))
            .map(a => ({t:(a.innerText||'').trim().replace(/\s+/g,' ').slice(0,24), h:a.getAttribute('href')}))
            .filter(x => x.t && /offer|discover|answer|earn|watch|shop|survey|play|gift|deal|daily/i.test(x.t+x.h))
            .slice(0, 40)"""
    )
    print("--- offer-section nav links ---")
    seen = set()
    for x in links:
        if x["h"] not in seen:
            seen.add(x["h"])
            print(f"  {x['t']!r:26} -> {x['h']}")

    # Targeted parse of the Discover paid offers (the MONEYMAKERs).
    print("\n=== DISCOVER (/discover-new) — paid offers ===")
    nav = await session.navigate("https://www.swagbucks.com/discover-new")
    await page.wait_for_timeout(4000)
    print("URL:", page.url, "| TITLE:", await page.title(), "| frames:", len(page.frames))
    card_js = r"""() => {
        const out = [];
        document.querySelectorAll("a[href*='/offer/click'], a[href*='/games/apps'], a[href*='offerID']").forEach(a => {
            const t = (a.innerText || a.getAttribute('aria-label') || '').trim().replace(/\s+/g, ' ');
            if (t) out.push({t: t.slice(0, 70), h: (a.getAttribute('href')||'').slice(0, 50)});
        });
        return out.slice(0, 25);
    }"""
    # The real test: does the framework's frame-aware observe() see the offers?
    obs = await session.observe()
    sb_els = [e for e in obs.elements if "sb" in e.name.lower() and any(c.isdigit() for c in e.name)]
    print(f"  observe() found {len(obs.elements)} elements total; "
          f"{len(sb_els)} mention an SB reward:")
    for e in sb_els[:15]:
        print(f"      [{e.ref}] {e.role}: {e.name[:60]!r}")

    for label, url in _SECTIONS.items():
        nav = await session.navigate(url)
        if not nav.success:
            print(f"\n[{label}] {url} -> BLOCKED: {nav.error}")
            continue
        await page.wait_for_timeout(3000)  # SB offer widgets are JS/iframe-rendered
        title = await page.title()
        offers = await _sb_offers(page)
        # also peek into iframes (offer walls are often framed)
        frame_offers = []
        for fr in page.frames[1:]:
            try:
                frame_offers += await _sb_offers(fr)
            except Exception:
                pass
        print(f"\n[{label}] {page.url}")
        print(f"  title={title!r}  frames={len(page.frames)}  elements={len(nav.observation.elements)}")
        print(f"  SB offers (main): {offers[:8]}")
        if frame_offers:
            print(f"  SB offers (iframes): {frame_offers[:8]}")

    await close_session()
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(_main())
