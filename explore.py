"""
Authenticated live exploration of TopCashback. Reuses the persistent profile
(log in once via login_once.py first), confirms we're logged in, then visits a
real merchant page to read the cashback rate and locate the "Get Cash Back"
control. A reusable diagnostic so we don't need ad-hoc commands.

    uv run python explore.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from churning_agent.tools.browser import close_session, get_session
from churning_agent.tools.offer_classifier import classify_portal_offer

_HOME = "https://www.topcashback.com/"
_MERCHANTS = [
    "https://www.topcashback.com/best-buy/",
    "https://www.topcashback.com/dell-home-and-home-office/",
    "https://www.topcashback.com/nordvpn/",
]


async def _signals(page) -> dict:
    return await page.evaluate(
        """() => {
            const txt = (document.body.innerText || '');
            const greet = (txt.match(/Hi[, ]+[A-Za-z]+/) || [''])[0];
            return {
                loggedIn: !!document.querySelector("a[href*='logout']"),
                greeting: greet,
                accountMenu: !!document.querySelector("a[href*='/account/']"),
            };
        }"""
    )


def _looks_like_rate(name: str) -> bool:
    n = name.lower()
    return "%" in name or "cash back" in n or "cashback" in n


async def _main() -> None:
    session = await get_session()
    page = session.page

    print("=== AUTH CHECK (homepage) ===")
    await session.navigate(_HOME)
    print("URL:", page.url, "| TITLE:", await page.title())
    print("SIGNALS:", await _signals(page))

    print("\n=== MERCHANT PAGE ===")
    for url in _MERCHANTS:
        nav = await session.navigate(url)
        if not nav.success:
            print(f"  {url} -> FAILED: {nav.error}")
            continue
        title = await page.title()
        if "not found" in title.lower():
            print(f"  {url} -> 404")
            continue

        obs = nav.observation
        # The headline rate is usually prominent text; grab rate-looking snippets.
        rate_texts = await page.evaluate(
            """() => Array.from(document.querySelectorAll('body *'))
                .map(e => (e.childElementCount === 0 ? (e.innerText||'').trim() : ''))
                .filter(t => /\\d+(\\.\\d+)?%|\\$\\d/.test(t) && t.length < 40)
                .slice(0, 8)"""
        )
        get_cb = [e for e in obs.elements
                  if any(k in e.name.lower() for k in ("get cash back", "get cashback", "shop now"))]
        rate_els = [e for e in obs.elements if _looks_like_rate(e.name)][:5]

        print(f"\n  {url}")
        print(f"    title: {title!r}")
        print(f"    rate-looking text on page: {rate_texts}")
        print(f"    'Get Cash Back'-type elements: {[(e.ref, e.role, e.name[:40]) for e in get_cb]}")
        print(f"    rate-bearing interactive els: {[(e.ref, e.name[:40]) for e in rate_els]}")

        # Dump buttons + offer links with hrefs to find the real activation control.
        controls = await page.evaluate(
            """() => Array.from(document.querySelectorAll('a[href], button'))
                .map(e => ({
                    tag: e.tagName,
                    t: (e.innerText||'').trim().replace(/\\s+/g,' ').slice(0,45),
                    h: e.getAttribute('href') || '',
                    cls: (e.className||'').toString().slice(0,40),
                }))
                .filter(x => x.t && (/cash ?back|shop|get|go to|activate/i.test(x.t) ||
                                     /redirect|out\\.php|visit|track/i.test(x.h)))
                .slice(0, 15)"""
        )
        print("    activation candidates:")
        for c in controls:
            print(f"      <{c['tag']}> {c['t']!r}  href={c['h'][:50]!r}  cls={c['cls']!r}")

        merchant = title.split(" Cash")[0].split(" cash")[0].strip()[:40]
        reward = rate_texts[0] if rate_texts else (rate_els[0].name if rate_els else "unknown")
        decision = classify_portal_offer(merchant=merchant, reward=reward)
        print(f"    CLASSIFY {merchant!r} / {reward!r} -> {decision['label']} "
              f"(${decision['estimated_value']}) :: {decision['reasoning'][:90]}")
        break  # one merchant is enough to validate the flow

    await close_session()
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(_main())
