"""
Utility (run manually) to cache real public pages as test fixtures, so unit
tests can run against real HTML without hitting the network.

    uv run python -m churning_agent.tests.cache_pages

Only fetches public, logged-out pages. Saves into tests/fixtures/real/.
"""

import asyncio
from pathlib import Path

from playwright.async_api import async_playwright

_OUT = Path(__file__).parent / "fixtures" / "real"

PAGES = {
    "topcashback_home.html": "https://www.topcashback.com/",
    "topcashback_onlinecashback.html": "https://www.topcashback.com/onlinecashback/",
}


async def _main() -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()
        for name, url in PAGES.items():
            try:
                await page.goto(url, timeout=30000)
                await page.wait_for_load_state("networkidle", timeout=15000)
            except Exception as e:
                print(f"  [warn] {url}: {e}")
            html = await page.content()
            (_OUT / name).write_text(html, encoding="utf-8")
            print(f"  saved {name} ({len(html)} bytes)")
        await browser.close()


if __name__ == "__main__":
    asyncio.run(_main())
