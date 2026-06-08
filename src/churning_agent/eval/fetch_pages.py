"""
Fetch post listings from a specific page range of Doctor of Credit and cache
their offer sections. No date filtering — useful for building eval datasets.

Run from churning_agent/:
    uv run python -m churning_agent.eval.fetch_pages --start 11 --end 20
"""

import asyncio
from pathlib import Path

from playwright.async_api import async_playwright
from rich.console import Console
from rich.progress import track

from churning_agent.tools.scraper import (
    DOC_URL, DOC_PAGE_URL, _parse_listings,
    fetch_offer_section, _read_cache,
)

console = Console()


async def fetch_pages(start: int, end: int) -> None:
    console.print(f"Fetching post listings from pages {start}–{end}...")

    all_posts = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()
        await page.set_extra_http_headers({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })

        for page_num in range(start, end + 1):
            url = DOC_URL if page_num == 1 else DOC_PAGE_URL.format(page=page_num)
            console.print(f"  [dim]Page {page_num}: {url}[/dim]")
            await page.goto(url)
            await page.wait_for_load_state("networkidle")
            html = await page.content()
            posts = _parse_listings(html)
            console.print(f"  Found {len(posts)} posts")
            all_posts.extend(posts)

        await browser.close()

    console.print(f"\nFetched {len(all_posts)} posts total. Caching offer sections...")

    skipped = 0
    for post in track(all_posts, description="Fetching offer sections"):
        if _read_cache(post.url) is not None:
            skipped += 1
            continue
        await fetch_offer_section(post.url, post.title)

    fetched = len(all_posts) - skipped
    console.print(f"\n[green]Done.[/green] {fetched} fetched, {skipped} already cached.")
    console.print("Run [bold]uv run python -m churning_agent.eval.generate_cases[/bold] to rebuild cases.json.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fetch DoC pages into the offer section cache")
    parser.add_argument("--start", type=int, required=True, help="First page number")
    parser.add_argument("--end", type=int, required=True, help="Last page number (inclusive)")
    args = parser.parse_args()
    asyncio.run(fetch_pages(args.start, args.end))
