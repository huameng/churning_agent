"""Quick debug script to test fetch_posts."""
import asyncio

from churning_agent.tools.scraper import fetch_posts


async def main():
    print("Fetching DoC front page (days_back=3)...")
    posts = await fetch_posts(days_back=3)
    print(f"Posts found: {len(posts)}")
    for p in posts:
        print(f"  {p['date']}: {p['title'][:70]}")


asyncio.run(main())
