"""Quick debug script to test fetch_posts."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from churning_agent.tools.scraper import fetch_posts


async def main():
    print("Fetching DoC front page (days_back=3)...")
    posts = await fetch_posts(days_back=3)
    print(f"Posts found: {len(posts)}")
    for p in posts:
        print(f"  {p['date']}: {p['title'][:70]}")


asyncio.run(main())
