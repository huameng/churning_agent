import hashlib
import json
from dataclasses import dataclass, asdict
from datetime import date, datetime, timedelta
from pathlib import Path

from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

_DATE_FORMATS = ["%B %d, %Y", "%b %d, %Y"]  # "June 6, 2026" or "Jun 6, 2026"
_HEADING_LEVELS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}
from churning_agent._paths import DATA_DIR
_CACHE_DIR = DATA_DIR / "post_cache"

DOC_URL = "https://www.doctorofcredit.com/"
DOC_PAGE_URL = "https://www.doctorofcredit.com/page/{page}/"
MAX_PAGES = 10


@dataclass
class Post:
    title: str
    url: str
    date: str  # ISO format string, easier to pass through ADK tool returns


async def fetch_posts(days_back: int = 1) -> list[dict]:
    """
    Fetch recent post listings from Doctor of Credit (titles + URLs only, no content).
    Paginates through up to 10 pages, stopping once posts older than the cutoff are found.

    Args:
        days_back: How many days of posts to fetch (1 = today only, 2 = today + yesterday, etc.)

    Returns:
        List of dicts with title, url, date fields. Ordered newest first.
    """
    cutoff = date.today() - timedelta(days=days_back - 1)
    results: list[dict] = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()
        await page.set_extra_http_headers({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })

        for page_num in range(1, MAX_PAGES + 1):
            url = DOC_URL if page_num == 1 else DOC_PAGE_URL.format(page=page_num)
            await page.goto(url)
            await page.wait_for_load_state("networkidle")
            html = await page.content()

            posts = _parse_listings(html)
            if not posts:
                break  # no more pages

            for post in posts:
                post_date = datetime.fromisoformat(post.date).date()
                if post_date >= cutoff:
                    results.append(asdict(post))

            # If the oldest post on this page is already before the cutoff, we're done
            oldest_on_page = min(datetime.fromisoformat(p.date).date() for p in posts)
            if oldest_on_page < cutoff:
                break

        await browser.close()

    return results


async def fetch_offer_section(url: str, title: str = "") -> str:
    """
    Fetch the 'The Offer' section of a DoC post, using a local cache when available.
    Cache is permanent — post content doesn't change.

    Returns plain text of the offer section, or first ~2000 chars of the article as fallback.
    """
    cached = _read_cache(url)
    if cached is not None:
        return cached

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(url)
        await page.wait_for_load_state("networkidle")
        html = await page.content()
        await browser.close()

    content = _extract_offer_section(html)
    _write_cache(url, content, title)
    return content


def _cache_path(url: str) -> Path:
    return _CACHE_DIR / f"{hashlib.md5(url.encode()).hexdigest()}.json"


def _read_cache(url: str) -> str | None:
    path = _cache_path(url)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))["offer_section"]


def _write_cache(url: str, offer_section: str, title: str = "") -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path(url).write_text(
        json.dumps({"url": url, "title": title, "fetched_at": datetime.utcnow().isoformat(), "offer_section": offer_section}),
        encoding="utf-8",
    )


def _extract_offer_section(html: str) -> str:
    """
    Extract the 'The Offer' section from a DoC post's HTML.
    Looks for h2–h4 or <p><strong> headings containing 'the offer'.
    Falls back to the first 2000 chars of article text if not found.
    """
    soup = BeautifulSoup(html, "html.parser")
    article = soup.find("div", class_="entry-content") or soup.find("article")
    if not article:
        return ""

    # Look for a heading tag (h2-h4) whose text contains "the offer"
    offer_heading = None
    for tag in article.find_all(["h2", "h3", "h4"]):
        if "the offer" in tag.get_text(strip=True).lower():
            offer_heading = tag
            break

    # Fallback: <p><strong>The Offer</strong></p> pattern
    if offer_heading is None:
        for strong in article.find_all("strong"):
            if "the offer" in strong.get_text(strip=True).lower():
                offer_heading = strong.find_parent(["p", "div"]) or strong
                break

    if offer_heading is None:
        return article.get_text(separator="\n", strip=True)[:2000]

    heading_level = _HEADING_LEVELS.get(offer_heading.name, 4)
    parts = [offer_heading.get_text(strip=True)]
    for sibling in offer_heading.find_next_siblings():
        sib_level = _HEADING_LEVELS.get(sibling.name, 99)
        if sib_level <= heading_level:
            break
        text = sibling.get_text(separator="\n", strip=True)
        if text:
            parts.append(text)

    return "\n".join(parts)


def _parse_date(text: str) -> date | None:
    """Parse DoC date strings like 'June 6, 2026'."""
    text = text.strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    # Fallback: try ISO
    try:
        return datetime.fromisoformat(text).date()
    except (ValueError, TypeError):
        return None


def _parse_listings(html: str) -> list[Post]:
    soup = BeautifulSoup(html, "html.parser")
    posts = []

    for article in soup.find_all("article"):
        title_el = article.find("h2") or article.find("h1")
        link_el = article.find("a", href=True)

        # DoC uses <span class="updated"> instead of <time datetime>
        date_span = article.find("span", class_="updated")
        time_el = article.find("time")

        if not (title_el and link_el):
            continue

        post_date = None
        if date_span:
            post_date = _parse_date(date_span.get_text())
        elif time_el:
            post_date = _parse_date(time_el.get("datetime", "") or time_el.get_text())

        if post_date is None:
            continue

        posts.append(Post(
            title=title_el.get_text(strip=True),
            url=link_el["href"],
            date=post_date.isoformat(),
        ))

    return posts
