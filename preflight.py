"""
Preflight check — run this before starting the agent to catch issues early.
Run from churning_agent/: uv run python preflight.py
"""
import asyncio
import sys
import os
from pathlib import Path

errors = []
warnings = []

def ok(msg): print(f"  [OK] {msg}")
def warn(msg): warnings.append(msg); print(f"  [WARN] {msg}")
def fail(msg): errors.append(msg); print(f"  [FAIL] {msg}")


# ── 1. Imports ────────────────────────────────────────────────────────────────
print("\n1. Imports")
try:
    from churning_agent.tools.profile import load_profile
    from churning_agent.tools.state import get_last_seen_url, set_last_seen_url
    from churning_agent.tools.classifier import classify
    from churning_agent.tools.scraper import fetch_posts, fetch_offer_section, _parse_listings
    from churning_agent.tools.notify import notify_moneymaker
    from churning_agent.tools.browser import BrowserSession
    from churning_agent.tools.sites import REGISTRY, is_allowed
    from churning_agent.tools.credentials import get_credentials
    ok("all tool imports")
except Exception as e:
    fail(f"import error: {e}")

try:
    from churning_agent.agent import root_agent
    ok(f"agent loads ({len(root_agent.tools)} tools, {len(root_agent.sub_agents)} sub-agents)")
except Exception as e:
    fail(f"agent load error: {e}")


# ── 2. Environment ────────────────────────────────────────────────────────────
print("\n2. Environment")
from dotenv import load_dotenv
from churning_agent._paths import PROJECT_ROOT
load_dotenv(PROJECT_ROOT / ".env")

api_key = os.environ.get("GOOGLE_API_KEY", "")
if api_key and not api_key.startswith("your_"):
    ok("GOOGLE_API_KEY is set")
else:
    fail("GOOGLE_API_KEY missing or placeholder — copy .env.example to .env and fill it in")


# ── 3. User profile ───────────────────────────────────────────────────────────
print("\n3. User profile")
profile_path = PROJECT_ROOT / "config" / "user_profile.yaml"
if not profile_path.exists():
    fail(f"config/user_profile.yaml not found — copy config/user_profile.example.yaml and fill it in")
else:
    try:
        profile = load_profile(profile_path)
        ok(f"profile loaded (state={profile.state}, threshold=${profile.preferences.min_profit_threshold:.0f})")
    except Exception as e:
        fail(f"profile parse error: {e}")


# ── 4. State file ─────────────────────────────────────────────────────────────
print("\n4. State file")
try:
    url = get_last_seen_url()
    ok(f"state readable (last_seen_url={url!r})")
    set_last_seen_url("https://example.com/test")
    assert get_last_seen_url() == "https://example.com/test"
    set_last_seen_url(url or "")  # restore
    ok("state read/write works")
except Exception as e:
    fail(f"state error: {e}")


# ── 5. Playwright ─────────────────────────────────────────────────────────────
print("\n5. Playwright")
async def check_playwright():
    try:
        from playwright.async_api import async_playwright
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto("https://example.com")
            title = await page.title()
            await browser.close()
        ok(f"chromium launches and fetches (got title: {title!r})")
    except Exception as e:
        fail(f"playwright error: {e}")

asyncio.run(check_playwright())


# ── 6. HTML parsing ───────────────────────────────────────────────────────────
print("\n6. HTML parsing")
try:
    sample_html = """
    <article>
      <h2><a href="https://doctorofcredit.com/test">Chase $300 Checking Bonus</a></h2>
      <a href="https://doctorofcredit.com/test">Read more</a>
      <time datetime="2026-06-06T10:00:00">June 6, 2026</time>
    </article>
    """
    posts = _parse_listings(sample_html)
    if posts:
        ok(f"HTML parser finds posts (got: {posts[0].title!r})")
    else:
        warn("HTML parser returned no posts from sample — DoC's HTML structure may have changed")
except Exception as e:
    fail(f"HTML parse error: {e}")


# ── 7. Classifier (live API call) ─────────────────────────────────────────────
print("\n7. Classifier (makes a live Gemini API call)")
if errors:
    warn("skipping — fix above errors first")
else:
    try:
        result = classify(
            title="Chase Total Checking $300 Bonus (Nationwide)",
            content="Chase is offering $300 for new checking customers. Set up direct deposit within 90 days. Available nationwide.",
        )
        ok(f"classifier works: label={result.label}, value={result.estimated_value}")
    except Exception as e:
        fail(f"classifier error: {e}")


# ── 8. Portal agent (whitelist, credentials, browser observe) ─────────────────
print("\n8. Portal agent")
# Whitelist sanity: an approved site is allowed, an off-list one is blocked.
if is_allowed("https://www.topcashback.com/") and not is_allowed("https://www.chase.com/"):
    ok(f"whitelist guard works ({len(REGISTRY)} site(s): {', '.join(REGISTRY)})")
else:
    fail("whitelist guard misbehaving — check tools/sites.py")

for site in REGISTRY:
    creds = get_credentials(REGISTRY[site].credential_key)
    if creds:
        ok(f"credentials present for {site}")
    else:
        warn(f"no credentials for {site} — set {site.upper()}_EMAIL/_PASSWORD in .secrets.env to enable login")

async def check_browser_observe():
    try:
        from churning_agent.tools.browser import BrowserSession
        s = BrowserSession(headless=True, allow_url=lambda u: True)
        await s.start()
        await s.page.set_content("<button>Hi</button><a href='#'>Link</a>")
        obs = await s.observe()
        await s.stop()
        ok(f"browser observe works ({len(obs.elements)} elements found)")
    except Exception as e:
        fail(f"browser observe error: {e}")

asyncio.run(check_browser_observe())


# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*50)
if errors:
    print(f"PREFLIGHT FAILED — {len(errors)} error(s):")
    for e in errors: print(f"  • {e}")
    sys.exit(1)
elif warnings:
    print(f"PREFLIGHT PASSED with {len(warnings)} warning(s):")
    for w in warnings: print(f"  • {w}")
else:
    print("PREFLIGHT PASSED — ready to run the agent.")
