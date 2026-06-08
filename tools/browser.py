"""
BrowserSession: a persistent, perceiving, acting browser the agent drives.

Unlike scraper.py (fetch-once-and-close), this keeps one headed browser with a
persistent profile so logins survive between runs. The agent perceives pages as
a numbered list of interactive elements (an accessibility-style snapshot, not
pixels) and acts by element ref.

Safety backstop: every action is checked against `allow_url` (default
sites.is_allowed). If the current/target page isn't whitelisted, the action
refuses. The agent literally cannot click outside the whitelist.
"""

import asyncio
from collections.abc import Callable
from pathlib import Path

from playwright.async_api import async_playwright
from pydantic import BaseModel

from . import sites

_PROFILE_DIR = Path(__file__).parent.parent / "data" / "browser_profile"
_TEXT_LIMIT = 3000

# JS run on each observe(): tag every visible interactive element with a stable
# ref and return its role/name so the agent can act on it by ref.
_OBSERVE_JS = r"""
(startRef) => {
  const SEL = 'a[href], button, input, select, textarea, [role=button], ' +
              '[role=link], [role=checkbox], [role=tab], [role=menuitem], [onclick]';
  const isVisible = (el) => {
    const r = el.getBoundingClientRect();
    if (r.width === 0 && r.height === 0) return false;
    const s = window.getComputedStyle(el);
    return s.visibility !== 'hidden' && s.display !== 'none';
  };
  const nameOf = (el) => {
    return (el.getAttribute('aria-label') ||
            (el.innerText || '').trim() ||
            el.getAttribute('placeholder') ||
            el.getAttribute('value') ||
            el.getAttribute('alt') ||
            el.getAttribute('title') || '').trim().slice(0, 200);
  };
  const els = [];
  let ref = startRef;
  document.querySelectorAll(SEL).forEach((el) => {
    if (!isVisible(el)) return;
    el.setAttribute('data-agent-ref', String(ref));
    const tag = el.tagName.toLowerCase();
    const role = el.getAttribute('role') ||
                 (tag === 'a' ? 'link' :
                  tag === 'input' ? (el.getAttribute('type') || 'text') : tag);
    els.push({
      ref: ref,
      role: role,
      name: nameOf(el),
      value: el.value !== undefined ? String(el.value) : null,
      enabled: !el.disabled,
    });
    ref += 1;
  });
  return {
    title: document.title,
    text: (document.body ? document.body.innerText : '').trim(),
    elements: els,
  };
}
"""


class Element(BaseModel):
    ref: int
    role: str
    name: str
    value: str | None = None
    enabled: bool = True


class PageObservation(BaseModel):
    url: str
    title: str
    elements: list[Element]
    text: str

    def signature(self) -> str:
        """Stable fingerprint of the page state, for the stuck detector."""
        parts = [self.url] + [f"{e.role}:{e.name}" for e in self.elements]
        return "|".join(parts)

    def summary(self) -> str:
        """Compact text rendering for an LLM tool result."""
        lines = [f"URL: {self.url}", f"Title: {self.title}", "", "Interactive elements:"]
        for e in self.elements:
            val = f" value={e.value!r}" if e.value else ""
            dis = "" if e.enabled else " (disabled)"
            lines.append(f"  [{e.ref}] {e.role}: {e.name!r}{val}{dis}")
        lines += ["", "Page text:", self.text[:_TEXT_LIMIT]]
        return "\n".join(lines)


class ActionResult(BaseModel):
    success: bool
    observation: PageObservation | None = None
    error: str | None = None


class BrowserSession:
    def __init__(
        self,
        user_data_dir: Path | None = None,
        headless: bool = False,
        allow_url: Callable[[str], bool] = sites.is_allowed,
        page=None,
    ):
        self._user_data_dir = user_data_dir or _PROFILE_DIR
        self._headless = headless
        self._allow_url = allow_url
        self._pw = None
        self._context = None
        self.page = page            # injected (pooled) or None (owns its context)
        self._owns_context = page is None
        self._ref_frames: dict[int, object] = {}  # ref -> Frame that owns it

    async def start(self) -> None:
        if self.page is not None:
            return
        self._user_data_dir.mkdir(parents=True, exist_ok=True)
        self._pw = await async_playwright().start()
        self._context = await self._pw.chromium.launch_persistent_context(
            str(self._user_data_dir), headless=self._headless
        )
        self.page = self._context.pages[0] if self._context.pages else await self._context.new_page()

    async def stop(self) -> None:
        # Pooled sessions don't own the shared context — leave it to the pool.
        if self._owns_context:
            if self._context:
                await self._context.close()
            if self._pw:
                await self._pw.stop()
        self._pw = self._context = self.page = None

    # ── perception ───────────────────────────────────────────────────────────

    async def observe(self) -> PageObservation:
        """Observe the page across all frames (offer walls render in iframes)."""
        elements: list[Element] = []
        self._ref_frames = {}
        next_ref = 0
        top_title = top_text = ""
        for frame in self.page.frames:
            try:
                raw = await frame.evaluate(_OBSERVE_JS, next_ref)
            except Exception:
                continue  # frame detached or not yet ready; skip it
            for e in raw["elements"]:
                self._ref_frames[e["ref"]] = frame
                elements.append(Element(**e))
            next_ref += len(raw["elements"])
            if frame is self.page.main_frame:
                top_title, top_text = raw["title"], raw["text"]
        return PageObservation(
            url=self.page.url, title=top_title, elements=elements, text=top_text
        )

    # ── actions (each guarded, each returns the resulting observation) ─────────

    async def navigate(self, url: str) -> ActionResult:
        if not self._allow_url(url):
            return ActionResult(success=False, error=f"Blocked: {url} is not a whitelisted site.")
        await self.page.goto(url, wait_until="domcontentloaded")
        return ActionResult(success=True, observation=await self.observe())

    async def click(self, ref: int) -> ActionResult:
        return await self._on_element(ref, lambda loc: loc.click())

    async def fill(self, ref: int, text: str) -> ActionResult:
        return await self._on_element(ref, lambda loc: loc.fill(text))

    async def press(self, key: str) -> ActionResult:
        guard = self._guard_current()
        if guard:
            return guard
        await self.page.keyboard.press(key)
        await self.page.wait_for_load_state("domcontentloaded")
        return ActionResult(success=True, observation=await self.observe())

    async def scroll(self, direction: str = "down") -> ActionResult:
        guard = self._guard_current()
        if guard:
            return guard
        dy = -800 if direction == "up" else 800
        await self.page.mouse.wheel(0, dy)
        return ActionResult(success=True, observation=await self.observe())

    # ── internals ──────────────────────────────────────────────────────────────

    def _guard_current(self) -> ActionResult | None:
        if not self._allow_url(self.page.url):
            return ActionResult(
                success=False,
                error=f"Blocked: current page {self.page.url} is not a whitelisted site.",
            )
        return None

    async def _on_element(self, ref: int, action) -> ActionResult:
        guard = self._guard_current()
        if guard:
            return guard
        frame = self._ref_frames.get(ref, self.page)
        loc = frame.locator(f"[data-agent-ref='{ref}']")
        if await loc.count() == 0:
            return ActionResult(success=False, error=f"No element with ref {ref}. Re-observe first.")
        await action(loc.first)
        await self.page.wait_for_load_state("domcontentloaded")
        return ActionResult(success=True, observation=await self.observe())


# ── Shared-context pool: one persistent browser, one page (tab) per lane ──────
# Lanes let independent agents (e.g. topcashback, swagbucks) drive their own tab
# concurrently while sharing the same logged-in profile/cookies.

_pw = None
_context = None
_sessions: dict[str, BrowserSession] = {}
_context_lock: asyncio.Lock = asyncio.Lock()
_POOL_HEADLESS = False  # headed by default (Cloudflare/bot bypass); tests flip this


async def _ensure_context():
    global _pw, _context
    if _context is None:
        _PROFILE_DIR.mkdir(parents=True, exist_ok=True)
        _pw = await async_playwright().start()
        _context = await _pw.chromium.launch_persistent_context(
            str(_PROFILE_DIR), headless=_POOL_HEADLESS
        )
    return _context


async def get_session(lane: str = "default") -> BrowserSession:
    """Return the BrowserSession for a lane (its own tab in the shared browser),
    creating it on first use. Different lanes can be driven concurrently."""
    if lane not in _sessions:
        async with _context_lock:
            if lane not in _sessions:                     # double-check under lock
                ctx = await _ensure_context()
                # Reuse the context's initial blank tab for the first lane.
                page = ctx.pages[0] if (ctx.pages and not _sessions) else await ctx.new_page()
                _sessions[lane] = BrowserSession(page=page)
    return _sessions[lane]


async def close_session() -> None:
    """Close every lane and the shared browser context."""
    global _pw, _context, _sessions
    if _context is not None:
        await _context.close()
    if _pw is not None:
        await _pw.stop()
    _pw = _context = None
    _sessions = {}
