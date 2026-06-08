"""
Per-site structured offer extraction.

Swagbucks renders only a slice of its offers as DOM cards (carousels lazy-mount
~19 of 60), so we instead read the page's own offers API response — it returns
the full list with stable ids, reward points, and metadata. The DOM parser is
kept as a generic fallback for sites without an API hook.
"""

import asyncio

# Swagbucks' Discover page fetches offers from this third-party endpoint. We
# don't navigate to it (the guard would block that) — we read the response the
# whitelisted page itself makes.
SWAGBUCKS_OFFERS_API = "disco-hub.prodegeapis.com/promo/api/offers"


def _build_detail(it: dict, things: list[str], events: list[dict]) -> str:
    """Human-readable detail for the classifier: what you must DO to earn the SB,
    and how the reward is split across goals (so 'up to N SB' isn't taken at face
    value)."""
    parts = []
    desc = (it.get("shortDescription") or it.get("description") or "").strip()
    if desc:
        parts.append(desc[:300])
    if things:
        parts.append("Requirements / things to know:")
        parts += [f"- {t}" for t in things[:8]]
    if events:
        parts.append("Reward breakdown (SB per goal):")
        parts += [f"- {e['name']}: {e['sb']} SB" for e in events[:25] if e["sb"] is not None]
    return "\n".join(parts)


def parse_swagbucks_api(payload: dict) -> list[dict]:
    """Turn one offers-API JSON payload into offer dicts. Pure + unit-testable.

    Captures not just title/reward but the requirements and per-goal reward
    breakdown, so the classifier can judge realistically attainable value."""
    out = []
    for it in payload.get("content") or []:
        pts = it.get("totalPoints")
        reward = (f"{'up to ' if it.get('useEarnUpTo') else ''}{pts:,} SB"
                  if isinstance(pts, int) else "SB reward")
        things = [t for t in (it.get("thingsToKnow") or []) if isinstance(t, str)]
        if not things and it.get("requirements"):
            things = [it["requirements"]]
        events = [
            {"name": e.get("name"), "sb": e.get("flatPoints")}
            for e in (it.get("events") or [])
            if isinstance(e, dict) and e.get("payable") and e.get("name")
        ]
        out.append({
            "key": str(it.get("id") or ""),
            "title": (it.get("productName") or it.get("name") or it.get("anchor") or "").strip(),
            "reward": reward,
            "sb": pts,
            "is_game": bool(it.get("isGame")),
            "things": things,
            "events": events,
            "detail": _build_detail(it, things, events),
        })
    return [o for o in out if o["key"] and o["title"]]


async def fetch_swagbucks_offers(session, offers_url: str) -> list[dict]:
    """Navigate to the Swagbucks offers page and capture the full offer list
    from its API response(s). Returns all offers (merged across API calls)."""
    captured: list[dict] = []

    async def on_resp(resp):
        if SWAGBUCKS_OFFERS_API in resp.url:
            try:
                captured.append(await resp.json())
            except Exception:
                pass

    handler = lambda r: asyncio.create_task(on_resp(r))  # noqa: E731
    session.page.on("response", handler)
    try:
        await session.navigate(offers_url)
        await session.page.wait_for_timeout(9000)  # let the offers API resolve
    finally:
        session.page.remove_listener("response", handler)

    by_key: dict[str, dict] = {}
    for payload in captured:
        for o in parse_swagbucks_api(payload):
            by_key[o["key"]] = o
    return list(by_key.values())


# ── Generic DOM card parser (fallback / sites without an API hook) ────────────

_SWAGBUCKS_DOM_JS = r"""
() => {
  const out = [];
  document.querySelectorAll("[class*='unified_card']").forEach(c => {
    const t = (c.innerText || '').replace(/\s+/g, ' ').trim();
    const m = t.match(/((?:up to )?[\d,]+)\s*SB/i);
    if (!m) return;
    const reward = m[0].trim();
    const title = t.replace(/earn\s*/i, '').replace(m[0], '').trim().slice(0, 60);
    if (title) out.push({title, reward});
  });
  return out;
}
"""

_DOM_PARSERS = {"swagbucks": _SWAGBUCKS_DOM_JS}


async def parse_offers(session, site: str) -> list[dict]:
    """Extract offers ({title, reward}) from the current page's DOM cards, across
    all frames. Returns [] for sites without a DOM parser."""
    js = _DOM_PARSERS.get(site)
    if js is None:
        return []
    results: list[dict] = []
    seen: set[str] = set()
    for frame in session.page.frames:
        try:
            for off in await frame.evaluate(js):
                if off["title"] not in seen:
                    seen.add(off["title"])
                    results.append(off)
        except Exception:
            continue
    return results
