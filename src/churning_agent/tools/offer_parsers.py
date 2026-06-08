"""
Per-site structured offer extraction.

Swagbucks renders only a slice of its offers as DOM cards (carousels lazy-mount
~19 of 60), so we instead read the page's own offers API response — it returns
the full list with stable ids, reward points, and metadata. The DOM parser is
kept as a generic fallback for sites without an API hook.
"""

import asyncio

from .offer import Offer, RewardGoal, RewardValue


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


def parse_swagbucks_api(payload: dict) -> list[Offer]:
    """Turn one offers-API JSON payload into Offers. Pure + unit-testable.

    Captures not just title/reward but the requirements and per-goal reward
    breakdown, so the classifier can judge realistically attainable value."""
    out = []
    for it in payload.get("content") or []:
        pts = it.get("totalPoints")
        reward_text = (f"{'up to ' if it.get('useEarnUpTo') else ''}{pts:,} SB"
                       if isinstance(pts, int) else "SB reward")
        things = [t for t in (it.get("thingsToKnow") or []) if isinstance(t, str)]
        if not things and it.get("requirements"):
            things = [it["requirements"]]
        events = [
            {"name": e.get("name"), "sb": e.get("flatPoints")}
            for e in (it.get("events") or [])
            if isinstance(e, dict) and e.get("payable") and e.get("name")
        ]
        out.append(Offer(
            site="swagbucks",
            key=str(it.get("id") or ""),
            title=(it.get("productName") or it.get("name") or it.get("anchor") or "").strip(),
            reward_text=reward_text,
            reward=RewardValue(amount=pts, unit="SB") if isinstance(pts, int) else None,
            is_game=bool(it.get("isGame")),
            requirements=things,
            reward_breakdown=[RewardGoal(name=e["name"], sb=e["sb"]) for e in events],
            detail=_build_detail(it, things, events),
        ))
    return [o for o in out if o.key and o.title]


# How to turn a site's offers-API payload into Offers. Add an entry to enable
# the API fetch path for a new site (and set offers_api_host on its adapter).
API_PARSERS = {"swagbucks": parse_swagbucks_api}


async def fetch_api_offers(session, offers_url: str, api_host: str, parser) -> list[Offer]:
    """Navigate to a site's offers page and capture its full offer list from the
    background API response(s) at `api_host`, parsed via `parser`. Returns all
    offers merged across API calls (deduped by key)."""
    captured: list[dict] = []

    async def on_resp(resp):
        if api_host in resp.url:
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

    by_key: dict[str, Offer] = {}
    for payload in captured:
        for o in parser(payload):
            by_key[o.key] = o
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


async def parse_offers(session, site: str) -> list[Offer]:
    """Extract Offers from the current page's DOM cards, across all frames.
    Returns [] for sites without a DOM parser. The reward unit is inferred from
    the reward text (SB / £ / $ / %)."""
    js = _DOM_PARSERS.get(site)
    if js is None:
        return []
    results: list[Offer] = []
    seen: set[str] = set()
    for frame in session.page.frames:
        try:
            for off in await frame.evaluate(js):
                if off["title"] not in seen:
                    seen.add(off["title"])
                    results.append(Offer(
                        site=site,
                        key=off["title"],
                        title=off["title"],
                        reward_text=off["reward"],
                        reward=RewardValue.parse(off["reward"]),
                    ))
        except Exception:
            continue
    return results
