"""
Generate eval/cases.json from cached posts.

Reads every entry in data/post_cache/, looks up titles from the classifications DB
(falls back to formatting the URL slug), runs classify() on each, and writes
eval/cases.json with the current label as `expected`.

Run from churning_agent/:
    uv run python -m churning_agent.eval.generate_cases
"""

import json
import sqlite3
import sys
from pathlib import Path

from rich.console import Console
from rich.progress import track

from churning_agent.tools.classifier import classify
from churning_agent.tools.scraper import _CACHE_DIR
from churning_agent._paths import DATA_DIR, PROJECT_ROOT

console = Console()

_EVAL_DIR = PROJECT_ROOT / "eval"
_DB_PATH = DATA_DIR / "classifications.db"
_OUT_PATH = _EVAL_DIR / "cases.json"
_EXCLUDED_PATH = _EVAL_DIR / "excluded_urls.json"


def _title_from_url(url: str) -> str:
    slug = url.rstrip("/").rsplit("/", 1)[-1]
    return slug.replace("-", " ").title()


def _load_db_titles() -> dict[str, str]:
    if not _DB_PATH.exists():
        return {}
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT url, title FROM classifications").fetchall()
    conn.close()
    return {r["url"]: r["title"] for r in rows}


def _load_existing_cases() -> tuple[list[dict], set[str]]:
    if not _OUT_PATH.exists():
        return [], set()
    cases = json.loads(_OUT_PATH.read_text(encoding="utf-8"))
    return cases, {c["url"] for c in cases}


def _load_excluded_urls() -> set[str]:
    if not _EXCLUDED_PATH.exists():
        return set()
    return set(json.loads(_EXCLUDED_PATH.read_text(encoding="utf-8")))


def main() -> None:
    db_titles = _load_db_titles()
    existing_cases, known_urls = _load_existing_cases()
    excluded_urls = _load_excluded_urls()
    skip_urls = known_urls | excluded_urls

    cache_files = sorted(_CACHE_DIR.glob("*.json"))
    if not cache_files:
        console.print("[red]No cached posts found. Run the agent first to populate the cache.[/red]")
        sys.exit(1)

    new_files = [f for f in cache_files
                 if json.loads(f.read_text(encoding="utf-8"))["url"] not in skip_urls]

    if not new_files:
        console.print(f"[dim]All {len(cache_files)} cached posts accounted for ({len(existing_cases)} in cases.json, {len(excluded_urls)} excluded). Nothing to do.[/dim]")
        return

    console.print(f"{len(new_files)} new cached posts to classify ({len(existing_cases)} already in cases.json)...")

    new_cases = []
    for f in track(new_files, description="Classifying"):
        entry = json.loads(f.read_text(encoding="utf-8"))
        url = entry["url"]
        content = entry["offer_section"]
        title = entry.get("title") or db_titles.get(url) or _title_from_url(url)

        try:
            result = classify(title, content)
            new_cases.append({
                "title": title,
                "url": url,
                "content": content,
                "expected": result.label,
                "estimated_value": result.estimated_value,
                "model_reasoning": result.reasoning,
                "human_reasoning": None,
            })
        except Exception as e:
            console.print(f"[yellow]Skipping {url}: {e}[/yellow]")

    all_cases = existing_cases + new_cases
    _OUT_PATH.write_text(json.dumps(all_cases, indent=2), encoding="utf-8")
    console.print(f"\n[green]Added {len(new_cases)} new cases. Total: {len(all_cases)}[/green]")

    label_counts: dict[str, int] = {}
    for c in new_cases:
        label_counts[c["expected"]] = label_counts.get(c["expected"], 0) + 1
    console.print("New cases by label:")
    for label, count in sorted(label_counts.items()):
        console.print(f"  {label}: {count}")


if __name__ == "__main__":
    main()
