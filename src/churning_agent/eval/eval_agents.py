"""
Agent trajectory evaluation harness.

The classifier evals (eval_classifier, eval_offers) test the LLM *decision*
components. This harness tests the *agents themselves*: does doc_agent follow
its workflow, does the orchestrator route to the right subagents, does a
portal agent assess and record offers without clicking through, does the
improvement agent reject an overfit prompt?

Each case in eval/agent_cases/*.json runs one real agent turn (real model,
stubbed tools): every tool is replaced by a stub that returns a canned
response from the case file and records the call. Declarative checks then
assert on the trajectory (which tools were called, with what args, how many
times) and on the final response text. Case file format: see
eval/agent_cases/README.md.

Run from churning_agent/:
    uv run python -m churning_agent.eval.eval_agents
    uv run python -m churning_agent.eval.eval_agents --agent doc_agent
    uv run python -m churning_agent.eval.eval_agents --case fanout
    uv run python -m churning_agent.eval.eval_agents --list
"""

import argparse
import asyncio
import inspect
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from churning_agent._paths import PROJECT_ROOT

load_dotenv(PROJECT_ROOT / ".env")

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.agent_tool import AgentTool
from google.genai import types

from churning_agent.agent import doc_agent, root_agent
from churning_agent.improvement_agent import improvement_agent
from churning_agent.portal import portal_agent, swagbucks_agent, topcashback_agent

console = Console()

_CASES_DIR = PROJECT_ROOT / "eval" / "agent_cases"
_RESULTS_DIR = PROJECT_ROOT / "eval" / "results"
_TURN_TIMEOUT_S = 300

# Agents evaluable by name. The orchestrator's agent name is "churning_agent".
AGENTS: dict[str, LlmAgent] = {
    a.name: a
    for a in (doc_agent, root_agent, topcashback_agent, swagbucks_agent,
              portal_agent, improvement_agent)
}


# ---------------------------------------------------------------------------
# Tool stubbing
# ---------------------------------------------------------------------------

class CannedResponses:
    """Resolves a tool call to its canned response from the case file.

    A tool's spec in "tool_responses" is one of:
      - a single value           -> returned on every call
      - a list                   -> consumed in order; the last value repeats
      - {"key_arg": "<arg>",
         "cases": {value: spec}, -> picks a sub-spec by the named argument
         "default": spec}           (sub-specs may themselves be values/lists)
    Tools with no entry return "ok".
    """

    def __init__(self, spec: dict):
        self._spec = spec
        self._cursor: dict[tuple, int] = {}

    def next(self, tool: str, args: dict):
        spec = self._spec.get(tool, "ok")
        key = ""
        if isinstance(spec, dict) and "key_arg" in spec:
            key = str(args.get(spec["key_arg"]))
            spec = spec["cases"].get(key, spec.get("default", "ok"))
        if isinstance(spec, list):
            i = self._cursor.get((tool, key), 0)
            self._cursor[(tool, key)] = i + 1
            return spec[min(i, len(spec) - 1)]
        return spec


def _stub_function_tool(fn, responses: CannedResponses, trace: list):
    """A stub with the real tool's name, docstring, and signature (so the
    agent sees the same tool schema) that records calls and returns canned data."""
    sig = inspect.signature(fn)

    def stub(*args, **kwargs):
        bound = sig.bind_partial(*args, **kwargs)
        call_args = {k: v for k, v in bound.arguments.items() if k != "tool_context"}
        trace.append({"tool": fn.__name__, "args": call_args})
        return responses.next(fn.__name__, call_args)

    stub.__name__ = fn.__name__
    stub.__qualname__ = fn.__name__
    stub.__doc__ = fn.__doc__
    stub.__signature__ = sig
    return stub


def _stub_agent_tool(tool: AgentTool, responses: CannedResponses, trace: list):
    """A stub standing in for a subagent (orchestrator cases): same tool name
    and description, returns the canned subagent report."""
    name, description = tool.agent.name, tool.agent.description

    def stub(request: str) -> str:
        trace.append({"tool": name, "args": {"request": request}})
        return responses.next(name, {"request": request})

    stub.__name__ = name
    stub.__qualname__ = name
    stub.__doc__ = description
    return stub


def _stubbed_clone(real: LlmAgent, responses: CannedResponses, trace: list) -> LlmAgent:
    """The real agent (same model, instruction, tool schemas) with every tool
    replaced by a recording stub."""
    tools = [
        _stub_agent_tool(t, responses, trace) if isinstance(t, AgentTool)
        else _stub_function_tool(t, responses, trace)
        for t in real.tools
    ]
    return LlmAgent(
        model=real.model,
        name=real.name,
        description=real.description,
        instruction=real.instruction,
        tools=tools,
    )


# ---------------------------------------------------------------------------
# Running one case
# ---------------------------------------------------------------------------

async def _run_turn(agent: LlmAgent, message: str) -> str:
    service = InMemorySessionService()
    runner = Runner(agent=agent, app_name="agent_eval", session_service=service)
    session = await service.create_session(app_name="agent_eval", user_id="eval")
    parts_out: list[str] = []
    events = runner.run_async(
        user_id="eval",
        session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text=message)]),
    )
    async for event in events:
        if event.is_final_response() and event.content and event.content.parts:
            for part in event.content.parts:
                if getattr(part, "text", None):
                    parts_out.append(part.text)
    return "\n".join(parts_out)


def _arg_matches(actual, expected) -> bool:
    """Loose arg match: equal values pass; otherwise case-insensitive
    substring of the stringified actual (so a URL fragment matches a URL)."""
    if actual is None:
        return False
    if actual == expected:
        return True
    return str(expected).lower() in str(actual).lower()


def run_checks(expect: dict, trace: list, response: str) -> list[dict]:
    """Evaluate every declarative check in `expect` against the recorded
    trajectory and final response. Returns one dict per check."""
    checks: list[dict] = []
    called = [t["tool"] for t in trace]

    def add(description: str, passed: bool, detail: str = ""):
        checks.append({"check": description, "passed": passed, "detail": detail})

    for tool in expect.get("tools_called", []):
        add(f"called {tool}", tool in called)

    for tool in expect.get("tools_not_called", []):
        add(f"did not call {tool}", tool not in called)

    for tool, (lo, hi) in expect.get("call_counts", {}).items():
        n = called.count(tool)
        add(f"{tool} called {lo}-{hi}x", lo <= n <= hi, f"called {n}x")

    for spec in expect.get("tool_args", []):
        tool, want = spec["tool"], spec["args_contain"]
        hits = [
            t for t in trace
            if t["tool"] == tool
            and all(_arg_matches(t["args"].get(k), v) for k, v in want.items())
        ]
        actual = [t["args"] for t in trace if t["tool"] == tool]
        add(f"{tool} called with {want}", bool(hits),
            "" if hits else f"actual calls: {actual}")

    for needle in expect.get("response_contains", []):
        add(f"response contains {needle!r}", needle.lower() in response.lower())

    for needle in expect.get("response_not_contains", []):
        add(f"response does not contain {needle!r}", needle.lower() not in response.lower())

    return checks


async def run_case(case: dict) -> dict:
    """Run one case end to end; never raises (errors become a failed result)."""
    agent_name = case["agent"]
    trace: list[dict] = []
    responses = CannedResponses(case.get("tool_responses", {}))
    agent = _stubbed_clone(AGENTS[agent_name], responses, trace)

    error = None
    response = ""
    try:
        response = await asyncio.wait_for(
            _run_turn(agent, case["user_message"]), timeout=_TURN_TIMEOUT_S
        )
    except Exception as e:  # noqa: BLE001 — one bad case must not kill the run
        error = f"{type(e).__name__}: {e}"

    checks = run_checks(case.get("expect", {}), trace, response)
    passed = all(c["passed"] for c in checks) and error is None
    return {
        "name": case["name"],
        "agent": agent_name,
        "description": case.get("description", ""),
        "passed": passed,
        "error": error,
        "checks": checks,
        "trace": [{"tool": t["tool"], "args": {k: str(v) for k, v in t["args"].items()}}
                  for t in trace],
        "response": response,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_cases(agent_filter: str | None, case_filter: str | None) -> list[dict]:
    cases = []
    for path in sorted(_CASES_DIR.glob("*.json")):
        case = json.loads(path.read_text(encoding="utf-8"))
        case.setdefault("name", path.stem)
        if case["agent"] not in AGENTS:
            console.print(f"[yellow]Skipping {path.name}: unknown agent {case['agent']!r} "
                          f"(known: {', '.join(AGENTS)})[/yellow]")
            continue
        if agent_filter and case["agent"] != agent_filter:
            continue
        if case_filter and case_filter not in case["name"]:
            continue
        cases.append(case)
    return cases


def _write_results(results: list[dict]) -> Path:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_at = datetime.now(timezone.utc).isoformat()
    filename = run_at.replace(":", "").replace("-", "").replace("T", "_")[:15] + "_agents.json"
    path = _RESULTS_DIR / filename
    path.write_text(
        json.dumps({
            "run_at": run_at,
            "passed": sum(1 for r in results if r["passed"]),
            "total": len(results),
            "cases": results,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def _print_report(results: list[dict]) -> None:
    table = Table(title="Agent Trajectory Evaluation", show_lines=True)
    table.add_column("Case", max_width=36)
    table.add_column("Agent", no_wrap=True)
    table.add_column("Checks", no_wrap=True)
    table.add_column("Result", no_wrap=True)

    for r in results:
        ok = sum(1 for c in r["checks"] if c["passed"])
        table.add_row(
            r["name"],
            r["agent"],
            f"{ok}/{len(r['checks'])}",
            "[green]PASS[/green]" if r["passed"] else "[red]FAIL[/red]",
        )
    console.print(table)

    for r in results:
        if r["passed"]:
            continue
        console.print(f"\n[red bold]FAIL[/red bold] {r['name']} — {r['description']}")
        if r["error"]:
            console.print(f"  [red]error:[/red] {r['error']}")
        for c in r["checks"]:
            if not c["passed"]:
                detail = f"  [dim]({c['detail']})[/dim]" if c["detail"] else ""
                console.print(f"  [red]✗[/red] {c['check']}{detail}")
        console.print(f"  [dim]trajectory: {' → '.join(t['tool'] for t in r['trace']) or '(no tool calls)'}[/dim]")
        if r["response"]:
            console.print(f"  [dim]response: {r['response'][:400]}[/dim]")


async def _main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate agent trajectories")
    parser.add_argument("--agent", help="Only run cases for this agent (e.g. doc_agent)")
    parser.add_argument("--case", help="Only run cases whose name contains this substring")
    parser.add_argument("--list", action="store_true", help="List cases and exit")
    args = parser.parse_args()

    cases = _load_cases(args.agent, args.case)
    if not cases:
        console.print(f"[yellow]No cases matched in {_CASES_DIR}[/yellow]")
        return 1

    if args.list:
        for c in cases:
            console.print(f"{c['name']}  [dim]({c['agent']}) — {c.get('description', '')}[/dim]")
        return 0

    results = []
    for case in cases:
        console.print(f"[dim]Running {case['name']} ({case['agent']})...[/dim]")
        results.append(await run_case(case))

    _print_report(results)
    passed = sum(1 for r in results if r["passed"])
    console.print(f"\n[bold]Results:[/bold] {passed}/{len(results)} cases passed")
    path = _write_results(results)
    console.print(f"[dim]Results saved to {path}[/dim]")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
