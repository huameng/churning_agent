# Agent trajectory eval cases

Each `*.json` file here is one scenario for `churning_agent.eval.eval_agents`.
A case runs **one real agent turn** (real model, real instruction) with every
tool replaced by a stub that returns canned data from this file and records
the call. Checks then assert on the trajectory and the final response.

Run:

```
uv run python -m churning_agent.eval.eval_agents              # all cases
uv run python -m churning_agent.eval.eval_agents --agent doc_agent
uv run python -m churning_agent.eval.eval_agents --case fanout
uv run python -m churning_agent.eval.eval_agents --list
```

Results land in `eval/results/<timestamp>_agents.json` (full trajectory and
response per case, so failures can be diagnosed after the fact).

## Case format

```jsonc
{
  "name": "doc_finds_moneymaker",          // defaults to the filename
  "agent": "doc_agent",                    // doc_agent | churning_agent (orchestrator) |
                                           // topcashback_agent | swagbucks_agent |
                                           // portal_agent | improvement_agent
  "description": "what behaviour this protects",
  "user_message": "Find new MONEYMAKERS.",

  // Canned tool responses. A tool not listed here returns "ok".
  "tool_responses": {
    "get_last_seen_url": "https://...",            // single value: returned every call
    "fetch_posts": [[{...}, {...}]],               // list: consumed in order, last repeats
    "fetch_and_classify": {                        // keyed: pick by an argument's value
      "key_arg": "url",
      "cases": {"https://...": {...}},             // values may themselves be lists
      "default": {...}
    }
  },

  // All checks must pass for the case to pass.
  "expect": {
    "tools_called": ["fetch_posts"],               // each appears >= once
    "tools_not_called": ["click_element"],         // never appears
    "call_counts": {"fetch_and_classify": [2, 2]}, // [min, max] inclusive
    "tool_args": [                                 // some call of the tool has matching args
      {"tool": "note_offer", "args_contain": {"offer_key": "sb-chime-123"}}
    ],                                             // match = equality or case-insensitive substring
    "response_contains": ["MONEYMAKERS"],          // case-insensitive substring of final response
    "response_not_contains": ["error"]
  }
}
```

## Guidance

- **One behaviour per case.** Name the case after the behaviour it protects,
  and say in `description` why it matters.
- **Add a case whenever you see a bad trajectory in production.** Copy the
  real tool outputs that triggered it into `tool_responses`, write the checks
  for what *should* have happened, and you have a permanent regression test.
- **Keep checks loose where the prompt allows freedom.** Don't assert call
  order or exact wording unless the agent's instruction mandates it —
  brittle checks get deleted instead of maintained. `tools_not_called` is
  often the highest-value check (e.g. portal agents must never click through
  an offer).
- Canned data should mimic the real tool's return shape (see the tool's
  docstring) so the agent behaves as it would live.
- Agents are non-deterministic; a flaky case usually means the *prompt* is
  ambiguous — tighten the prompt or loosen the check, deliberately.
