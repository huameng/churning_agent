from google.adk.agents import LlmAgent
from google.adk.tools.agent_tool import AgentTool

from .tools.classifier import fetch_and_classify
from .tools.notify import notify_moneymaker
from .tools.scraper import fetch_posts
from .tools.state import get_last_seen_url, set_last_seen_url
from .tools.store import query_classifications
from .tools.profile import update_profile
from .portal import swagbucks_agent, topcashback_agent
from .improvement_agent import improvement_agent

_MODEL = "gemini-3-flash-preview"

# ── DoC source agent ──────────────────────────────────────────────────────────
# Scans Doctor of Credit for MONEYMAKERS. Invoked by the orchestrator as a tool,
# so it returns its findings as text (and surfaces UNCERTAINs as questions)
# rather than pausing for the human itself.
doc_agent = LlmAgent(
    model=_MODEL,
    name="doc_agent",
    description="Finds new MONEYMAKER opportunities on Doctor of Credit.",
    instruction="""You find MONEYMAKERS on Doctor of Credit (doctorofcredit.com).

Workflow when asked for MONEYMAKERS:
1. Call get_last_seen_url to find where you left off.
2. Call fetch_posts(days_back=1) for today's posts.
3. Work newest to oldest; stop at the last seen URL (if none, process today's posts only).
4. For each new post call fetch_and_classify(title, url). Don't skip; don't summarize early.
5. Collect every MONEYMAKER. For an UNCERTAIN result, collect the post + its `question` (do NOT pause — you are invoked by an orchestrator).
6. After all posts, call set_last_seen_url with the newest post URL.

You are invoked by an orchestrator, not the human directly. Do NOT wait for human input. Return:
MONEYMAKERS:
- <title> | ~$<value> | <url> | <one-line why>
(write 'none' if no new MONEYMAKERS)
QUESTIONS:
- <the specific question for any UNCERTAIN post worth resolving>
(omit if none)

You also have query_classifications (read-only SELECT over the classifications DB:
classifications(id, url, title, label, reasoning, estimated_value, classified_at)) for questions about past decisions.
""",
    tools=[
        fetch_posts,
        fetch_and_classify,
        get_last_seen_url,
        set_last_seen_url,
        notify_moneymaker,
        query_classifications,
    ],
)


# ── Orchestrator (root) ───────────────────────────────────────────────────────
# Fans out to the source agents, aggregates their MONEYMAKERS, reports to the
# human, and handles the human-facing parts (questions, profile updates).
root_agent = LlmAgent(
    model=_MODEL,
    name="churning_agent",
    description="Orchestrates money-making opportunity discovery across Doctor of Credit, TopCashback, and Swagbucks.",
    instruction="""You are the orchestrator. You find money-making opportunities for the user by delegating to source agents and then handling everything human-facing.

You have four subagents as tools:
- doc_agent — Doctor of Credit posts
- topcashback_agent — TopCashback cashback rates
- swagbucks_agent — Swagbucks paid offers
- improvement_agent — runs classifier evals and improves the classifier prompt

When the user asks to "show me MONEYMAKERS" (or similar, without naming a source), fan out to ALL THREE IN PARALLEL: issue the doc_agent, topcashback_agent, and swagbucks_agent tool calls together in the same turn so they run concurrently (each portal agent drives its own browser tab, so this is safe). If the user names a source ("check swagbucks"), call just that one.

After collecting results:
1. Present a single unified report of MONEYMAKERS, grouped by source and ranked by estimated USD value within each. Include reward and a one-line why.
2. Gather every QUESTION the agents returned. Present them to the user clearly and stop to let them answer. When the user answers, call update_profile to record the fact(s) (prefer a specific field over additional_context), then re-invoke the relevant agent so it can finish classifying with the new info.
3. If the user tells you something about themselves at any time, call update_profile immediately.

When the user asks to improve the classifier or run evals, delegate to improvement_agent.

Be concise and concrete. Don't re-run an agent you already ran this turn unless the user asks or you've recorded new profile info that affects it.
""",
    tools=[
        AgentTool(agent=doc_agent),
        AgentTool(agent=topcashback_agent),
        AgentTool(agent=swagbucks_agent),
        AgentTool(agent=improvement_agent),
        update_profile,
    ],
)
