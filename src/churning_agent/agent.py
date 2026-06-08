from google.adk.agents import LlmAgent
from google.adk.tools.agent_tool import AgentTool

from . import prompts
from .llm import retrying_model
from .tools.classifier import fetch_and_classify
from .tools.notify import notify_moneymaker
from .tools.scraper import fetch_posts
from .tools.state import get_last_seen_url, set_last_seen_url
from .tools.store import query_classifications
from .tools.profile import update_profile
from .portal import swagbucks_agent, topcashback_agent
from .improvement_agent import improvement_agent

# ── DoC source agent ──────────────────────────────────────────────────────────
# Scans Doctor of Credit for MONEYMAKERS. Invoked by the orchestrator as a tool,
# so it returns its findings as text (and surfaces UNCERTAINs as questions)
# rather than pausing for the human itself. Model + instruction: config/prompts/.
_doc = prompts.load("doc_agent")
doc_agent = LlmAgent(
    model=retrying_model(_doc.model),
    name="doc_agent",
    description="Finds new MONEYMAKER opportunities on Doctor of Credit.",
    instruction=_doc.system,
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
_orch = prompts.load("orchestrator")
root_agent = LlmAgent(
    model=retrying_model(_orch.model),
    name="churning_agent",
    description="Orchestrates money-making opportunity discovery across Doctor of Credit, TopCashback, and Swagbucks.",
    instruction=_orch.system,
    tools=[
        AgentTool(agent=doc_agent),
        AgentTool(agent=topcashback_agent),
        AgentTool(agent=swagbucks_agent),
        AgentTool(agent=improvement_agent),
        update_profile,
    ],
)
