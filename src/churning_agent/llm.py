"""
Shared LLM-call resilience.

A transient transport drop (httpx.ReadError and friends) or a 5xx / UNAVAILABLE
server error should never kill a query — they're retried with exponential
backoff. Two entry points, covering the two ways we call the model:

- retry_transient: a tenacity decorator for direct genai calls (the classifiers
  and the prompt improver).
- retrying_model(model_id): an ADK BaseLlm wrapping Gemini so the agents' own
  model calls (including sub-agents invoked via AgentTool) are retried too.
"""
import asyncio

import httpx
from google.adk.models.google_llm import Gemini
from google.genai import errors as genai_errors
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

_MAX_ATTEMPTS = 5
_BASE_WAIT = 2   # seconds; exponential backoff, capped at _MAX_WAIT
_MAX_WAIT = 60


def is_transient(exc: BaseException) -> bool:
    """A failure worth retrying: a network/transport drop or a server-side 5xx.
    Client errors (bad request, auth, 404 model) are NOT retried."""
    if isinstance(exc, (httpx.TransportError, genai_errors.ServerError)):
        return True
    # Fallback for wrapped/stringified transport + UNAVAILABLE cases.
    s = str(exc)
    return "503" in s or "UNAVAILABLE" in s or "ReadError" in s


retry_transient = retry(
    retry=retry_if_exception(is_transient),
    wait=wait_exponential(multiplier=_BASE_WAIT, min=_BASE_WAIT, max=_MAX_WAIT),
    stop=stop_after_attempt(_MAX_ATTEMPTS),
    reraise=True,
)


class _RetryingGemini(Gemini):
    """Gemini that retries transient transport/server errors — but only while
    nothing has been streamed yet, so a partial response is never double-emitted."""

    async def generate_content_async(self, llm_request, stream=False):
        attempt = 0
        while True:
            attempt += 1
            yielded = False
            try:
                async for resp in super().generate_content_async(llm_request, stream=stream):
                    yielded = True
                    yield resp
                return
            except Exception as e:
                if yielded or attempt >= _MAX_ATTEMPTS or not is_transient(e):
                    raise
                await asyncio.sleep(min(_BASE_WAIT * 2 ** (attempt - 1), _MAX_WAIT))


def retrying_model(model_id: str) -> Gemini:
    """An ADK model for `model_id` that self-heals through transient errors."""
    return _RetryingGemini(model=model_id)
