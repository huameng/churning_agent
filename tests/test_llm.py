"""
LLM resilience tests: transient transport/server errors are retried; client
errors are not. Covers both the tenacity decorator (direct calls) and the
RetryingGemini wrapper (ADK agent model calls).
"""

import httpx
import pytest
from google.adk.models.google_llm import Gemini

from churning_agent import llm
from churning_agent.llm import _RetryingGemini, is_transient, retry_transient, retrying_model, send_message


async def _noop(*_a, **_k):
    pass


def test_is_transient_classification():
    assert is_transient(httpx.ReadError("dropped"))
    assert is_transient(httpx.ConnectError("refused"))
    assert is_transient(RuntimeError("503 UNAVAILABLE"))
    assert not is_transient(ValueError("bad request"))
    assert not is_transient(KeyError("missing"))


def test_retry_transient_does_not_retry_client_errors():
    calls = {"n": 0}

    @retry_transient
    def boom():
        calls["n"] += 1
        raise ValueError("not retryable")

    with pytest.raises(ValueError):
        boom()
    assert calls["n"] == 1   # raised immediately, no retries


async def test_retrying_gemini_recovers_from_transient(monkeypatch):
    monkeypatch.setattr(llm.asyncio, "sleep", _noop)   # don't actually back off
    attempts = {"n": 0}

    async def fake(self, llm_request, stream=False):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise httpx.ReadError("connection dropped mid-read")
        yield "RESPONSE"

    monkeypatch.setattr(Gemini, "generate_content_async", fake)
    model = retrying_model("gemini-test")
    out = [r async for r in model.generate_content_async("req")]
    assert out == ["RESPONSE"]
    assert attempts["n"] == 2          # retried once, then succeeded


async def test_retrying_gemini_does_not_double_emit(monkeypatch):
    """If a transient error strikes AFTER a partial yield, we must re-raise (not
    replay) so the consumer never sees a duplicated response."""
    monkeypatch.setattr(llm.asyncio, "sleep", _noop)

    async def fake(self, llm_request, stream=False):
        yield "PARTIAL"
        raise httpx.ReadError("drop after first chunk")

    monkeypatch.setattr(Gemini, "generate_content_async", fake)
    model = _RetryingGemini(model="gemini-test")
    got = []
    with pytest.raises(httpx.ReadError):
        async for r in model.generate_content_async("req"):
            got.append(r)
    assert got == ["PARTIAL"]          # emitted once, then surfaced the error


class _FakePart:
    def __init__(self, text):
        self.text = text


class _FakeEvent:
    def __init__(self, text):
        self.content = type("C", (), {"parts": [_FakePart(text)]})()

    def is_final_response(self):
        return True


class _FakeRunner:
    """Records the message it was sent and replays one final event."""
    def __init__(self, reply):
        self._reply = reply
        self.sent = None

    async def run_async(self, *, user_id, session_id, new_message):
        self.sent = new_message.parts[0].text
        yield _FakeEvent(self._reply)


async def test_send_message_prints_final_text(capsys):
    runner = _FakeRunner("here are your moneymakers")
    await send_message(runner, "sess-1", "show me")
    assert runner.sent == "show me"
    assert "here are your moneymakers" in capsys.readouterr().out
