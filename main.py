"""Interactive churning agent. Usage: uv run python -m churning_agent.main"""
import asyncio

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from .agent import root_agent

_APP = "churning_agent"


def _is_503(exc: BaseException) -> bool:
    return "503" in str(exc) or "UNAVAILABLE" in str(exc)


@retry(
    retry=retry_if_exception(_is_503),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    stop=stop_after_attempt(5),
    reraise=True,
)
async def _send(runner: Runner, session_id: str, text: str) -> None:
    events = runner.run_async(
        user_id="user",
        session_id=session_id,
        new_message=types.Content(role="user", parts=[types.Part(text=text)]),
    )
    async for event in events:
        if event.is_final_response() and event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    print(part.text)


async def _main() -> None:
    session_service = InMemorySessionService()
    runner = Runner(agent=root_agent, app_name=_APP, session_service=session_service)
    session = await session_service.create_session(app_name=_APP, user_id="user")

    print("Churning agent ready (try 'show me MONEYMAKERS'). Ctrl+C or Ctrl+D to exit.\n")
    try:
        while True:
            try:
                text = input("> ").strip()
            except (KeyboardInterrupt, EOFError):
                print()
                break
            if not text:
                continue
            await _send(runner, session.id, text)
            print()
    finally:
        from .tools.browser import close_session
        await close_session()


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
