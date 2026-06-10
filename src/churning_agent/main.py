"""Interactive churning agent. Usage: uv run python -m churning_agent.main"""
import asyncio

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

from .agent import root_agent
from .llm import send_message

_APP = "churning_agent"


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
            await send_message(runner, session.id, text)
            print()
    finally:
        from .tools.browser import close_session
        await close_session()


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
