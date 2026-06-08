from dotenv import load_dotenv

from churning_agent._paths import PROJECT_ROOT

load_dotenv(PROJECT_ROOT / ".env")

from . import agent
