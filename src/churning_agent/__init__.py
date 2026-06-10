import logging
import os

from dotenv import load_dotenv

from churning_agent._paths import PROJECT_ROOT

load_dotenv(PROJECT_ROOT / ".env")

# Package logger: INFO to stderr so subagent activity (offers evaluated and their
# results) is visible under `adk web` and the CLI. Override with CHURNING_LOG_LEVEL.
_logger = logging.getLogger("churning_agent")
if not _logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
    _logger.addHandler(_handler)
_logger.setLevel(os.environ.get("CHURNING_LOG_LEVEL", "INFO").upper())
_logger.propagate = False

from . import agent
