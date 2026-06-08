"""Put the workspace dir on sys.path so `import churning_agent...` resolves
when running pytest from inside the churning_agent directory."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
