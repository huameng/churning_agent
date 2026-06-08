"""Stable paths for the churning_agent project root and key data directories.

Import these instead of computing Path(__file__).parent.parent... chains.
    from churning_agent._paths import PROJECT_ROOT, CONFIG_DIR, DATA_DIR
"""
from pathlib import Path

# src/churning_agent/_paths.py  →  parent = src/churning_agent/
#                                  parent.parent = src/
#                                  parent.parent.parent = churning_agent/ (project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
