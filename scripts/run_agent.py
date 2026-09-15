#!/usr/bin/env python3
"""Thin CLI wrapper: python scripts/run_agent.py --message "..."

Equivalent to `python -m src.agent.agent --message "..."`; provided as a
top-level script per the assignment's example usage.

Windows:
    python scripts\\run_agent.py --message "My Amazon order has not arrived yet"

macOS/Linux:
    python scripts/run_agent.py --message "My package is late"
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent.agent import _cli  # noqa: E402

if __name__ == "__main__":
    _cli()
