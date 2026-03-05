"""
output_log.py — Centralized run log for tracking what happened and when.

Every time train.py, evaluate, or visualize runs, it prepends a structured
entry to `output.log` in the project root. This gives you a single file
to check what the last command did, what it produced, and whether it succeeded.

The most recent entry is always at the top — newest first.
"""

import json
from datetime import datetime
from pathlib import Path

from config import PROJECT_ROOT

LOG_PATH = PROJECT_ROOT / "output.log"


def log_run(command: str, status: str, summary: dict, details: str = ""):
    """
    Append a structured entry to output.log.

    Args:
        command:  What was run (e.g. "train.py", "evaluate", "visualize")
        status:   "SUCCESS" or "ERROR"
        summary:  Dict of key metrics/outputs (e.g. {"test_acc": 0.974})
        details:  Optional multi-line string with full output or error trace
    """
    entry = {
        "timestamp": datetime.now().isoformat(),
        "command": command,
        "status": status,
        "summary": summary,
    }

    separator = "=" * 70
    lines = [
        separator,
        f"[{entry['timestamp']}] {command} — {status}",
        separator,
        json.dumps(summary, indent=2),
    ]

    if details:
        lines.append("")
        lines.append(details)

    lines.append("")  # trailing newline

    new_entry = "\n".join(lines) + "\n"

    # Prepend: newest entries at the top
    existing = ""
    if LOG_PATH.exists():
        existing = LOG_PATH.read_text(encoding="utf-8")
    LOG_PATH.write_text(new_entry + existing, encoding="utf-8")

    print(f"Run logged to {LOG_PATH}")