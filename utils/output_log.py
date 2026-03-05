"""
output_log.py — Centralized run log for tracking what happened and when.

Every time train.py, evaluate, or visualize runs, it appends a structured
entry to `output.log` in the project root. This gives you a single file
to check what the last command did, what it produced, and whether it succeeded.

The log is append-only so you can see history, but the most recent entry
is always at the bottom.
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

    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Run logged to {LOG_PATH}")