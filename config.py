"""
config.py — Central control panel for the entire project.

Every setting lives here so you can tweak the experiment from one place
instead of hunting through multiple files.

Key settings:
  - BATCH_SIZE:     How many images to process at once (bigger = faster but uses more RAM)
  - LEARNING_RATE:  How aggressively the model adjusts its weights each step
  - EPOCHS:         How many full passes through the training data
  - MODEL_NAME:     Which neural network architecture to use
  - DEVICE:         "cpu" or "cuda" (GPU) — where the math runs
"""

import os
import subprocess
from datetime import datetime
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
RUNS_DIR = EXPERIMENTS_DIR / "runs"

# Training (env vars override for run_all.py pipeline)
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = int(os.environ.get("MNIST_EPOCHS", 5))
DEVICE = "cpu"  # switch to "cuda" if GPU available

# Model
MODEL_NAME = os.environ.get("MNIST_MODEL", "simple_fc")  # options: "simple_fc", "cnn"


def get_git_commit():
    """Get short git commit hash, or 'unknown' if not in a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=PROJECT_ROOT,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except FileNotFoundError:
        return "unknown"


def get_run_dir(model_name=None):
    """Create and return a timestamped run directory.

    Format: experiments/runs/YYYY-MM-DD_HH-MM_modelname/
    """
    model_name = model_name or MODEL_NAME
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    run_name = f"{timestamp}_{model_name}"
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
