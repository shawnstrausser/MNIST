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
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

# Training (env vars override for run_all.py pipeline)
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = int(os.environ.get("MNIST_EPOCHS", 5))
DEVICE = "cpu"  # switch to "cuda" if GPU available

# Model
MODEL_NAME = os.environ.get("MNIST_MODEL", "simple_fc")  # options: "simple_fc", "cnn"
