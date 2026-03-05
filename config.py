from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

# Training
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 5
DEVICE = "cpu"  # switch to "cuda" if GPU available

# Model
MODEL_NAME = "simple_fc"  # options: "simple_fc", "cnn", etc.
