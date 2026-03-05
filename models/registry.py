"""
registry.py — Model lookup table.

Maps a short name (like "simple_fc") to the actual class that builds the
neural network. This lets config.py choose a model by name without importing
every model file directly.

To add a new model:
  1. Create the class in models/ (e.g., cnn.py)
  2. Import it here
  3. Add an entry to MODEL_REGISTRY
"""

from models.simple_fc import SimpleFCNet
from models.cnn import CNNNet

MODEL_REGISTRY = {
    "simple_fc": SimpleFCNet,
    "cnn": CNNNet,
}


def get_model(name):
    """Look up a model by name, create a fresh instance, and return it."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name]()
