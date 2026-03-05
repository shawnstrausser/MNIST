from models.simple_fc import SimpleFCNet

MODEL_REGISTRY = {
    "simple_fc": SimpleFCNet,
}


def get_model(name):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name]()
