# src/model_registry.py
"""One place to build and load every trained model (custom CNN + transfer models)."""
from pathlib import Path

import torch

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
MODEL_NAMES = ["custom_cnn", "efficientnet", "mobilenet", "resnet50"]
DISPLAY_NAMES = {
    "custom_cnn": "Custom CNN",
    "efficientnet": "EfficientNet-B0",
    "mobilenet": "MobileNetV2",
    "resnet50": "ResNet50",
}


def build_model(name, num_classes=len(CLASS_NAMES)):
    """Create an untrained model with the architecture used for `name`."""
    if name == "custom_cnn":
        from custom_cnn import CustomCNN
        return CustomCNN(num_classes)
    from transfer_learning import get_transfer_model
    return get_transfer_model(name, num_classes, pretrained=False)


def available_models():
    """Names of the models that have a trained checkpoint in models/."""
    return [n for n in MODEL_NAMES if (MODELS_DIR / f"{n}.pth").exists()]


def load_trained_model(name, device="cpu", num_classes=len(CLASS_NAMES)):
    model = build_model(name, num_classes)
    state = torch.load(MODELS_DIR / f"{name}.pth", map_location=device)
    model.load_state_dict(state)
    return model.to(device).eval()
