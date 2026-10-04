# src/transfer_learning.py
"""Transfer learning with ImageNet-pretrained ResNet50 / MobileNetV2 / EfficientNet-B0.

Stage 1 trains only the new classification head on a frozen backbone.
Stage 2 unfreezes the top of the backbone and fine-tunes with a small learning rate.
"""
import torch.nn as nn
import torch.optim as optim
from torchvision import models

TRANSFER_MODELS = ["efficientnet", "mobilenet", "resnet50"]


def get_transfer_model(name, num_classes, device='cpu', pretrained=True):
    """Pretrained backbone with its final layer replaced by a new `num_classes` head."""
    weights_map = {
        "resnet50": models.ResNet50_Weights.IMAGENET1K_V2,
        "mobilenet": models.MobileNet_V2_Weights.IMAGENET1K_V2,
        "efficientnet": models.EfficientNet_B0_Weights.IMAGENET1K_V1,
    }
    model_fn = {
        "resnet50": models.resnet50,
        "mobilenet": models.mobilenet_v2,
        "efficientnet": models.efficientnet_b0,
    }
    if name not in model_fn:
        raise ValueError(f"Unknown transfer model: {name}")

    weights = weights_map[name] if pretrained else None
    base = model_fn[name](weights=weights)

    for p in base.parameters():
        p.requires_grad = False

    if hasattr(base, "fc"):
        in_features = base.fc.in_features
        base.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, num_classes))
    else:
        in_features = base.classifier[-1].in_features
        base.classifier[-1] = nn.Linear(in_features, num_classes)

    return base.to(device)


def unfreeze_top_layers(model, name):
    """Unfreeze the last stage(s) of the backbone for fine-tuning."""
    if name == "resnet50":
        top = [model.layer4]
    else:  # mobilenet / efficientnet keep their conv stages in `features`
        top = list(model.features)[-3:]
    for module in top:
        for p in module.parameters():
            p.requires_grad = True


def train_transfer_model(name, train_loader, val_loader, num_classes, device,
                         epochs_frozen=5, epochs_finetune=15, patience=5):
    from training import fit

    model = get_transfer_model(name, num_classes, device=device)

    print(f"[{name}] stage 1: training new head on frozen backbone")
    head_params = [p for p in model.parameters() if p.requires_grad]
    history, best = fit(model, train_loader, val_loader, optim.AdamW(head_params, lr=1e-3),
                        name, epochs=epochs_frozen, device=device, patience=patience)

    print(f"[{name}] stage 2: fine-tuning top backbone layers")
    unfreeze_top_layers(model, name)
    trainable = [p for p in model.parameters() if p.requires_grad]
    fit(model, train_loader, val_loader, optim.AdamW(trainable, lr=1e-4, weight_decay=1e-4),
        name, epochs=epochs_finetune, device=device, patience=patience,
        history=history, best_val_loss=best)


if __name__ == "__main__":
    import argparse
    from data_preprocessing import get_loaders, get_device

    parser = argparse.ArgumentParser(description="Train transfer-learning models")
    parser.add_argument("models", nargs="*", default=TRANSFER_MODELS, choices=TRANSFER_MODELS)
    parser.add_argument("--epochs-frozen", type=int, default=5)
    parser.add_argument("--epochs-finetune", type=int, default=15)
    args = parser.parse_args()

    train_loader, val_loader, _, classes = get_loaders()
    device = get_device()
    for n in args.models:
        print(f"\n=== Training {n.upper()} on {device} ===")
        train_transfer_model(n, train_loader, val_loader, len(classes), device,
                             args.epochs_frozen, args.epochs_finetune)
