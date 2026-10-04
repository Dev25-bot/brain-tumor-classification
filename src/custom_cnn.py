# src/custom_cnn.py
"""Custom CNN built from scratch for 4-class brain MRI classification."""
import torch.nn as nn
import torch.optim as optim


def conv_block(in_ch, out_ch):
    """Two 3x3 convolutions with batch norm + ReLU, followed by 2x2 max pooling."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class CustomCNN(nn.Module):
    """Four conv blocks (32-64-128-256 filters) -> global average pooling -> dense head.

    Global average pooling replaces a huge flatten layer, which keeps the model
    small (~1.2M parameters, ~5 MB) and reduces overfitting. Dropout and batch
    normalisation further regularise and stabilise training.
    """

    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 32),     # 224 -> 112
            conv_block(32, 64),    # 112 -> 56
            conv_block(64, 128),   # 56  -> 28
            conv_block(128, 256),  # 28  -> 14
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


if __name__ == "__main__":
    import argparse
    from data_preprocessing import get_loaders, get_device
    from training import fit

    parser = argparse.ArgumentParser(description="Train the custom CNN")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=7)
    args = parser.parse_args()

    train_loader, val_loader, _, classes = get_loaders()
    device = get_device()
    print(f"Training custom_cnn on {device}")
    model = CustomCNN(len(classes))
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    fit(model, train_loader, val_loader, optimizer, "custom_cnn",
        epochs=args.epochs, device=device, patience=args.patience)
