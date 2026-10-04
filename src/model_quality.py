from collections import defaultdict
from pathlib import Path
import random

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Subset
from torchvision import datasets


DATA_ROOT = Path(__file__).resolve().parent.parent / "data"


def build_balanced_test_subset(transform, samples_per_class=8, seed=42):
    test_dir = DATA_ROOT / "Testing"
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Testing data folder not found: {test_dir}")

    dataset = datasets.ImageFolder(test_dir, transform=transform)
    by_class = defaultdict(list)
    for index, (_, label) in enumerate(dataset.samples):
        by_class[label].append(index)

    rng = random.Random(seed)
    selected = []
    for label in range(len(dataset.classes)):
        indices = by_class[label]
        rng.shuffle(indices)
        selected.extend(indices[:samples_per_class])

    selected.sort()
    return Subset(dataset, selected), dataset.classes


def evaluate_model_quality(model, transform, samples_per_class=8, batch_size=4):
    subset, classes = build_balanced_test_subset(transform, samples_per_class=samples_per_class)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)

    predictions = []
    targets = []
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images)
            predictions.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            targets.extend(labels.cpu().numpy().tolist())

    accuracy = float(np.mean(np.array(predictions) == np.array(targets))) if targets else 0.0
    report = classification_report(
        targets,
        predictions,
        labels=list(range(len(classes))),
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(targets, predictions, labels=list(range(len(classes))))

    return {
        "accuracy": accuracy,
        "classes": classes,
        "confusion_matrix": matrix,
        "report": report,
        "sample_count": len(targets),
    }
