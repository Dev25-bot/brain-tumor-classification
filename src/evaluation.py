# src/evaluation.py
"""Evaluate every trained model on the held-out Testing set and compare them.

Outputs (in models/):
  <model>_cm.png          confusion matrix
  <model>_report.json     per-class precision / recall / F1
  comparison.csv/.json    side-by-side comparison of all models
"""
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from model_registry import MODELS_DIR, DISPLAY_NAMES, available_models, load_trained_model


def evaluate(name, test_loader, classes, device):
    model = load_trained_model(name, device=device, num_classes=len(classes))

    all_pred, all_true = [], []
    start = time.perf_counter()
    with torch.no_grad():
        for xb, yb in tqdm(test_loader, desc=f"Evaluating {name}"):
            preds = model(xb.to(device)).argmax(1).cpu().numpy()
            all_pred.extend(preds)
            all_true.extend(yb.numpy())
    ms_per_image = (time.perf_counter() - start) * 1000 / len(all_true)

    acc = float(np.mean(np.array(all_pred) == np.array(all_true)))
    report = classification_report(all_true, all_pred, target_names=classes,
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(all_true, all_pred)

    with open(MODELS_DIR / f"{name}_report.json", "w") as f:
        json.dump({"accuracy": acc, "report": report, "confusion_matrix": cm.tolist()}, f, indent=2)

    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(f"{DISPLAY_NAMES[name]} confusion matrix (accuracy {acc:.2%})")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(MODELS_DIR / f"{name}_cm.png", dpi=120)
    plt.close()

    params = sum(p.numel() for p in model.parameters())
    size_mb = (MODELS_DIR / f"{name}.pth").stat().st_size / 1e6
    wavg = report["weighted avg"]
    return {
        "model": name,
        "accuracy": acc,
        "precision": wavg["precision"],
        "recall": wavg["recall"],
        "f1": wavg["f1-score"],
        "params_millions": params / 1e6,
        "size_mb": size_mb,
        "ms_per_image": ms_per_image,
    }


if __name__ == "__main__":
    from data_preprocessing import get_loaders, get_device

    _, _, test_loader, classes = get_loaders(batch_size=32)
    device = get_device()

    names = available_models()
    if not names:
        raise SystemExit("No trained model files found in the models directory.")

    rows = [evaluate(n, test_loader, classes, device) for n in names]
    df = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    df.to_csv(MODELS_DIR / "comparison.csv", index=False)
    best = df.iloc[0]["model"]
    with open(MODELS_DIR / "comparison.json", "w") as f:
        json.dump({"best_model": best, "results": df.to_dict(orient="records")}, f, indent=2)

    print("\n=== MODEL COMPARISON (test set) ===")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nBest model -> {best}")
