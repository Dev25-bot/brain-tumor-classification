# src/training.py
"""Shared training loop with ModelCheckpoint, EarlyStopping and history plots."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def _run_epoch(model, loader, criterion, device, optimizer=None):
    """One pass over `loader`. Trains when an optimizer is given, otherwise evaluates."""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(training):
        for xb, yb in tqdm(loader, leave=False, desc="train" if training else "val"):
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * xb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
            total += yb.size(0)

    return total_loss / total, correct / total


def fit(model, train_loader, val_loader, optimizer, model_name, epochs, device,
        patience=5, history=None, best_val_loss=float("inf")):
    """Train `model` and keep the checkpoint with the lowest validation loss.

    - ModelCheckpoint: saves models/<model_name>.pth whenever val loss improves.
    - EarlyStopping: stops after `patience` epochs without improvement.
    - ReduceLROnPlateau: halves the learning rate when val loss stalls.

    `history` and `best_val_loss` can be passed in to continue a previous stage
    (used for the frozen -> fine-tuning stages of transfer learning).
    Returns (history, best_val_loss).
    """
    MODELS_DIR.mkdir(exist_ok=True)
    checkpoint_path = MODELS_DIR / f"{model_name}.pth"
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
    history = history or {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    epochs_without_improvement = 0

    model.to(device)
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = _run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = _run_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        for key, value in zip(history, (train_loss, train_acc, val_loss, val_acc)):
            history[key].append(value)

        msg = (f"[{model_name}] epoch {len(history['val_loss'])} - "
               f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
               f"val loss {val_loss:.4f} acc {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint_path)
            msg += "  -> saved best model"
        else:
            epochs_without_improvement += 1
        print(msg, flush=True)

        if epochs_without_improvement >= patience:
            print(f"[{model_name}] early stopping: no val-loss improvement "
                  f"for {patience} epochs", flush=True)
            break

    save_history(history, model_name)
    return history, best_val_loss


def save_history(history, model_name):
    """Write the training history as JSON and as accuracy/loss plots."""
    with open(MODELS_DIR / f"{model_name}_history.json", "w") as f:
        json.dump(history, f, indent=2)

    epochs = range(1, len(history["val_loss"]) + 1)
    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(11, 4))
    ax_acc.plot(epochs, history["train_acc"], label="train")
    ax_acc.plot(epochs, history["val_acc"], label="validation")
    ax_acc.set(title=f"{model_name} accuracy", xlabel="epoch", ylabel="accuracy")
    ax_loss.plot(epochs, history["train_loss"], label="train")
    ax_loss.plot(epochs, history["val_loss"], label="validation")
    ax_loss.set(title=f"{model_name} loss", xlabel="epoch", ylabel="loss")
    for ax in (ax_acc, ax_loss):
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(MODELS_DIR / f"{model_name}_history.png", dpi=120)
    plt.close(fig)
