"""
train.py — Main entry point. Ties everything together.

Run this file to train a model end-to-end:
  python train.py

It reads settings from config.py, loads data, builds the model, trains it
for the configured number of epochs, evaluates after each epoch, and saves
the final trained weights to experiments/<model_name>.pt.
"""

import json
import time
from datetime import datetime

import matplotlib.pyplot as plt
import torch

from config import DEVICE, EPOCHS, LEARNING_RATE, MODEL_NAME, EXPERIMENTS_DIR, BATCH_SIZE
from models.registry import get_model
from training.trainer import train_one_epoch
from evaluation.evaluate import evaluate
from utils.data import get_data_loaders


def main():
    # print out the config details:
    print(f"Model: {MODEL_NAME} | Device: {DEVICE} | Epochs: {EPOCHS}")
    print("-" * 50)

    train_loader, test_loader = get_data_loaders()
    model = get_model(MODEL_NAME).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Count model parameters
    total_params = sum(p.numel() for p in model.parameters())

    epoch_log = []
    total_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, DEVICE)
        test_loss, test_acc = evaluate(model, test_loader, DEVICE, show_progress=True)
        epoch_time = time.time() - epoch_start

        print(f"Epoch {epoch}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Test Loss: {test_loss:.4f} Acc: {test_acc:.4f} | "
              f"Time: {epoch_time:.1f}s")

        epoch_log.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "test_loss": round(test_loss, 4),
            "test_acc": round(test_acc, 4),
            "epoch_time_sec": round(epoch_time, 2),
        })

    total_time = time.time() - total_start

    # Save model
    EXPERIMENTS_DIR.mkdir(exist_ok=True)
    save_path = EXPERIMENTS_DIR / f"{MODEL_NAME}.pt"
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to {save_path}")

    # Save training metadata
    metadata = {
        "model_name": MODEL_NAME,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "device": DEVICE,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "optimizer": "Adam",
        },
        "model_params": total_params,
        "total_training_time_sec": round(total_time, 2),
        "avg_epoch_time_sec": round(total_time / EPOCHS, 2),
        "final_train_loss": epoch_log[-1]["train_loss"],
        "final_train_acc": epoch_log[-1]["train_acc"],
        "final_test_loss": epoch_log[-1]["test_loss"],
        "final_test_acc": epoch_log[-1]["test_acc"],
        "epochs": epoch_log,
    }

    meta_path = EXPERIMENTS_DIR / f"{MODEL_NAME}_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {meta_path}")

    # Plot loss curves
    epochs_range = [e["epoch"] for e in epoch_log]
    train_losses = [e["train_loss"] for e in epoch_log]
    test_losses = [e["test_loss"] for e in epoch_log]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs_range, train_losses, "o-", label="Train Loss")
    ax.plot(epochs_range, test_losses, "o-", label="Test Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Loss Curve — {MODEL_NAME}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plot_path = EXPERIMENTS_DIR / f"{MODEL_NAME}_loss_curve.png"
    fig.savefig(plot_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Loss curve saved to {plot_path}")


if __name__ == "__main__":
    main()
