"""
train.py — Main entry point. Ties everything together.

Run this file to train a model end-to-end:
  python train.py

It reads settings from config.py, loads data, builds the model, trains it
for the configured number of epochs, evaluates after each epoch, and saves
the final trained weights to experiments/<model_name>.pt.
"""

import json
import os
import time
import traceback
from datetime import datetime

# Fix OpenMP duplicate library crash (PyTorch + matplotlib both bundle libiomp5md.dll)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch

from config import DEVICE, EPOCHS, LEARNING_RATE, MODEL_NAME, EXPERIMENTS_DIR, BATCH_SIZE
from models.registry import get_model
from training.trainer import train_one_epoch
from evaluation.evaluate import evaluate
from utils.data import get_data_loaders
from utils.output_log import log_run


def print_architecture(model, model_name):
    """Print a clean ASCII diagram of the model architecture to the terminal."""
    print(f"\n{'=' * 50}")
    print(f"  Architecture: {model_name}")
    print(f"{'=' * 50}")

    layers = []
    for name, module in model.named_modules():
        if name == "":
            continue
        # Skip container modules (Sequential, etc.)
        if len(list(module.children())) > 0:
            continue

        params = sum(p.numel() for p in module.parameters())
        layer_type = module.__class__.__name__

        if hasattr(module, "in_features"):
            shape = f"{module.in_features} -> {module.out_features}"
        elif hasattr(module, "in_channels"):
            shape = f"{module.in_channels}ch -> {module.out_channels}ch, {module.kernel_size}"
        elif isinstance(module, torch.nn.MaxPool2d):
            shape = f"kernel={module.kernel_size}"
        else:
            shape = ""

        layers.append((layer_type, shape, params))

    total_params = sum(p.numel() for p in model.parameters())

    for i, (ltype, shape, params) in enumerate(layers):
        prefix = "  ├─" if i < len(layers) - 1 else "  └─"
        param_str = f"  ({params:,} params)" if params > 0 else ""
        shape_str = f"  [{shape}]" if shape else ""
        print(f"{prefix} {ltype}{shape_str}{param_str}")

    print(f"  {'─' * 46}")
    print(f"  Total parameters: {total_params:,}")
    print(f"{'=' * 50}\n")


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

    # Show architecture
    print_architecture(model, MODEL_NAME)

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

    # Plot training curves (optional — requires seaborn)
    try:
        import seaborn as sns
        import matplotlib.pyplot as plt

        sns.set_theme(style="darkgrid")

        epochs_range = [e["epoch"] for e in epoch_log]
        train_losses = [e["train_loss"] for e in epoch_log]
        test_losses = [e["test_loss"] for e in epoch_log]
        train_accs = [e["train_acc"] for e in epoch_log]
        test_accs = [e["test_acc"] for e in epoch_log]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Loss curve
        sns.lineplot(x=epochs_range, y=train_losses, marker="o", label="Train Loss", ax=ax1)
        sns.lineplot(x=epochs_range, y=test_losses, marker="o", label="Test Loss", ax=ax1)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title(f"Loss — {MODEL_NAME}")

        # Accuracy curve
        sns.lineplot(x=epochs_range, y=train_accs, marker="o", label="Train Acc", ax=ax2)
        sns.lineplot(x=epochs_range, y=test_accs, marker="o", label="Test Acc", ax=ax2)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_title(f"Accuracy — {MODEL_NAME}")
        ax2.set_ylim(0.9, 1.0)

        fig.suptitle(f"Training Curves — {MODEL_NAME}", fontsize=14, y=1.02)
        fig.tight_layout()

        plot_path = EXPERIMENTS_DIR / f"{MODEL_NAME}_training_curves.png"
        fig.savefig(plot_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"Training curves saved to {plot_path}")
    except ImportError:
        print("Skipping training curves plot (seaborn not installed)")

    # Build details string for output log
    detail_lines = []
    for e in epoch_log:
        detail_lines.append(
            f"  Epoch {e['epoch']}: train_loss={e['train_loss']:.4f} "
            f"train_acc={e['train_acc']:.4f} | "
            f"test_loss={e['test_loss']:.4f} test_acc={e['test_acc']:.4f} "
            f"({e['epoch_time_sec']:.1f}s)"
        )

    log_run(
        command=f"python train.py (model={MODEL_NAME})",
        status="SUCCESS",
        summary={
            "model": MODEL_NAME,
            "epochs": EPOCHS,
            "final_test_acc": epoch_log[-1]["test_acc"],
            "final_test_loss": epoch_log[-1]["test_loss"],
            "total_time_sec": round(total_time, 2),
            "model_saved": str(save_path),
            "metadata_saved": str(meta_path),
        },
        details="\n".join(detail_lines),
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log_run(
            command=f"python train.py (model={MODEL_NAME})",
            status="ERROR",
            summary={"error": traceback.format_exc().splitlines()[-1]},
            details=traceback.format_exc(),
        )
        raise
