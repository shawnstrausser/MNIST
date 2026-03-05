"""
visualize.py — See what the model actually learned.

Generates two visualizations after training:

  1. Confusion Matrix — A 10x10 grid showing which digits get mixed up.
     Perfect model = bright diagonal, everything else dark.

  2. Sample Predictions — A grid of test images with the model's guess.
     Correct predictions in green, wrong ones in red.

Usage:
  python -m evaluation.visualize               # uses config defaults
  python -m evaluation.visualize --model cnn   # specify model
"""

import argparse
import os
import traceback

# Fix OpenMP duplicate library crash (PyTorch + matplotlib both bundle libiomp5md.dll)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from config import DEVICE, MODEL_NAME, EXPERIMENTS_DIR
from models.registry import get_model
from utils.data import get_data_loaders
from utils.output_log import log_run


def load_trained_model(model_name, device):
    """Load a trained model from experiments/."""
    model = get_model(model_name).to(device)
    weights_path = EXPERIMENTS_DIR / f"{model_name}.pt"
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model


def get_all_predictions(model, test_loader, device):
    """Run the model on all test data, return (true_labels, predicted_labels, images)."""
    all_labels = []
    all_preds = []
    all_images = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(1)

            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())
            all_images.append(images.cpu())

    return (
        torch.cat(all_labels).numpy(),
        torch.cat(all_preds).numpy(),
        torch.cat(all_images),
    )


def plot_confusion_matrix(true_labels, pred_labels, model_name):
    """Show a confusion matrix heatmap and save to experiments/."""
    sns.set_theme(style="white")
    cm = confusion_matrix(true_labels, pred_labels)

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(10),
                yticklabels=range(10), ax=ax, cbar=False, linewidths=0.5)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {model_name}")

    save_path = EXPERIMENTS_DIR / f"{model_name}_confusion_matrix.png"
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    print(f"Saved confusion matrix to {save_path}")
    plt.close(fig)


def plot_sample_predictions(images, true_labels, pred_labels, model_name, n=25):
    """Show a grid of sample predictions (green=correct, red=wrong)."""
    cols = 5
    rows = n // cols

    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    fig.suptitle(f"Sample Predictions — {model_name}", fontsize=14)

    # Pick random indices
    rng = np.random.default_rng(42)
    indices = rng.choice(len(images), size=n, replace=False)

    for i, idx in enumerate(indices):
        ax = axes[i // cols][i % cols]
        img = images[idx].squeeze().numpy()
        true = true_labels[idx]
        pred = pred_labels[idx]

        ax.imshow(img, cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])

        color = "green" if true == pred else "red"
        ax.set_title(f"T:{true} P:{pred}", color=color, fontsize=10)

    save_path = EXPERIMENTS_DIR / f"{model_name}_samples.png"
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    print(f"Saved sample predictions to {save_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_NAME, help="Model name to visualize")
    args = parser.parse_args()

    model_name = args.model
    print(f"Generating visualizations for: {model_name}")

    model = load_trained_model(model_name, DEVICE)
    _, test_loader = get_data_loaders()
    true_labels, pred_labels, images = get_all_predictions(model, test_loader, DEVICE)

    accuracy = (true_labels == pred_labels).mean()
    print(f"Test accuracy: {accuracy:.4f}")

    EXPERIMENTS_DIR.mkdir(exist_ok=True)

    cm_path = EXPERIMENTS_DIR / f"{model_name}_confusion_matrix.png"
    samples_path = EXPERIMENTS_DIR / f"{model_name}_samples.png"

    plot_confusion_matrix(true_labels, pred_labels, model_name)
    plot_sample_predictions(images, true_labels, pred_labels, model_name)

    log_run(
        command=f"python -m evaluation.visualize (model={model_name})",
        status="SUCCESS",
        summary={
            "model": model_name,
            "test_accuracy": round(float(accuracy), 4),
            "confusion_matrix_saved": str(cm_path),
            "sample_predictions_saved": str(samples_path),
        },
    )

    print("Done!")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log_run(
            command="python -m evaluation.visualize",
            status="ERROR",
            summary={"error": traceback.format_exc().splitlines()[-1]},
            details=traceback.format_exc(),
        )
        raise
