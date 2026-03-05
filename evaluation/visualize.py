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

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from config import DEVICE, MODEL_NAME, EXPERIMENTS_DIR
from models.registry import get_model
from utils.data import get_data_loaders


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
    cm = confusion_matrix(true_labels, pred_labels)
    disp = ConfusionMatrixDisplay(cm, display_labels=range(10))

    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
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
    plot_confusion_matrix(true_labels, pred_labels, model_name)
    plot_sample_predictions(images, true_labels, pred_labels, model_name)

    print("Done!")


if __name__ == "__main__":
    main()
