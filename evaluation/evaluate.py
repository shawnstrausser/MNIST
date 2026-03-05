"""
evaluate.py — Measures how well a trained model performs on unseen data.

After training a model, we need to evaluate its performance.

This module tests against images the model has NEVER seen before, giving us
an honest measure of how well it learned.

Two modes:
  - evaluate()          — fast, returns (loss, accuracy). Used during training loop.
  - evaluate_detailed() — full, returns all labels/probs for comprehensive metrics.
                          Used once after training for the final metrics report.
"""

import sys
import time

import numpy as np
import torch
import torch.nn as nn


def evaluate(model, test_loader, device, show_progress=False):
    """
    Fast evaluation — returns (loss, accuracy). Used every epoch during training.

    Args:
        model:       The trained neural network we want to test.
        test_loader: Feeds us batches of test images + their correct labels.
        device:      Where to run the math — "cpu" or "cuda" (GPU).

    Returns:
        (loss, accuracy) — e.g. (0.08, 0.97) means low error, 97% correct.
    """
    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    total_loss = 0
    correct = 0
    total = 0
    num_batches = len(test_loader)

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader, 1):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)

            if show_progress:
                pct = batch_idx / num_batches
                bar_len = 30
                filled = int(bar_len * pct)
                bar = "#" * filled + "-" * (bar_len - filled)
                acc_so_far = correct / total
                sys.stdout.write(f"\r  Eval:     [{bar}] {pct:>6.1%}  loss={loss.item():.4f}  acc={acc_so_far:.4f}")
                sys.stdout.flush()

    if show_progress:
        print()

    return total_loss / total, correct / total


def evaluate_detailed(model, test_loader, device):
    """
    Full evaluation — collects all predictions, probabilities, and performance stats.

    Used once after training to compute comprehensive metrics (precision, recall,
    AUC, FP/FN, confusion pairs, etc.)

    Args:
        model:       The trained neural network.
        test_loader: Test data batches.
        device:      "cpu" or "cuda".

    Returns:
        dict with keys:
            - true_labels:  numpy array (N,)
            - pred_labels:  numpy array (N,)
            - probabilities: numpy array (N, 10) — softmax probs per class
            - loss: float
            - perf: dict with throughput, inference_time_ms, total_eval_time_sec
    """
    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    all_labels = []
    all_preds = []
    all_probs = []
    total_loss = 0
    total = 0

    eval_start = time.time()

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            # Softmax to get probabilities
            probs = torch.softmax(outputs, dim=1)

            total_loss += loss.item() * images.size(0)
            total += images.size(0)

            all_labels.append(labels.cpu())
            all_preds.append(outputs.argmax(1).cpu())
            all_probs.append(probs.cpu())

    eval_time = time.time() - eval_start

    true_labels = torch.cat(all_labels).numpy()
    pred_labels = torch.cat(all_preds).numpy()
    probabilities = torch.cat(all_probs).numpy()

    return {
        "true_labels": true_labels,
        "pred_labels": pred_labels,
        "probabilities": probabilities,
        "loss": total_loss / total,
        "perf": {
            "total_eval_time_sec": round(eval_time, 3),
            "throughput_images_per_sec": round(total / eval_time, 1),
            "avg_inference_time_ms": round((eval_time / total) * 1000, 4),
            "total_samples": total,
        },
    }
