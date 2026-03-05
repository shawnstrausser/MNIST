"""
Measures how well a trained model performs on 
evaluate.py — Measures how well a trained model performs on unseen data.

After training a model, we need to evaluate its performance.

This module tests against images the model has NEVER seen before, giving us
an honest measure of how well it learned.

Returns two numbers:
  - loss:     How "wrong" the model's predictions are (lower is better)
  - accuracy: What percentage of digits it guessed correctly (higher is better)
"""

import sys

import torch
import torch.nn as nn


def evaluate(model, test_loader, device, show_progress=False):
    """
    Run the model on test data and measure its performance.

    Args:
        model:       The trained neural network we want to test.
        test_loader: Feeds us batches of test images + their correct labels.
        device:      Where to run the math — "cpu" or "cuda" (GPU).

    Returns:
        (loss, accuracy) — e.g. (0.08, 0.97) means low error, 97% correct.
    """

    # Tell PyTorch: "we're just testing, not training."
    # This turns off behaviors (like dropout) that only matter during training.
    model.eval()

    # CrossEntropyLoss measures how far off the model's guesses are from the
    # correct answers. It's the same scoring method we used during training.
    loss_fn = nn.CrossEntropyLoss()

    total_loss = 0  # running sum of errors
    correct = 0     # how many digits we got right
    total = 0       # how many digits we've looked at

    # torch.no_grad() tells PyTorch: "don't bother tracking how to improve —
    # we're just measuring, not learning." This saves memory and speeds things up.
    num_batches = len(test_loader)

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader, 1):
            # Move this batch to the right device (cpu or gpu)
            images, labels = images.to(device), labels.to(device)

            # Ask the model: "What digit do you think each image is?"
            outputs = model(images)

            # Score how wrong it was
            loss = loss_fn(outputs, labels)

            # Accumulate results across all batches
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)

            if show_progress:
                pct = batch_idx / num_batches
                bar_len = 30
                filled = int(bar_len * pct)
                bar = "█" * filled + "░" * (bar_len - filled)
                acc_so_far = correct / total
                sys.stdout.write(f"\r  Eval:     [{bar}] {pct:>6.1%}  loss={loss.item():.4f}  acc={acc_so_far:.4f}")
                sys.stdout.flush()

    if show_progress:
        print()

    # Return averages: total error per image, and fraction correct
    return total_loss / total, correct / total
