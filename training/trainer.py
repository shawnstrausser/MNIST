"""
trainer.py — The training loop (a.k.a. "the workout").

Each call to train_one_epoch does one full pass through all 60,000 training
images. For every batch of images it:

  1. Feeds them through the model to get predictions
  2. Measures how wrong those predictions are (loss)
  3. Calculates which direction to nudge the weights (backpropagation)
  4. Nudges the weights a small step in that direction (optimizer.step)

Returns:
  (avg_loss, accuracy) for the epoch — so we can watch the model improve.
"""

import torch
import torch.nn as nn


def train_one_epoch(model, train_loader, optimizer, device):
    """Train the model for one full pass through the training data."""
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()       # reset gradients from last batch
        outputs = model(images)     # forward pass: get predictions
        loss = loss_fn(outputs, labels)  # how wrong were we?
        loss.backward()             # compute gradients
        optimizer.step()            # update weights

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total
