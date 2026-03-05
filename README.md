# MNIST Digit Classifier

Teach a computer to recognize handwritten digits (0-9) using PyTorch.

## What This Does

You have images of handwritten digits. This project trains a neural network to look at a new image and correctly guess which digit it is.

## Project Flow

When you run `python train.py`, here's what happens:

```
                         ┌─────────────┐
                         │  config.py  │
                         │  (settings) │
                         └──────┬──────┘
                                │
                    Sets: batch size, learning rate,
                    epochs, which model to use
                                │
                 ┌──────────────┴──────────────┐
                 │                              │
          ┌──────▼──────┐              ┌────────▼────────┐
          │ utils/      │              │ models/         │
          │ data.py     │              │ registry.py     │
          │             │              │                 │
          │ Loads 60k   │              │ Looks up model  │
          │ training +  │              │ by name from    │
          │ 10k test    │              │ config          │
          │ images      │              │                 │
          └──────┬──────┘              └────────┬────────┘
                 │                              │
                 │                    ┌─────────▼─────────┐
                 │                    │ models/            │
                 │                    │ simple_fc.py       │
                 │                    │                    │
                 │                    │ The "brain":       │
                 │                    │ 784 → 128 → 64 →  │
                 │                    │ 10 scores          │
                 │                    └─────────┬──────────┘
                 │                              │
                 └──────────────┬───────────────┘
                                │
                         ┌──────▼──────┐
                         │  train.py   │
                         │  (main)     │
                         └──────┬──────┘
                                │
                    For each epoch (1 to 5):
                                │
                 ┌──────────────┴──────────────┐
                 │                              │
        ┌────────▼────────┐            ┌────────▼────────┐
        │ training/       │            │ evaluation/     │
        │ trainer.py      │            │ evaluate.py     │
        │                 │            │                 │
        │ THE WORKOUT     │            │ THE EXAM        │
        │                 │            │                 │
        │ Show images →   │            │ Show NEW images │
        │ Check guesses → │            │ → Count how     │
        │ Measure error → │            │ many it gets    │
        │ Adjust weights  │            │ right           │
        │                 │            │ (no learning)   │
        └────────┬────────┘            └────────┬────────┘
                 │                              │
                 └──────────────┬───────────────┘
                                │
                         ┌──────▼──────┐
                         │ experiments/│
                         │             │
                         │ Saves the   │
                         │ trained     │
                         │ model as    │
                         │ .pt file    │
                         └─────────────┘
```

## Project Structure

```
MNIST/
  config.py              — Control panel: batch size, learning rate, epochs, model selection
  train.py               — Main entry point — ties everything together
  data/MNIST/raw/        — Raw MNIST data files (60k train + 10k test images)
  models/
    simple_fc.py         — Simple 3-layer feedforward network (784 → 128 → 64 → 10)
    registry.py          — Model lookup table (add future models here)
  training/
    trainer.py           — Training loop: feed data, measure error, adjust weights
  evaluation/
    evaluate.py          — Test loop: measure accuracy on unseen data
  experiments/           — Saved model weights (.pt files)
  utils/
    data.py              — Data loading + image transforms
```

## How to Run

```bash
# Install dependencies
pip install torch torchvision

# Train the model
cd Desktop/MNIST
python train.py
```

## Expected Output

```
Model: simple_fc | Device: cpu | Epochs: 5
--------------------------------------------------
Epoch 1/5 | Train Loss: ~0.35 Acc: ~0.90 | Test Loss: ~0.18 Acc: ~0.95
Epoch 2/5 | Train Loss: ~0.15 Acc: ~0.95 | Test Loss: ~0.13 Acc: ~0.96
Epoch 3/5 | Train Loss: ~0.10 Acc: ~0.97 | Test Loss: ~0.10 Acc: ~0.97
Epoch 4/5 | Train Loss: ~0.08 Acc: ~0.98 | Test Loss: ~0.09 Acc: ~0.97
Epoch 5/5 | Train Loss: ~0.07 Acc: ~0.98 | Test Loss: ~0.09 Acc: ~0.97

Model saved to experiments/simple_fc.pt
```

- **Loss** goes down each epoch (model is making fewer mistakes)
- **Accuracy** goes up each epoch (model is guessing more digits correctly)
- Final test accuracy ~97% for the simple model

## Key Concepts

- **Epoch**: One full pass through all 60,000 training images
- **Batch**: A group of 64 images processed at once (faster than one at a time)
- **Loss**: A number measuring how wrong the model is (lower = better)
- **Accuracy**: Percentage of digits guessed correctly (higher = better)
- **Training vs Test**: Train on 60k images, test on a separate 10k the model has never seen — this tells us if it actually *learned* vs just *memorized*
