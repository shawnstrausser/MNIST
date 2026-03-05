# evaluate.py — What Happens When You Run It

## Quick Summary

`python -m evaluation.evaluate` loads a trained model from `experiments/`, runs it against 10k test images it's never seen, and prints an honest accuracy score. No learning happens — just measuring. Supports an optional progress bar when called from train.py.

## Step by Step

### 1. Config gets imported

Python reads `config.py` and grabs:
- `DEVICE` (cpu)
- `MODEL_NAME` ("simple_fc")
- `EXPERIMENTS_DIR` (experiments/)

### 2. Trained model gets loaded

Looks in `experiments/` for `simple_fc.pt` (or whatever `MODEL_NAME` is set to), loads the saved weights into a fresh model instance. This is the model that was trained by `train.py`.

### 3. Model enters eval mode

```python
model.eval()
```

This tells PyTorch "we're testing, not training." It turns off training-only behaviors like dropout (randomly disabling neurons). During testing, we want the full model every time.

### 4. Test data gets loaded

`data.py` loads the 10k test images — same transform pipeline (ToTensor + Normalize), same batch size of 64. These are images the model has **never seen** during training.

### 5. The test loop runs

```
For each batch of 64 test images:
  1. Feed images → model → predictions
  2. Compare predictions to real labels
  3. Calculate loss (how wrong)
  4. Count correct guesses

  NO backprop. NO weight updates. Just measuring.
```

Key line: `torch.no_grad()` — tells PyTorch "don't track gradients." Since we're not learning, this saves memory and speeds things up.

When called with `show_progress=True` (from train.py), a live progress bar is shown:

```
  Eval:     [██████████████████████████████] 100.0%  loss=0.0891  acc=0.9752
```

### 6. Results print

```
Test Loss: 0.0876 | Test Accuracy: 0.9723
```

- **Loss** — how wrong on average (lower = better)
- **Accuracy** — percentage correct (higher = better)

## Why This Exists Separately

You might wonder: train.py already evaluates after each epoch, why have a standalone evaluate?

- **Quick checks** — test a saved model without retraining (saves minutes)
- **Compare models** — swap `MODEL_NAME` in config and re-evaluate
- **Verify saved weights** — confirm the .pt file loads and works correctly