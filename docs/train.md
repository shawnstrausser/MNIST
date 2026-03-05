# train.py — What Happens When You Run It

## Quick Summary

`python train.py` is the main entry point. It loads settings, builds a model, trains it on 60k images, tests it on 10k images, saves the trained weights, saves training metadata to JSON, and plots a loss curve.

## Step by Step

### 1. Config gets imported

Python reads `config.py` and grabs:
- `DEVICE` (cpu)
- `EPOCHS` (5)
- `BATCH_SIZE` (64)
- `LEARNING_RATE` (0.001)
- `MODEL_NAME` ("simple_fc")
- `EXPERIMENTS_DIR` (experiments/)

### 2. Data gets loaded

`data.py` downloads 70k handwritten digit images (if not already downloaded), transforms them to tensors, normalizes them, and wraps them in two DataLoaders:
- **Train loader:** 60k images, served in batches of 64
- **Test loader:** 10k images, served in batches of 64

### 3. Model gets built

`registry.py` looks up `"simple_fc"` in its dictionary, finds `SimpleFCNet`, and creates a fresh instance — random weights, knows nothing yet.

### 4. Optimizer gets created

An Adam optimizer is set up with the learning rate (0.001) — this is the thing that will actually nudge the weights.

### 5. The training loop runs (5 epochs)

For each epoch, two things happen — both with live progress bars:

```
  Training: [████████████████████░░░░░░░░░░] 68.2%  loss=0.1234  acc=0.9456
  Eval:     [██████████████████████████████] 100.0%  loss=0.0891  acc=0.9752
```

```
TRAINING (trainer.py)                    TESTING (evaluate.py)
─────────────────────                    ────────────────────
For each batch of 64 images:             For each batch of 64 images:
  1. Feed images → model → predictions     1. Feed images → model → predictions
  2. Compare predictions to real labels     2. Compare predictions to real labels
  3. Calculate loss (how wrong)             3. Calculate loss (how wrong)
  4. Backprop (figure out which             4. That's it — NO learning
     weights caused the error)
  5. Optimizer nudges weights
     a tiny bit in the right direction

Result: model gets slightly smarter        Result: honest accuracy score
        each batch                                 (model never saw these)
```

### 6. Progress prints each epoch

```
Epoch 1/5 | Train Loss: 0.2653 Acc: 0.9217 | Test Loss: 0.1264 Acc: 0.9614 | Time: 43.3s
```

Loss goes down, accuracy goes up — the model is learning!

### 7. Model weights get saved

After all 5 epochs, the trained weights are saved to `experiments/simple_fc.pt` — a file containing all the numbers the model learned. You can reload this later without retraining.

### 8. Training metadata gets saved

A JSON file is saved to `experiments/simple_fc_metadata.json` with:
- Config used (device, epochs, batch size, learning rate, optimizer)
- Total parameter count
- Total training time and average time per epoch
- Final train/test loss and accuracy
- Per-epoch breakdown (loss, accuracy, time for each epoch)

### 9. Loss curve gets plotted

A PNG chart is saved to `experiments/simple_fc_loss_curve.png` showing train loss and test loss across all epochs. This makes it easy to spot overfitting (when train loss keeps dropping but test loss starts rising).

## Outputs

After training completes, three files are saved to `experiments/`:

| File | Contents |
|------|----------|
| `simple_fc.pt` | Trained model weights (430 KB) |
| `simple_fc_metadata.json` | Training metadata (config, timing, per-epoch stats) |
| `simple_fc_loss_curve.png` | Loss curve chart (train vs test) |

## Total Time

About 3-4 minutes on CPU (~41 sec/epoch). The model goes from random guessing (~10% accuracy) to ~97% accuracy in 5 passes through the data.