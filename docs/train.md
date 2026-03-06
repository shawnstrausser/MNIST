# train.py — What Happens When You Run It

## Quick Summary

`python train.py` is the main entry point. It loads settings, builds a model, trains it on 60k images, tests it on 10k images, saves the trained weights, prints an ASCII architecture diagram, saves training metadata to JSON, plots training curves (loss + accuracy), and logs the run to `output.log`.

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

### 9. Architecture diagram prints to terminal

An ASCII diagram of the model architecture is printed, showing each layer, its shape, and parameter count:

```
==================================================
  Architecture: simple_fc
==================================================
  ├─ Flatten
  ├─ Linear  [784 -> 128]  (100,480 params)
  ├─ ReLU
  ├─ Linear  [128 -> 64]  (8,256 params)
  ├─ ReLU
  └─ Linear  [64 -> 10]  (650 params)
  ──────────────────────────────────────────────
  Total parameters: 109,386
==================================================
```

### 10. Training curves get plotted

A PNG chart is saved to `experiments/simple_fc_training_curves.png` with two side-by-side plots:
- **Left:** Train loss vs test loss across all epochs (spot overfitting here)
- **Right:** Train accuracy vs test accuracy across all epochs

Uses seaborn for clean styling. If seaborn isn't installed, this step is skipped gracefully.

### 11. Run gets logged to output.log

A structured entry is appended to `output.log` in the project root with the timestamp, config, final metrics, and per-epoch breakdown. On error, the full traceback is logged instead.

### 12. Detailed metrics get computed

After training finishes, `evaluate_detailed()` runs the test set one more time, collecting softmax probabilities for every sample. These are fed into `metrics.py` which computes 26 metrics: precision, recall, F1 (macro), MCC, Cohen's Kappa, balanced accuracy, log loss, AUC-ROC, AUC-PR, top-2 accuracy, per-class breakdowns (TP/FP/FN/TN, specificity), and the top 5 most confused digit pairs.

A detailed metrics report is printed to the terminal, and all metrics are saved into the metadata JSON.

### 13. System info gets captured

`system_info.py` collects the hardware/software context: Python version, PyTorch version, OS, CPU model, core count, RAM, and GPU/CUDA info. This is printed to the terminal and saved into the metadata JSON for reproducibility.

### 14. Run gets tracked

The training run is saved to a timestamped directory under `experiments/runs/` (e.g., `runs/20260305_110000_simple_fc/`) containing copies of the model weights, metadata, and training curves. The master index at `experiments/runs/run_index.json` is updated with a summary entry (timestamp, model, epochs, accuracy, F1) so you can compare runs at a glance.

## Outputs

After training completes, these files are saved/updated:

| File | Contents |
|------|----------|
| `experiments/simple_fc.pt` | Trained model weights (430 KB) |
| `experiments/simple_fc_metadata.json` | Training metadata (config, timing, per-epoch stats, 26 metrics, system info) |
| `experiments/simple_fc_training_curves.png` | Loss + accuracy curves (side-by-side) |
| `experiments/runs/<timestamp>_simple_fc/` | Timestamped run directory (weights, metadata, curves) |
| `experiments/runs/run_index.json` | Master index of all runs (newest first) |
| `output.log` | Run log with timestamps and metrics (newest at top) |

## Total Time

About 3-4 minutes on CPU (~41 sec/epoch). The model goes from random guessing (~10% accuracy) to ~97% accuracy in 5 passes through the data.