# MNIST Digit Classifier

Teach a computer to recognize handwritten digits (0-9) using PyTorch.

## What This Does

You have images of handwritten digits. This project trains a neural network to look at a new image and correctly guess which digit it is.

## Project Structure

```
MNIST/
  config.py              — Control panel: batch size, learning rate, epochs, model selection
  train.py               — Main entry point — ties everything together
  data/MNIST/raw/        — Raw MNIST data files (60k train + 10k test images)
  models/
    simple_fc.py         — Simple 3-layer feedforward network (784 → 128 → 64 → 10)
    cnn.py               — Convolutional neural network (~99% accuracy)
    registry.py          — Model lookup table (add future models here)
  training/
    trainer.py           — Training loop: feed data, measure error, adjust weights
  evaluation/
    evaluate.py          — Test loop: measure accuracy on unseen data
    visualize.py         — Confusion matrix + sample prediction visualizations (seaborn)
  experiments/           — Saved outputs: model weights (.pt), metadata (.json), charts (.png)
  utils/
    data.py              — Data loading + image transforms
    output_log.py        — Structured run logging (appends to output.log)
  docs/
    train.md             — Deep dive: what happens when you run train.py
    evaluate.md          — Deep dive: what happens when you run evaluate.py
    visualize.md         — Deep dive: what happens when you run visualize.py
```

## How to Run

```bash
# Install dependencies
pip install torch torchvision seaborn scikit-learn

# Train the model
cd Desktop/MNIST
python train.py

# Test accuracy without retraining
python -m evaluation.evaluate
python -m evaluation.evaluate --model cnn

# Generate visualizations (confusion matrix + sample predictions)
python -m evaluation.visualize             # uses default model from config
python -m evaluation.visualize --model cnn # specify model
```

## Expected Output

```
Model: simple_fc | Device: cpu | Epochs: 5
--------------------------------------------------
  Training: [██████████████████████████████] 100.0%  loss=0.0512  acc=0.9837
  Eval:     [██████████████████████████████] 100.0%  loss=0.0891  acc=0.9752
Epoch 1/5 | Train Loss: 0.2653 Acc: 0.9217 | Test Loss: 0.1264 Acc: 0.9614 | Time: 43.3s
  ...
Epoch 5/5 | Train Loss: 0.0500 Acc: 0.9837 | Test Loss: 0.0891 Acc: 0.9752 | Time: 38.7s

Model saved to experiments/simple_fc.pt

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

Metadata saved to experiments/simple_fc_metadata.json
Training curves saved to experiments/simple_fc_training_curves.png
Run logged to output.log
```

- **Loss** goes down each epoch (model is making fewer mistakes)
- **Accuracy** goes up each epoch (model is guessing more digits correctly)
- Final test accuracy ~97% for the simple FC model, ~99% for the CNN
- **Architecture diagram** prints to terminal after training
- **Training curves** (loss + accuracy side-by-side) saved as PNG
- **Run log** appended to `output.log` with timestamps, config, and results

## Key Concepts

- **Epoch**: One full pass through all 60,000 training images
- **Batch**: A group of 64 images processed at once (faster than one at a time)
- **Loss**: A number measuring how wrong the model is (lower = better)
- **Accuracy**: Percentage of digits guessed correctly (higher = better)
- **Training vs Test**: Train on 60k images, test on a separate 10k the model has never seen — this tells us if it actually *learned* vs just *memorized*

## System Architecture

Every runnable file imports constants from config.py. Here's who imports what, and what they produce:

```
                    ┌─────────────────────────────┐
                    │         config.py            │
                    │  (shared constants for all)  │
                    └──────┬──────────┬────────────┘
                           │          │
          imports:         │          │        imports:
    DEVICE, EPOCHS,        │          │   DEVICE, MODEL_NAME,
    LEARNING_RATE,         │          │   EXPERIMENTS_DIR
    MODEL_NAME,            │          │
    EXPERIMENTS_DIR        │          │
                           │          │
              ┌────────────┘          └──────────────────┐
              │                                          │
              v                                          v
     ┌──────────────────┐                  ┌───────────────────────┐
     │    train.py      │                  │  evaluate.py          │
     │   (main entry)   │                  │  (standalone test)    │
     │                  │                  │                       │
     │ Uses:            │                  │ Uses:                 │
     │  - data.py       │                  │  - data.py            │
     │  - registry.py   │                  │  - registry.py        │
     │  - trainer.py    │                  │  - experiments/*.pt   │
     │  - evaluate.py   │                  │                       │
     │                  │                  │ Output:               │
     │ Output:          │                  │  - terminal text      │
     │  - .pt model     │                  │    (loss + accuracy)  │
     │  - .json metadata│                  └───────────────────────┘
     │  - training      │
     │    curves PNG    │
     │  - architecture  │
     │    (terminal)    │
     │  - output.log    │
     └────────┬─────────┘
              │                            ┌───────────────────────┐
              │   saved to                 │  visualize.py         │
              v                            │  (standalone viz)     │
     ┌──────────────────┐                  │                       │
     │  experiments/    │                  │ Uses:                 │
     │                  │ <── loaded by ── │  - data.py            │
     │  simple_fc.pt   │                  │  - registry.py        │
     │  cnn.pt         │                  │  - experiments/*.pt   │
     │  *_confusion.png│ <── saved by ──  │  - --model CLI flag   │
     │  *_samples.png  │                  │                       │
     │  *_training_    │                  │ Output:               │
     │    curves.png   │                  │  - 2 PNG files        │
     └──────────────────┘                  │  - output.log entry   │
                                           └───────────────────────┘
```

## Entry Points

There are **3 entry points** you can run. Each one imports from config.py:

```
1. python train.py                          (THE MAIN WORKFLOW)
            │
            ├─ Imports: DEVICE, EPOCHS, LEARNING_RATE, MODEL_NAME, EXPERIMENTS_DIR, BATCH_SIZE
            ├─ data.py downloads/loads MNIST images
            ├─ registry.py builds the chosen model
            │
            ├─ For each epoch (1 to 5):
            │   ├─ trainer.py trains on 60k images (live progress bar)
            │   ├─ evaluate.py tests on 10k images (live progress bar)
            │   └─ Prints: loss ↓ accuracy ↑ time ⏱
            │
            ├─ Prints ASCII architecture diagram to terminal
            │
            └─ Saves to experiments/:
                ├─ simple_fc.pt                    (trained model weights)
                ├─ simple_fc_metadata.json         (config, timing, per-epoch stats)
                └─ simple_fc_training_curves.png   (loss + accuracy side-by-side)
            └─ Appends run to output.log


2. python -m evaluation.evaluate            (TEST ONLY — no training)
            │
            ├─ Imports: DEVICE, MODEL_NAME, EXPERIMENTS_DIR
            ├─ Loads trained model from experiments/*.pt
            ├─ Loads test data via data.py
            │
            └─ Output: prints loss + accuracy to terminal


3. python -m evaluation.visualize           (VISUALIZE — no training)
            │
            ├─ Imports: DEVICE, MODEL_NAME, EXPERIMENTS_DIR
            ├─ Accepts --model flag to override config
            ├─ Loads trained model from experiments/*.pt
            ├─ Loads test data via data.py
            │
            └─ Output: saves 2 PNGs to experiments/
                       (confusion matrix + sample predictions)
```

## Deep Dive: File by File

### config.py — The Control Panel

Everything you'd want to tweak lives here: batch size, learning rate, epochs, which model to use, CPU vs GPU. One file to rule them all. Change `MODEL_NAME` to `"cnn"` and the whole pipeline switches architectures.

Key settings:
- `BATCH_SIZE = 64` — How many images to process at once (bigger = faster but uses more RAM)
- `LEARNING_RATE = 1e-3` — How aggressively the model adjusts its weights each step
- `EPOCHS = 5` — How many full passes through the training data
- `MODEL_NAME = "simple_fc"` — Which neural network architecture to use (options: `"simple_fc"`, `"cnn"`)
- `DEVICE = "cpu"` — Where the math runs (`"cpu"` or `"cuda"` for GPU)

### utils/data.py — Data Loading

Downloads MNIST (60k training + 10k test images), applies two transforms:
- **ToTensor()** — Converts the 28x28 grayscale image to a PyTorch tensor with values in [0, 1]
- **Normalize()** — Shifts values so the mean is ~0 and std is ~1, which helps the model learn faster and more stably. (0.1307 and 0.3081 are the precomputed MNIST mean/std.)

Then wraps them in DataLoaders that serve batches of 64 images at a time.

### models/simple_fc.py — The Simple Model

A fully-connected (dense) neural network. The simplest architecture for MNIST: flatten each 28x28 image into a 784-number list, then pass it through three layers that progressively narrow down to 10 output scores — one per digit (0-9).

```
Architecture:  784 → 128 → 64 → 10
```

The highest score wins: if output[3] is the largest, the model predicts "3". Achieves ~97% test accuracy — solid for such a simple design.

How the layers work:
- `nn.Flatten()` — 28x28 image → flat list of 784 numbers
- `nn.Linear(784, 128)` — First dense layer, 784 inputs → 128 outputs
- `nn.ReLU()` — Zeros out negatives (adds non-linearity so the model can learn complex patterns)
- `nn.Linear(128, 64)` — Second dense layer, narrows down
- `nn.ReLU()`
- `nn.Linear(64, 10)` — Final layer, outputs 10 scores (one per digit)

### models/cnn.py — The Smart Model

Unlike the fully-connected model, a CNN looks at small patches of the image at a time using sliding filters (convolutions). This lets it learn spatial patterns like edges, curves, and loops — exactly the kind of features that distinguish handwritten digits.

```
Architecture:
  Conv(1→32, 3x3) → ReLU → Conv(32→64, 3x3) → ReLU → MaxPool(2x2)
  → Flatten → Linear(12544→128) → ReLU → Linear(128→10)
```

Two stages:
- **Features stage** — Two convolutional layers detect patterns (edges, curves), then MaxPool shrinks the image from 28x28 → 14x14 to focus on what matters
- **Classifier stage** — Flatten the feature maps and pass through dense layers to get 10 digit scores

Typically reaches ~99% test accuracy — a nice bump over the ~97% FC model.

### models/registry.py — Model Lookup Table

Maps a short name (like `"simple_fc"`) to the actual class that builds the neural network. This lets config.py choose a model by name without importing every model file directly.

To add a new model:
1. Create the class in `models/` (e.g., `my_model.py`)
2. Import it in `registry.py`
3. Add an entry to `MODEL_REGISTRY`

### training/trainer.py — The Workout

One function: `train_one_epoch`. Does one full pass through all 60,000 training images. For every batch of images it:

1. Feeds them through the model to get predictions (forward pass)
2. Measures how wrong those predictions are (loss via CrossEntropyLoss)
3. Calculates which direction to nudge the weights (backpropagation via `loss.backward()`)
4. Nudges the weights a small step in that direction (`optimizer.step()`)

Shows a live progress bar during training with real-time loss and accuracy:
```
  Training: [████████████████████░░░░░░░░░░] 68.2%  loss=0.1234  acc=0.9456
```

Returns `(avg_loss, accuracy)` for the epoch — so we can watch the model improve.

### evaluation/evaluate.py — The Test

Same idea as training but **no learning happens** — just measuring. Key differences:
- `model.eval()` — Tells PyTorch "we're testing, not training" (turns off training-only behaviors like dropout)
- `torch.no_grad()` — Tells PyTorch "don't track gradients" (saves memory and speeds things up)

Runs all 10k test images through the model and reports loss + accuracy. This is the honest score since the model never saw these images during training.

When called from train.py with `show_progress=True`, shows a live progress bar:
```
  Eval:     [██████████████████████████████] 100.0%  loss=0.0891  acc=0.9752
```

### evaluation/visualize.py — The Eye Candy

Generates two PNG visualizations after training:

1. **Confusion Matrix** — A 10x10 grid showing which digits get mixed up. Perfect model = bright diagonal, everything else dark.
2. **Sample Predictions** — A grid of test images with the model's guess. Correct predictions in green, wrong ones in red.

How it works:
- Loads a trained model from `experiments/`
- Runs all test images through it to collect predictions
- Uses seaborn + scikit-learn to generate the plots
- Saves PNGs to `experiments/`
- Appends run to `output.log`

### utils/output_log.py — Run Logging

Every time train.py, evaluate, or visualize runs, it appends a structured entry to `output.log` in the project root. Each entry includes a timestamp, the command that ran, success/error status, and a summary of key metrics. This gives you a single file to check what happened, when, and whether it worked.

The log is append-only — newest entries at the bottom. Added to `.gitignore` so each machine has its own local log.