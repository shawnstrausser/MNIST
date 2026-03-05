# visualize.py — What Happens When You Run It

## Quick Summary

`python -m evaluation.visualize` loads a trained model, runs it against all 10k test images, and generates two PNG images that show you what the model actually learned (and where it struggles).

## Step by Step

### 1. Config gets imported + CLI flag checked

Python reads `config.py` and grabs:
- `DEVICE` (cpu)
- `MODEL_NAME` ("simple_fc")
- `EXPERIMENTS_DIR` (experiments/)

Then checks if you passed `--model` on the command line (e.g., `--model cnn`). If so, that overrides the config value.

### 2. Trained model gets loaded

Same as evaluate.py — loads saved weights from `experiments/<model_name>.pt` into a fresh model instance, sets it to eval mode.

### 3. All test predictions get collected

Runs every test image through the model and collects three things:
- **true_labels** — what the digit actually is
- **pred_labels** — what the model guessed
- **images** — the actual pixel data (for display)

### 4. Confusion matrix gets generated

```
             Predicted
           0  1  2  3  4  5  6  7  8  9
        0 [■  .  .  .  .  .  .  .  .  .]
        1 [.  ■  .  .  .  .  .  .  .  .]
Actual  2 [.  .  ■  .  .  .  .  .  .  .]
        3 [.  .  .  ■  .  .  .  .  .  .]
        ...
        9 [.  .  .  .  .  .  .  .  .  ■]

■ = bright (many correct)    . = dark (few/no mistakes)
```

A 10x10 grid. Each cell shows how many times a digit (row) was predicted as another digit (column). A perfect model = bright diagonal, everything else dark. Off-diagonal bright spots reveal which digits get confused (e.g., 4s and 9s, 3s and 8s).

Uses `sklearn.metrics.confusion_matrix` + `seaborn.heatmap` to render.

Saved to: `experiments/<model_name>_confusion_matrix.png`

### 5. Sample predictions grid gets generated

A 5x5 grid of 25 randomly selected test images, each labeled with:
- **T:** — the true digit
- **P:** — the model's prediction
- **Green title** — correct prediction
- **Red title** — wrong prediction

Uses a fixed random seed (42) so the same images are selected every time.

Saved to: `experiments/<model_name>_samples.png`

### 6. Run gets logged to output.log

A structured entry is appended to `output.log` with the timestamp, model name, test accuracy, and paths to saved files. On error, the full traceback is logged instead.

### 7. Done

```
Generating visualizations for: simple_fc
Test accuracy: 0.9723
Saved confusion matrix to experiments/simple_fc_confusion_matrix.png
Saved sample predictions to experiments/simple_fc_samples.png
Run logged to output.log
Done!
```

## Dependencies

This is the only file that requires extra packages beyond PyTorch:
- `seaborn` — for rendering the heatmap (pulls in matplotlib)
- `scikit-learn` — for computing the confusion matrix

Install with: `pip install seaborn scikit-learn`