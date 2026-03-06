# Pipeline Architecture

This doc is the single reference for how the MNIST pipeline works end-to-end. It covers every entry point (Makefile commands), how `run_all.py` orchestrates training and visualization, the environment variable override mechanism, and all 9 output artifacts produced by a full run. Use this alongside `validation_make_quick.md` to understand and verify pipeline behavior.

## Entry Points (Makefile)

All commands are defined in the Makefile and funnel through `run_all.py`:

| Command | What it does | Args passed to run_all.py |
|---------|-------------|--------------------------|
| `make train` | Full training (default config) | _(none)_ |
| `make cnn` | Train the CNN model | `--model cnn` |
| `make quick` | Smoke test (1 epoch) | `--epochs 1` |
| `make viz` | Visualize only (no training) | `--skip-train` |
| `make eval` | Evaluate only (no viz) | `--skip-viz` |
| `make all` | Train + visualize | _(none)_ |
| `make clean` | Remove `__pycache__` dirs | _(direct shell, no run_all.py)_ |
| `make help` | Show available commands | _(direct shell, no run_all.py)_ |

## Pipeline Flow

```
                         ENTRY POINTS (Makefile)
 ┌──────────┬──────────┬──────────┬──────────┬──────────┐
 │make quick│make train│ make cnn │ make viz │make eval │
 │--epochs 1│ defaults │--model   │--skip-   │--skip-   │
 │          │          │  cnn     │  train   │  viz     │
 └────┬─────┴────┬─────┴────┬─────┴────┬─────┴────┬─────┘
      │          │          │          │          │
      └──────────┴──────┬───┴──────────┴──────────┘
                        v
               ┌─────────────────┐
               │   run_all.py    │
               │  (orchestrator) │
               │                 │
               │ 1. Parse args   │
               │ 2. Set env vars │
               │ 3. Time estimate│
               └───────┬─────────┘
                       │
            ┌──────────┴──────────┐
            v                     v
   ┌─────────────────┐   ┌──────────────────┐
   │   STEP 1: Train │   │ STEP 2: Visualize│
   │   train.py      │   │ evaluation/      │
   │                 │   │   visualize.py   │
   │ Reads:          │   │                  │
   │  - config.py    │   │ Reads:           │
   │  - MNIST data/  │   │  - config.py     │
   │                 │   │  - model.pt      │
   │ Calls:          │   │  - MNIST data/   │
   │  - trainer.py   │   │                  │
   │  - evaluate.py  │   │ Calls:           │
   │  - metrics.py   │   │  - evaluate.py   │
   │  - system_info  │   │  - seaborn       │
   │  - output_log   │   │  - sklearn       │
   │                 │   │  - output_log    │
   └────────┬────────┘   └────────┬─────────┘
            │                     │
            v                     v
   OUTPUTS (5 files)     OUTPUTS (2 files)
   ┌─────────────────┐   ┌──────────────────┐
   │ simple_fc.pt    │   │ simple_fc_       │
   │  (model weights)│   │  confusion_      │
   │                 │   │  matrix.png      │
   │ simple_fc_      │   │  (10x10 grid)    │
   │  metadata.json  │   │                  │
   │  (config,       │   │ simple_fc_       │
   │   metrics,      │   │  samples.png     │
   │   timing,       │   │  (5x5 grid of    │
   │   system info,  │   │   predictions)   │
   │   epoch log,    │   │                  │
   │   per-class)    │   │ + output.log     │
   │                 │   │   entry (viz)    │
   │ simple_fc_      │   └──────────────────┘
   │  training_      │
   │  curves.png     │
   │  (loss + acc)   │
   │                 │
   │ + output.log    │
   │   entry (train) │
   │                 │
   │ + run archive   │
   │   runs/YYYY-MM- │
   │   DD_HH-MM_     │
   │   model/        │
   │   ├─ model.pt   │
   │   ├─ metadata   │
   │   │  .json      │
   │   └─ training_  │
   │      curves.png │
   │                 │
   │ + run_index     │
   │   .json (entry  │
   │   prepended)    │
   └─────────────────┘
```

## Output Artifacts (per run)

A full pipeline run (`make train` or `make quick`) produces **9 artifacts** — 5 from train, 2 from visualize, plus 2 shared outputs:

### From train.py (Step 1)

| # | File | Location | Contents |
|---|------|----------|----------|
| 1 | `simple_fc.pt` | `experiments/` | Trained model weights (loadable by PyTorch) |
| 2 | `simple_fc_metadata.json` | `experiments/` | Full run record: config, 26 metrics, timing, system info, epoch log, per-class stats, confusion pairs |
| 3 | `simple_fc_training_curves.png` | `experiments/` | Side-by-side plots: train/test loss + train/test accuracy per epoch |
| 4 | Run archive directory | `experiments/runs/YYYY-MM-DD_HH-MM_model/` | Copies of model.pt, metadata.json, and training_curves.png |
| 5 | Run index entry | `experiments/runs/run_index.json` | New entry prepended (newest-first): id, model, accuracy, f1, epochs, timestamp, git commit, path |
| 6 | Train log entry | `output.log` | Appended entry with timestamp, epochs, accuracy, f1, training time |

### From visualize.py (Step 2)

| # | File | Location | Contents |
|---|------|----------|----------|
| 7 | `simple_fc_confusion_matrix.png` | `experiments/` | 10x10 seaborn heatmap — rows = true digit, cols = predicted digit, diagonal = correct |
| 8 | `simple_fc_samples.png` | `experiments/` | 5x5 grid of test images with T (true) and P (predicted) labels; green = correct, red = wrong |
| 9 | Visualize log entry | `output.log` | Appended entry with timestamp, model, test accuracy, saved file paths |

## Data Flow Summary

```
┌─────────┐    ┌──────────────┐    ┌───────────────┐
│  MNIST   │───>│    Train     │───>│   model.pt    │
│  60k/10k │    │  (1-N epochs)│    │   (weights)   │
└─────────┘    └──────┬───────┘    └───────┬───────┘
                      │                    │
                      v                    v
                ┌───────────┐       ┌────────────┐
                │ metadata  │       │ Visualize  │
                │ .json     │       └──────┬─────┘
                │           │              │
                │ - config  │        ┌─────┴──────┐
                │ - metrics │        v            v
                │ - timing  │  ┌──────────┐ ┌──────────┐
                │ - sysinfo │  │confusion │ │ samples  │
                │ - epochs  │  │matrix.png│ │  .png    │
                │ - classes │  └──────────┘ └──────────┘
                └───────────┘
```

## Environment Variable Override Mechanism

`run_all.py` translates CLI args into environment variables, then passes them to subprocesses via `env_overrides`:

```
make quick
  └─> run_all.py --epochs 1
        └─> env_overrides = {"MNIST_EPOCHS": "1"}
              └─> subprocess.run(["python", "train.py"], env={**os.environ, **env_overrides})
                    └─> config.py reads: EPOCHS = int(os.environ.get("MNIST_EPOCHS", 5))
                          └─> EPOCHS = 1
```

This is how `make quick` overrides the default 5 epochs to 1 without modifying config.py.

| CLI arg | Env variable | config.py field | Default |
|---------|-------------|-----------------|---------|
| `--epochs N` | `MNIST_EPOCHS` | `EPOCHS` | 5 |
| `--model NAME` | `MNIST_MODEL` | `MODEL_NAME` | `simple_fc` |

## Validation Checklist

When validating a pipeline run, cross-check these 7 items:

1. **metadata.json** — config matches config.py, metrics are internally consistent, epoch count correct
2. **confusion_matrix.png** — diagonal values match metadata `per_class_metrics` tp counts
3. **samples.png** — 25 samples displayed, green/red labels match T==P or T!=P
4. **training_curves.png** — data points match `epoch_log` entries in metadata
5. **Run archive** — `runs/<id>/` contains model.pt + metadata.json + curves.png, metadata identical to top-level copy
6. **run_index.json** — newest entry matches metadata (accuracy, f1, epochs, timestamp, git commit)
7. **output.log** — train + visualize entries present, metrics match metadata

See `validation_make_quick.md` for a worked example.