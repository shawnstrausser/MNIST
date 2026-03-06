# Validation: `make quick` Output

**Date:** 2026-03-05 9:19pm PST
**Command:** `make quick` -> `python run_all.py --epochs 1`
**Model:** simple_fc
**Epochs:** 1
**Run ID:** 2026-03-05_18-31_simple_fc

---

## Checklist

| # | Output | Status | Notes |
|---|--------|--------|-------|
| 1 | simple_fc_metadata.json | PASS | All config, metrics, timing, and epoch log validated |
| 2 | simple_fc_confusion_matrix.png | PASS | All 10 diagonal values match metadata tp counts |
| 3 | simple_fc_samples.png | PASS | 25 samples, all correct (T==P), all green labels |
| 4 | simple_fc_training_curves.png | PASS | 1 data point, values match epoch_log |
| 5 | runs/2026-03-05_18-31_simple_fc/ | PASS | 3 files present, metadata identical to top-level copy |
| 6 | runs/run_index.json | PASS | Newest entry matches metadata, newest-first order |
| 7 | output.log | PASS | Train + Visualize entries present, metrics match metadata |

---

## Detailed Validation

### 1. simple_fc_metadata.json — PASS

**Config vs config.py:**

| Field | metadata.json | config.py | Match? |
|-------|---------------|-----------|--------|
| epochs | 1 | MNIST_EPOCHS=1 (via env override) | Yes |
| batch_size | 64 | BATCH_SIZE = 64 | Yes |
| learning_rate | 0.001 | LEARNING_RATE = 1e-3 | Yes |
| device | cpu | DEVICE = "cpu" | Yes |
| optimizer | Adam | train.py line 112 | Yes |
| model_params | 109,386 | — | Plausible for 784->128->64->10 FC net |

**Performance:**

| Metric | Value | Sanity check |
|--------|-------|--------------|
| total_training_time_sec | 39.99 | Reasonable for 1 epoch on CPU |
| avg_epoch_time_sec | 39.99 | Equal to total (1 epoch) — correct |
| training_throughput | 1,500.4 img/sec | 60,000 / 39.99 = 1,500.4 — exact match |
| eval_throughput | 2,372.4 img/sec | Faster than train (no backprop) — expected |
| model_size_mb | 0.42 | 440,253 bytes / 1,048,576 = 0.42 — correct |

**Final metrics consistency:**

| Metric | Value |
|--------|-------|
| accuracy | 0.959 |
| precision_macro | 0.9588 |
| recall_macro | 0.9588 |
| f1_macro | 0.9587 |
| balanced_accuracy | 0.9588 |

All within ~0.1% of each other — consistent (no class imbalance skew).

**Epoch log:** 1 entry. train_acc (0.9202) < test_acc (0.959) — normal for epoch 1 (dropout active during training).

**Dataset:** 60,000 train / 10,000 test — standard MNIST.

### 2. simple_fc_confusion_matrix.png — PASS

**Diagonal (tp) values cross-referenced against metadata per_class_metrics:**

| Digit | Matrix diagonal | metadata tp | Match? |
|-------|----------------|-------------|--------|
| 0 | 964 | 964 | Yes |
| 1 | 1101 | 1101 | Yes |
| 2 | 993 | 993 | Yes |
| 3 | 955 | 955 | Yes |
| 4 | 954 | 954 | Yes |
| 5 | 846 | 846 | Yes |
| 6 | 913 | 913 | Yes |
| 7 | 986 | 986 | Yes |
| 8 | 919 | 919 | Yes |
| 9 | 959 | 959 | Yes |

**Top confusion pairs cross-referenced against matrix off-diagonal cells:**

| Pair | metadata count | Matrix cell (row, col) | Match? |
|------|---------------|------------------------|--------|
| 9->4 | 21 | row 9, col 4 = 21 | Yes |
| 1->8 | 20 | row 1, col 8 = 20 | Yes |
| 4->9 | 17 | row 4, col 9 = 17 | Yes |
| 6->4 | 17 | row 6, col 4 = 17 | Yes |
| 8->4 | 16 | row 8, col 4 = 16 | Yes |

### 3. simple_fc_samples.png — PASS

- 5x5 grid = 25 sample predictions displayed
- All 25 labels are green (T == P in every case)
- Title reads "Sample Predictions — simple_fc" — correct model name
- Visual spot check: digits visually match their T labels

### 4. simple_fc_training_curves.png — PASS

- Only 1 data point per series — correct for 1 epoch
- **Loss plot:** Train Loss dot at ~0.27, Test Loss dot at ~0.13 — matches epoch_log (train_loss=0.2718, test_loss=0.1337)
- **Accuracy plot:** Train Acc dot at ~0.92, Test Acc dot at ~0.955 — matches epoch_log (train_acc=0.9202, test_acc=0.959)
- Note: with only 1 epoch, no "curve" to show — just isolated points. This is expected behavior, not a bug.

### 5. Run archive (runs/2026-03-05_18-31_simple_fc/) — PASS

| File | Size | Check |
|------|------|-------|
| metadata.json | 4.5K | `diff` against top-level copy = identical |
| model.pt | 430K | Present, matches model_size_bytes (440,253) |
| simple_fc_training_curves.png | 65K | Present |

### 6. run_index.json — PASS

**Newest entry cross-referenced against metadata:**

| Field | run_index | metadata | Match? |
|-------|-----------|----------|--------|
| id | 2026-03-05_18-31_simple_fc | (run_dir basename) | Yes |
| model | simple_fc | simple_fc | Yes |
| accuracy | 0.959 | 0.959 | Yes |
| f1_macro | 0.9586833690119342 | 0.9586833690119342 | Yes |
| epochs | 1 | 1 | Yes |
| timestamp | 2026-03-05T18:31:37.929202 | 2026-03-05T18:31:37.929202 | Yes |
| git_commit | 72fd517 | 72fd517 | Yes |

Run order: newest-first (18-31, 18-05, 16-36, 13-01) — correct.

### 7. output.log — PASS

**Most recent train entry (line 12):**

| Field | output.log | metadata | Match? |
|-------|-----------|----------|--------|
| timestamp | 2026-03-05T18:31:41 | 2026-03-05T18:31:37 | ~4s diff (log written after metadata) |
| epochs | 1 | 1 | Yes |
| final_test_acc | 0.959 | 0.959 | Yes |
| f1_macro | 0.9586833690119342 | 0.9586833690119342 | Yes |
| total_time_sec | 39.99 | 39.99 | Yes |

**Most recent visualize entry (line 2):**
- test_accuracy: 0.959 — matches metadata accuracy
- Saved paths: confusion_matrix + samples — both present on disk

---

## Summary

**Result: 7/7 PASS**

All outputs from `make quick` are internally consistent and cross-validated. The --epochs 1 override is correctly applied (bug fix confirmed working). No anomalies found.