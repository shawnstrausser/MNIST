# Task Report: Build Timestamped Run Tracking

**Status:** APPROVED — ready to build
**Mode:** DEVELOP (2 min max)
**Date:** 2026-03-05

---

## 1. The Problem
- Training runs overwrite previous results (model.pt, metadata.json, plots)
- No way to compare runs or track experiment history
- Lose valuable data every time we retrain

## 2. The Solution
- Each run saves to its own timestamped folder: `experiments/runs/YYYY-MM-DD_HH-MM_modelname/`
- A `run_index.json` master index tracks all runs for quick comparison
- Git commit hash added to metadata for reproducibility
- Design: Option A (simple folders, no external deps)

## 3. File Changes

| File | Action | What Changes |
|------|--------|-------------|
| `config.py` | Modify | Add `get_run_dir()` helper that creates timestamped folder |
| `training/trainer.py` | Modify | Save model + metadata to run dir instead of flat `experiments/` |
| `evaluation/visualize.py` | Modify | Save plots to run dir |
| `train.py` | Modify | Wire up run dir through training + eval pipeline |
| `experiments/run_index.json` | New | Master index of all runs |

## 4. New Data/Fields

**In metadata.json:**
- `git_commit` — short hash of current commit
- `run_dir` — path to this run's folder

**run_index.json schema:**
```json
{
  "runs": [
    {
      "id": "2026-03-05_14-30_simple_fc",
      "model": "simple_fc",
      "accuracy": 0.9537,
      "timestamp": "2026-03-05T14:30:00",
      "git_commit": "770a460",
      "path": "experiments/runs/2026-03-05_14-30_simple_fc/"
    }
  ]
}
```

## 5. Flow

```
train.py starts
    |
    v
config.get_run_dir() --> creates experiments/runs/YYYY-MM-DD_HH-MM_modelname/
    |
    v
trainer.train() --> saves model.pt + metadata.json to run dir
    |
    v
visualize.py --> saves plots to run dir
    |
    v
update run_index.json --> appends new entry (newest first)
    |
    v
Done! All artifacts in one timestamped folder
```

## 6. Backward Compatibility
- Existing `experiments/simple_fc_metadata.json` and `experiments/*.png` stay untouched
- Old flat structure remains as-is, new runs go to `experiments/runs/`
- No breaking changes to any existing code paths

## 7. Smoke Test Plan

**Command:**
```bash
cd ~/Desktop/MNIST && MNIST_EPOCHS=1 python train.py
```

**Expected:**
- New folder created: `experiments/runs/YYYY-MM-DD_HH-MM_simple_fc/`
- Contains: `model.pt`, `metadata.json`, all plot PNGs
- `metadata.json` has `git_commit` and `run_dir` fields
- `experiments/run_index.json` exists with one entry
- Old `experiments/simple_fc_metadata.json` still exists (not deleted)

## 8. Risks
- Clock format edge cases (midnight, timezone) — mitigated by using UTC or local consistently
- Run folder name collision if two runs start same minute — low risk, add seconds if needed
- Git commit hash fails if not in a git repo — fallback to "unknown"
