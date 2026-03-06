# Output Report: run_all.py Feature Additions

**Task #:** 3
**Completed:** 2026-03-05 10:30pm PST

## What Was Done

Added three new flags to `run_all.py` and updated the Makefile:

### 1. `--dry-run`
- Shows pipeline plan (model, epochs, time estimate) then exits
- Works with `--quiet` for a one-line summary
- Makefile: `make dry-run`

### 2. `--quiet`
- Suppresses step banners (`=====`, `[Train] Starting...`)
- Suppresses plan display (model/epochs/estimate header)
- Pipeline summary and errors still print
- Subprocess output still streams (train.py prints its own output)

### 3. `--eval-only`
- Loads existing `.pt` weights from `experiments/`
- Runs `evaluate_detailed` + `compute_all_metrics` + `print_metrics_report`
- No training, no visualization
- Exits with code 1 if model file not found
- Makefile: `make eval` (renamed from old behavior of `--skip-viz`)

## Files Changed

| File | Changes |
|------|---------|
| `run_all.py` | Added `--dry-run`, `--quiet`, `--eval-only` args; `run_eval_only()` function; quiet support in `run_step()` and plan display |
| `Makefile` | `make eval` now uses `--eval-only`; added `make dry-run` target; updated help text |

## Before / After

| Command | Before | After |
|---------|--------|-------|
| `make eval` | Trains model, skips viz | Evaluates existing model only |
| `make dry-run` | Did not exist | Shows plan, exits |
| `--quiet` flag | Did not exist | Minimal output mode |
| `--eval-only` flag | Did not exist | Eval-only mode |
