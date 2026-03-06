# Audit Output: run_all.py

**Task #:** 2
**Completed:** 2026-03-05 10:30pm PST
**File:** `run_all.py` (167 lines post-fix)

## Findings & Fixes

### 1. BUG — `--epochs 0` silently ignored (FIXED)
- **Lines:** 84, 89
- **Problem:** `if args.epochs:` and `args.epochs or ...` are falsy when epochs=0, so `--epochs 0` falls back to default (5)
- **Fix:** Changed to `if args.epochs is not None:` and ternary with `is not None`

### 2. BUG — `--skip-train --skip-viz` prints "All steps passed!" (FIXED)
- **Lines:** 130-132
- **Problem:** Both steps skipped -> empty results list -> `all([])` is True -> misleading success message
- **Fix:** Added early exit with clear message when both flags are set

### 3. REDUNDANCY — Visualize gets model name twice (FIXED)
- **Lines:** 138, 152
- **Problem:** Model passed via both `--model` CLI arg and `MNIST_MODEL` env var
- **Fix:** Removed CLI arg pass; env override is sufficient since visualize.py reads `MODEL_NAME` from config.py which respects `MNIST_MODEL` env var

### 4. EDGE CASE — Negative epochs accepted (FIXED)
- **Lines:** 114-117
- **Problem:** `--epochs -1` would pass through unchecked, producing negative time estimates
- **Fix:** Added validation: `if args.epochs is not None and args.epochs < 1: sys.exit(1)`

### 5. CODE QUALITY — No type hints (FIXED)
- **Problem:** All four functions lacked type annotations
- **Fix:** Added full type hints to `estimate_pipeline_time`, `run_step`, `run_eval_only`, `main`, `print_summary`

### 6. CONSISTENCY — Makefile naming issues (FLAGGED)
- `make train` and `make all` are identical commands
- `make eval` was misnamed (trained instead of evaluating) — fixed in task 3

## Summary

| Category | Issues Found | Fixed | Flagged |
|----------|-------------|-------|---------|
| Bugs | 2 | 2 | 0 |
| Edge cases | 1 | 1 | 0 |
| Redundancy | 1 | 1 | 0 |
| Code quality | 1 | 1 | 0 |
| Consistency | 1 | 0 | 1 |
| Missing features | 3 | 0 | 3 (became task 3) |

**Verdict:** run_all.py is solid after fixes. Two real bugs squashed, one edge case guarded, type hints added. Makefile consistency addressed in task 3.
