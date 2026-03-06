# Task Report: run_all.py Feature Additions

**Task #:** 3
**Status:** Complete
**Created:** 2026-03-05 10:15pm PST

## 1. The Problem
- run_all.py has no way to preview the plan without executing it
- Output verbosity is fixed — no quiet mode for CI or verbose mode for debugging
- No evaluate-only mode — `make eval` is misnamed and actually trains

## 2. The Solution
Add three flags to run_all.py:
- `--dry-run` — show plan (model, epochs, time estimate) then exit without running
- `--quiet` — suppress step banners and time estimates, only show results
- `--eval-only` — run evaluation on existing weights without training or visualization

## 3. File Changes

| File | Action | What changes |
|------|--------|-------------|
| `run_all.py` | Modify | Add `--dry-run`, `--quiet`, `--eval-only` flags + logic |
| `Makefile` | Modify | Fix `make eval` to use `--eval-only`, add `make dry-run` |

## 4. New Data/Fields
- No new config or metadata changes

## 5. Flow
```
argparse
  |
  +-- --dry-run?  --> print plan --> sys.exit(0)
  |
  +-- --eval-only? --> skip train + viz, run evaluate_detailed --> print metrics
  |
  +-- --quiet?  --> suppress banners in run_step + estimate display
  |
  v
(existing pipeline continues as normal)
```

## 6. Backward Compatibility
- All new flags are opt-in, defaults unchanged
- `make eval` behavior changes (now runs eval-only instead of train-only)

## 7. Smoke Test Plan
- `python run_all.py --dry-run` — should print plan and exit, no training
- `python run_all.py --quiet --epochs 1` — should run with minimal output
- `python run_all.py --eval-only` — should load existing model and print metrics
- `make eval` — should now run eval-only
- `make dry-run` — should show plan

## 8. Risks
- `--eval-only` needs a saved model file to exist — will error if no `.pt` found
- `--quiet` needs careful scoping so errors still show
