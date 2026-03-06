# Task Report: Audit run_all.py

**Task #:** 2
**Status:** Complete
**Created:** 2026-03-05 9:45pm PST

## 1. The Problem
- `run_all.py` is the central orchestrator — every pipeline command flows through it
- Never been formally audited for correctness, edge cases, or missing features
- Already found one bug here (env overrides not passed to subprocess)
- Need confidence the file is solid before relying on it for all future runs

## 2. The Solution
Line-by-line audit covering:
- **Correctness** — does every code path do what it claims?
- **Edge cases** — what happens with bad inputs, missing files, unexpected state?
- **Consistency** — does it match config.py, Makefile, and docs?
- **Code quality** — naming, structure, duplication, readability
- **Missing features** — anything obviously useful that's absent?

## 3. File Changes

| File | Action | What changes |
|------|--------|-------------|
| `run_all.py` | Modify | Fix any issues found during audit |
| `docs/task_reports/task2_audit_run_all.md` | New | This task report |
| `docs/output_reports/task2_audit_run_all_output.md` | New | Findings + fixes applied |

## 4. New Data/Fields
- None expected (audit, not feature work)

## 5. Flow
```
Read run_all.py line by line
  |
  v
Check each section:
  argparse --> env_overrides --> estimate --> plan display
                                                  |
                                            run_step (Train)
                                                  |
                                            run_step (Visualize)
                                                  |
                                            print_summary
  |
  v
Document findings (bugs, edge cases, improvements)
  |
  v
Fix issues, update docs
```

## 6. Backward Compatibility
- Audit only — no breaking changes
- Any fixes will preserve existing behavior

## 7. Smoke Test Plan
- `make quick` after any fixes — should produce same output as before
- Verify edge cases: `--epochs 0`, `--skip-train --skip-viz`, missing model name

## 8. Risks
- Low risk — read-heavy task, minimal code changes expected