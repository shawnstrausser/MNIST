# Task Report: Update Docs for Metrics System

**Status:** Complete
**Mode:** DEVELOP (2 min max)
**Date:** 2026-03-05

---

## 1. The Problem
- README and train.md don't mention the metrics pipeline (metrics.py, system_info.py, evaluate_detailed)
- No docs for run tracking (timestamped run dirs, run_index.json)
- No docs for Makefile shortcuts
- README output.log description says "newest at bottom" — it's actually newest at top

## 2. The Solution
- Add sections to README: metrics pipeline, run tracking, Makefile shortcuts
- Update train.md: add steps for detailed metrics, system info, run tracking
- Fix output.log description in README

## 3. File Changes

| File | Action | What Changes |
|------|--------|-------------|
| `README.md` | Modify | Add metrics, run tracking, Makefile sections. Fix output.log order note |
| `docs/train.md` | Modify | Add steps 12-14: detailed metrics, system info, run dir + index |

## 4. New Data/Fields
- None

## 5. Flow
```
Read current docs -> Identify gaps -> Add missing sections -> Done
```

## 6. Backward Compatibility
- Docs only — no code changes

## 7. Smoke Test Plan
- Visual review of updated docs for accuracy and completeness

## 8. Risks
- None

---

## Progress Log

| Date | Update |
|------|--------|
| 2026-03-05 5:20pm | Report created |
| 2026-03-05 | Task completed: README + train.md updated |
