# Task Report: Makefile for Pipeline Shortcuts

**Status:** BLOCKED — waiting on `make` install
**Mode:** DEVELOP (2 min max)
**Date:** 2026-03-05

---

## 1. The Problem
- Have to type `python run_all.py --flags` every time
- No quick shortcuts for common workflows (train, viz, smoke test, CNN)

## 2. The Solution
- Create a `Makefile` with targets for every common command
- One-word shortcuts: `make train`, `make viz`, `make all`, `make quick`, `make cnn`

## 3. File Changes

| File | Action | What Changes |
|------|--------|-------------|
| `Makefile` | New | Build script with pipeline shortcuts |
| `README.md` | Modify | Add Makefile commands section |

## 4. New Data/Fields
- None

## 5. Flow
```
make <target>
    |
    v
Makefile translates to python command
    |
    v
run_all.py / train.py / evaluate / visualize
```

## 6. Backward Compatibility
- Purely additive — all existing commands still work
- Nothing changes, just adds shortcuts

## 7. Smoke Test Plan
```bash
cd Desktop/MNIST && make quick
```
**Expected:** Runs `python run_all.py --epochs 1`, trains 1 epoch, visualizes, prints summary with "All steps passed!"

## 8. Risks
- `make` might not be installed on Windows — need to verify. Fallback: use a `.bat` file instead

---

## Progress Log

| Date | Update |
|------|--------|
| 2026-03-05 8:00pm | Report created, approved |
| 2026-03-05 8:10pm | `make` not found on system. Attempting install via `choco install make` |
| 2026-03-05 8:12pm | Choco failed — stale lock file + not running as admin |
| 2026-03-05 8:15pm | Lock file already gone. Need to retry `choco install make -y` from admin PowerShell |
| 2026-03-05 8:15pm | **BLOCKED** — waiting for Shawn to install `make` from admin PowerShell |

## Unblock Steps
1. Open PowerShell as Administrator
2. Run `choco install make -y`
3. Close and reopen terminals
4. Verify with `make --version` (expect `GNU Make 4.4.1`)
5. Then I'll create the Makefile
