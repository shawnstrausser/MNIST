# Output Report: Audit README.md

**Task #:** 4
**Completed:** 2026-03-05 11:30pm PST

## What Was Done

Line-by-line audit of README.md against the actual codebase. Found and fixed 6 categories of issues.

## Findings + Fixes

| # | Section | Issue | Fix |
|---|---------|-------|-----|
| 1 | Project Structure | Missing `run_all.py`, duplicate `evaluation/` heading, missing `docs/pipeline.md`, `task_reports/`, `output_reports/` | Added all missing entries, consolidated structure |
| 2 | How to Run | No `run_all.py` commands, missing `--dry-run`, `--quiet`, `--eval-only` | Added full `run_all.py` command block with all flags |
| 3 | Makefile Shortcuts | Missing `make dry-run`, `make eval` described as "Run detailed evaluation only" (vague) | Added `make dry-run`, updated `make eval` to "Evaluate existing model (no training)" |
| 4 | Expected Output | No mention of run directories, metrics report, or prepend behavior | Added run directory save line, metrics report snippet, updated bullet points |
| 5 | Entry Points | Listed 3 entry points, missing `run_all.py` | Added `run_all.py` as entry point #1, renumbered others, updated count to 4 |
| 6 | Deep Dive | No section for `run_all.py` | Added "run_all.py -- The Pipeline Orchestrator" section with all CLI flags |

## Files Changed

| File | Changes |
|------|---------|
| `README.md` | 6 sections updated (structure, how to run, makefile, expected output, entry points, deep dive) |
| `docs/task_reports/task4_audit_readme.md` | Created (task report) |
| `docs/output_reports/task4_audit_readme_output.md` | Created (this file) |

## Verification

- All file paths referenced in README exist in the project
- All CLI commands match actual argparse flags in `run_all.py`
- Makefile shortcuts match actual Makefile targets
- Entry point count matches actual runnable scripts
