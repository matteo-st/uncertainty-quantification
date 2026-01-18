# Inconsistencies Log

This document tracks identified inconsistencies in experiments and reporting to prevent recurrence.

---

## Issue #1: Raw LDA Evaluation Using Fixed Hyperparameters Across Seeds

**Date Identified:** 2026-01-18

**File:** `scripts/run_raw_lda_evaluation.py`

**Problem:**
The script loaded hyperparameters only from `seed-split-1` directory for ALL seeds (1-9), instead of loading per-seed hyperparameters from each seed's respective directory.

**Root Cause:**
1. `load_score_configs()` function had hardcoded `seed-split-1` path
2. Directory sorting was alphabetical (reverse), which picked `test-lda-binning` over `lda-binning-grid-20260118`

**Impact:**
- Seeds 2-9 used seed-1's hyperparameters instead of their own
- Example: Seed 1 had gini T=1.2, but Seed 2 should have T=0.8, Seed 3 should have T=0.7
- Results in Section 8.1 of experimental report were computed with incorrect hyperparameters

**Fix Applied:**
- Added `seed_split` parameter to `load_score_configs()`
- Changed path from `seed-split-1` to `f"seed-split-{seed_split}"`
- Fixed sorting to use modification time instead of alphabetical order
- Moved config loading inside the seed loop

**Commit:** `0bf08bf` - "Fix per-seed hyperparameter loading in raw LDA evaluation"

**Status:** Fixed, awaiting re-run on server

---

## Best Practices to Prevent Future Inconsistencies

1. **Per-seed hyperparameter selection is mandatory** - Hyperparameters must be selected independently for each seed on its res split, never shared across seeds.

2. **Always verify config loading** - When loading configs from previous experiments, print/log what was loaded to verify correctness.

3. **Use tools to copy values** - When reporting results, use tools to extract values directly from result files to avoid transcription errors.

4. **Cross-check report sections** - When adding new results, verify consistency with related sections in the report.

5. **Test locally before pushing** - Run scripts with limited scope locally to catch import/logic errors before server runs.

6. **Sort by modification time, not name** - When finding "most recent" directories, use `stat().st_mtime` not alphabetical sorting.
