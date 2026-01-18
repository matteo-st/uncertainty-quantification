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

## Issue #2: Report Sections Lack Clarity on Per-Seed Hyperparameter Selection

**Date Identified:** 2026-01-18

**File:** `docs/experimental_procedure_report.md` - Multiple sections

**Problem:**
Multiple sections stated hyperparameters as single values or used vague language like "selected on res split" without clarifying that selection is done **per-seed**. This could mislead readers into thinking hyperparameters are fixed across all seeds.

**Affected Sections:**
1. "Experiment: Uniform Mass Binning" - listed single hyperparameters per dataset/model
2. "Isotonic Regression Calibration (9 seeds)" - vague "selected on res split"
3. "Isotonic Regression with Bin Splitting (9 seeds)" - vague "selected on res split"
4. "Experiment: K-means Partition Binning (9 seeds)" - vague "selected on res split"
5. "8. LDA Binning" - per-score hyperparameters not marked as per-seed

**Actual Behavior (correct):**
Hyperparameters vary across seeds. Example CIFAR-10 ResNet-34:
- Seed 1: temperature=1.2, magnitude=0.002
- Seed 2: temperature=0.8, magnitude=0.004
- Seed 3: temperature=0.7, magnitude=0.004

**Type:** Reporting inconsistency (experiments were correct, report was unclear)

**Fix Applied:**
- Added "**per-seed**" emphasis to all affected sections
- Uniform Mass: Changed specific values to ranges
- Isotonic: Added "(hyperparameters vary across seeds, e.g., temperature 0.7-1.2)"
- K-means: Added "**per-seed** by FPR@95 on each res split"
- LDA Binning: Added "(per-seed)" to each score source

**Status:** Fixed

---

## Issue #3: Inconsistent Precision and Incorrect Std Values for Doctor Baseline

**Date Identified:** 2026-01-18

**File:** `docs/experimental_procedure_report.md` - Multiple sections

**Problem:**
1. **Inconsistent precision:** Some sections used 3-decimal places (0.930 ± 0.011), others used 4-decimal (0.9297 ± 0.0108) for the same values
2. **Incorrect std values:** K-means section had different std values than actual computed values

**Root Cause:**
Values were **manually typed/transcribed** instead of being programmatically extracted and copied from result files. This led to:
- Typos in std values (0.0176 instead of 0.0166 - likely 7↔6 transposition)
- Inconsistent rounding conventions between sections written at different times

**Examples:**
- Isotonic section: FPR=0.1982 ± 0.0166, ROC-AUC=0.9297 ± 0.0108 (correct)
- Uniform Mass section: FPR=0.198 ± 0.017, ROC-AUC=0.930 ± 0.011 (rounded but inconsistent)
- K-means section: FPR=0.1982 ± 0.0176, ROC-AUC=0.9297 ± 0.0115 (incorrect std)

**Correct values (from doctor-eval runs):**
| Dataset | Model | FPR (test) | ROC-AUC (test) |
|---------|-------|------------|----------------|
| CIFAR-10 | ResNet-34 | 0.1982 ± 0.0166 | 0.9297 ± 0.0108 |
| CIFAR-10 | DenseNet-121 | 0.2650 ± 0.0219 | 0.9124 ± 0.0052 |
| CIFAR-100 | ResNet-34 | 0.3948 ± 0.0177 | 0.8726 ± 0.0039 |
| CIFAR-100 | DenseNet-121 | 0.4614 ± 0.0137 | 0.8570 ± 0.0038 |

**Fix Applied:**
- Standardized all Doctor baseline values to 4-decimal precision
- Fixed K-means section std values to match actual computed values
- Updated: Uniform Mass section, Selection Rule Analysis, LDA Binning comparison

**Status:** Fixed

---

## Best Practices to Prevent Future Inconsistencies

1. **Per-seed hyperparameter selection is mandatory** - Hyperparameters must be selected independently for each seed on its res split, never shared across seeds.

2. **Always verify config loading** - When loading configs from previous experiments, print/log what was loaded to verify correctness.

3. **Use tools to copy values** - When reporting results, use tools to extract values directly from result files to avoid transcription errors.

4. **Cross-check report sections** - When adding new results, verify consistency with related sections in the report.

5. **Test locally before pushing** - Run scripts with limited scope locally to catch import/logic errors before server runs.

6. **Sort by modification time, not name** - When finding "most recent" directories, use `stat().st_mtime` not alphabetical sorting.

7. **Use consistent precision** - All baseline values should use the same decimal precision across all sections (prefer 4-decimal for metrics like ROC-AUC, FPR). When the same value appears in multiple sections, use identical formatting.
