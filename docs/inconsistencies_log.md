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

## Issue #4: Raw LDA Evaluation Missing ODIN Input Perturbation

**Date Identified:** 2026-01-18

**File:** `scripts/run_raw_lda_evaluation.py`

**Problem:**
The `compute_gini_score()`, `compute_margin_score()`, and `compute_msp_score()` functions accept a `magnitude` parameter but **never use it**. The magnitude parameter is for ODIN-style input perturbation, which significantly affects score quality.

**Evidence:**
```python
def compute_gini_score(logits, temperature=1.0, magnitude=0.0, normalize=True):
    """Compute Gini impurity score (Doctor score)."""
    scaled_logits = logits / temperature  # Uses temperature
    probs = torch.softmax(scaled_logits, dim=-1)
    # magnitude is NEVER USED - no gradient-based perturbation applied!
```

**Impact:**
- Raw LDA gini seed 1 with T=1.2, mag=0.002: **fpr_test=0.4349** (no perturbation)
- Doctor-eval grid T=1.2, mag=0.002: **fpr_test=0.2335** (with perturbation)
- This is a ~2x difference in FPR due to missing perturbation
- Results are **not comparable** with Doctor baseline

**Root Cause:**
The ODIN perturbation requires:
1. Forward pass to compute loss
2. Backward pass to get input gradients
3. Perturb inputs in gradient direction
4. Second forward pass with perturbed inputs

The raw LDA script only does a single forward pass without gradient computation, making perturbation impossible with its current design.

**Fix Required:**
**Recommended approach:** Refactor `run_raw_lda_evaluation.py` to use the existing postprocessor implementations from `error_estimation.utils.postprocessors` instead of reimplementing score computation. This follows the code reuse principle documented in CLAUDE.md.

Alternative: Extend `run_detection.py` to support raw LDA evaluation as a new mode, avoiding the need for a separate script entirely.

**Status:** Not yet fixed - requires refactoring to use main codebase functions

---

## Issue #5: [RECLASSIFIED - NOT AN ISSUE] eval-grid Mode Selects by fpr_cal

**Date Identified:** 2026-01-18

**Date Reclassified:** 2026-01-18

**File:** `src/error_estimation/utils/results_io.py` - `select_best_row()` function

**Initial Concern:**
The `select_best_row()` function selects by `fpr_cal` when res column is not available, causing doctor-eval-grid to select different hyperparameters than doctor-res-grid.

**Evidence:**
- doctor-res-grid seed 1: selected **T=1.2** (best fpr_res)
- doctor-eval-grid seed 1: selected **T=0.8** (best fpr_cal)

**Why This Is Actually Correct:**

This behavior is **intentional** for the experimental design:

1. **Binning methods** use res split (1k samples) for hyperparameter selection, then fit on cal (4k samples)

2. **Raw score "full power" baseline** uses cal split (4k samples) for hyperparameter selection to show what's achievable when the raw score has access to the same amount of labeled data as binning methods

3. This enables a **fair comparison**: both approaches have access to the same calibration data. The "full power" baseline represents the upper bound of raw score performance with equivalent labeled data.

The different hyperparameters between res-based and cal-based selection are expected and serve different experimental purposes:
- **res-based (doctor-res-grid):** For methods that must select hyperparameters before seeing cal data (e.g., LDA binning)
- **cal-based (doctor-eval-grid):** For "full power" raw score baseline comparison

**Status:** Reclassified as intended behavior, not an issue

---

## Best Practices to Prevent Future Inconsistencies

1. **Per-seed hyperparameter selection is mandatory** - Hyperparameters must be selected independently for each seed on its res split, never shared across seeds.

2. **Always verify config loading** - When loading configs from previous experiments, print/log what was loaded to verify correctness.

3. **Use tools to copy values** - When reporting results, use tools to extract values directly from result files to avoid transcription errors.

4. **Cross-check report sections** - When adding new results, verify consistency with related sections in the report.

5. **Test locally before pushing** - Run scripts with limited scope locally to catch import/logic errors before server runs.

6. **Sort by modification time, not name** - When finding "most recent" directories, use `stat().st_mtime` not alphabetical sorting.

7. **Use consistent precision** - All baseline values should use the same decimal precision across all sections (prefer 4-decimal for metrics like ROC-AUC, FPR). When the same value appears in multiple sections, use identical formatting.

8. **Verify all parameters are used** - When implementing score functions with hyperparameters, verify each parameter is actually applied. If a function accepts `magnitude` for ODIN perturbation, it must actually apply the perturbation.

9. **Document selection split choice** - When selecting hyperparameters, clearly document which split is used (res or cal) and why. Different experiments may intentionally use different splits (e.g., res-based for binning methods, cal-based for "full power" raw score baselines).

10. **Test score computation against reference** - When implementing score computation, verify outputs match the reference implementation (e.g., Doctor postprocessor) on the same inputs before running full experiments.
