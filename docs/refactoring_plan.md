# Refactoring Plan: Clean Modular Architecture

## Current Issues

1. **`run_detection.py` has grown complex** - Contains ~1000+ lines with duplicated logic for loading latents, perturbation, evaluation
2. **`run_raw_lda_evaluation.py` reimplemented score computation** - Didn't use existing `AblationDetector` which already handles perturbation
3. **Perturbation logic duplicated** - Both `run_detection.py` and `AblationDetector.get_pertubated_scores()` implement ODIN perturbation

## Existing Architecture (Good Parts to Keep)

```
evaluators.py
├── EvaluatorAblation          # Orchestrates experiment workflow
│   ├── Uses AblationDetector for each split (res, cal, test)
│   ├── Handles grid search, CV, etc.
│   └── Calls postprocessor.fit() and evaluation

eval.py
├── AblationDetector           # Evaluates on a single split
│   ├── get_pertubated_scores() # ODIN perturbation - CORRECT implementation
│   ├── get_scores()           # Gets scores for multiple detectors
│   └── evaluate()             # Computes metrics

postprocessors/
├── DoctorPostprocessor        # Computes gini score from logits
├── ODINPostprocessor          # Computes MSP score from logits
├── MarginPostprocessor        # Computes margin score from logits
├── LDABinningPostprocessor    # Combines scores + binning
└── ...
```

## Proposed Changes

### 1. Add Per-Score Perturbation to `AblationDetector`

**File:** `src/error_estimation/utils/eval.py`

Add a method to compute multiple scores with different perturbation settings:

```python
def get_multi_score_with_perturbation(
    self,
    score_configs: Dict[str, Dict],  # {score_name: {temperature, magnitude, normalize}}
    postprocessor_cls_map: Dict[str, type],  # {score_name: PostprocessorClass}
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Compute multiple scores where each score can have different perturbation.

    Returns:
        scores_dict: {score_name: np.ndarray of scores}
        detector_labels: np.ndarray of binary labels
    """
    # For each score:
    #   - Create temporary postprocessor with that score's config
    #   - Use get_pertubated_scores() if magnitude > 0
    #   - Use direct logits computation if magnitude = 0
```

### 2. Create `RawLDAPostprocessor`

**File:** `src/error_estimation/utils/postprocessors/raw_lda_postprocessor.py`

```python
class RawLDAPostprocessor(BasePostprocessor):
    """
    Raw LDA score combination WITHOUT binning.

    Workflow:
    1. fit_lda(res_evaluator) - Fit LDA on res split with per-score perturbation
    2. __call__(logits) - Apply LDA projection (for quick evaluation without perturbation)
    3. evaluate_with_perturbation(evaluator) - Full evaluation with perturbation
    """

    def fit_lda(self, evaluator: AblationDetector):
        """Fit LDA using AblationDetector's perturbation support."""
        scores_dict, labels = evaluator.get_multi_score_with_perturbation(
            score_configs=self.score_configs,
            postprocessor_cls_map=self.postprocessor_cls_map,
        )
        # Stack and fit LDA
        X = np.column_stack([scores_dict[name] for name in self.base_scores])
        self.lda.fit(X, labels)

    def evaluate_with_perturbation(self, evaluator: AblationDetector):
        """Evaluate on a split using proper perturbation."""
        scores_dict, labels = evaluator.get_multi_score_with_perturbation(...)
        X = np.column_stack([scores_dict[name] for name in self.base_scores])
        lda_scores = self.lda.transform(X).ravel()
        return lda_scores, labels
```

### 3. Simplify `run_detection.py`

Move complex logic to appropriate classes:

**Before (run_detection.py):**
```python
# 100+ lines of _load_latent_values()
# 200+ lines of _evaluate_grid()
# Duplicated perturbation logic
```

**After (run_detection.py):**
```python
def main():
    # Parse args, load configs
    evaluator = EvaluatorAblation(...)
    evaluator.run()
    # Save results
```

Most logic moved to `EvaluatorAblation` and `AblationDetector`.

### 4. Update `LDABinningPostprocessor` to Use Same Pattern

Ensure consistency between `RawLDAPostprocessor` and `LDABinningPostprocessor`:

```python
class LDABinningPostprocessor(RawLDAPostprocessor):
    """Extends RawLDA with binning."""

    def fit(self, logits, detector_labels, cal_evaluator=None, **kwargs):
        # LDA already fitted on res via fit_lda()
        # Now fit binning on cal
        if cal_evaluator:
            lda_scores, labels = self.evaluate_with_perturbation(cal_evaluator)
        else:
            lda_scores = self._apply_lda(logits)
            labels = detector_labels
        self._fit_binning(lda_scores, labels)
```

## Implementation Steps

### Step 1: Create git branch
```bash
git checkout -b refactor/modular-architecture
```

### Step 2: Add `get_multi_score_with_perturbation()` to `AblationDetector`
- Reuses existing `get_pertubated_scores()`
- Test that existing Doctor/ODIN evaluations still work

### Step 3: Create `RawLDAPostprocessor`
- Uses `AblationDetector` for perturbation
- Test with simple config

### Step 4: Refactor `LDABinningPostprocessor`
- Inherit from or share code with `RawLDAPostprocessor`
- Test that LDA binning results don't change

### Step 5: Simplify `run_detection.py`
- Move `_load_latent_values()` logic to `AblationDetector`
- Move `_evaluate_grid()` logic to `EvaluatorAblation`
- Keep `run_detection.py` as thin entry point

### Step 6: Verify results don't change
```bash
# Run reference experiment
python -m error_estimation.experiments.run_detection \
  --config-detection configs/postprocessors/doctor/cifar10_resnet34_hyperparams_search.yml \
  --mode search_res --seed-splits 1 --run-tag "test-before-refactor"

# Compare with after refactor
# Results should be identical
```

## Files to Modify

| File | Change |
|------|--------|
| `src/error_estimation/utils/eval.py` | Add `get_multi_score_with_perturbation()` |
| `src/error_estimation/utils/postprocessors/raw_lda_postprocessor.py` | NEW - Raw LDA postprocessor |
| `src/error_estimation/utils/postprocessors/lda_binning_postprocessor.py` | Refactor to use shared code |
| `src/error_estimation/utils/postprocessors/__init__.py` | Register new postprocessor |
| `src/error_estimation/experiments/run_detection.py` | Simplify, delegate to classes |
| `src/error_estimation/evaluators.py` | Minor updates if needed |
| `scripts/run_raw_lda_evaluation.py` | DELETE - replaced by new postprocessor |

## Rollback Strategy

1. All changes on feature branch `refactor/modular-architecture`
2. Main branch unchanged until verified
3. Each step is a separate commit for easy bisect
4. Run validation tests after each step

## Success Criteria

1. `run_detection.py` is < 500 lines (currently ~1000+)
2. Doctor/ODIN grid search produces identical results
3. LDA binning produces identical results
4. Raw LDA evaluation works correctly with perturbation
5. No code duplication between `run_detection.py` and evaluator classes
