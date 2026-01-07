# Uniform-Mass Binning Investigation Report

Last updated: 2026-01-07

## Objective
Understand how uniform-mass binning affects detection performance relative to the continuous score, and identify why performance drops across CIFAR-10 and CIFAR-100.

## Experimental setup
- Datasets: CIFAR-10, CIFAR-100
- Model: ResNet-34 (preprocessor: `ce`)
- Score: Gini (doctor-style), temperature = 1.0
- Split: res = 3000, cal = 2000, test = 5000, seed-split = 9
- Metric used for selection: ROC-AUC on res (selection never uses cal)
- Guarantee: bins learned on res, confidence intervals on cal, evaluation on test

## Notation and parameters
- `K`: number of uniform-mass bins (quantile bins).
- res/cal/test: resolution split for binning, calibration split for CIs, and test for evaluation.
- FPR@95: false positive rate at 95% true positive rate on the test split.

## Methods compared
- Continuous score (no binning)
- Uniform-mass (quantile bins)

## Runs and artifacts (server paths)
- CIFAR-10 expanded grid (K up to 500):  
  `results/cifar10/resnet34_ce/partition/runs/unif-mass-grid-v2/seed-split-9/`
- CIFAR-100 expanded grid (K up to 500):  
  `results/cifar100/resnet34_ce/partition/runs/unif-mass-grid-v2/seed-split-9/`

## Summary metrics (test set)

### CIFAR-10 / ResNet-34
| Method | Bin config | ROC-AUC | FPR@95 | AUPR_in | AUPR_out |
| --- | --- | --- | --- | --- | --- |
| Continuous | n/a | 0.9219 | 0.3560 | 0.4000 | 0.9952 |
| Uniform-mass (best grid v2) | k=500 | 0.7447 | 0.7259 | 0.1072 | 0.9803 |
| Uniform-mass (CV grid) | k=20 | 0.9001 | 0.6188 | 0.3455 | 0.9919 |

### CIFAR-100 / ResNet-34
| Method | Bin config | ROC-AUC | FPR@95 | AUPR_in | AUPR_out |
| --- | --- | --- | --- | --- | --- |
| Uniform-mass (best grid v2) | k=500 | 0.7485 | 0.7732 | 0.3921 | 0.8977 |
| Uniform-mass (CV grid) | k=20 | 0.8644 | 0.4884 | 0.6007 | 0.9523 |

Notes:
- "Continuous" is the raw 1D score (no binning).
- "Best grid" is selected by ROC-AUC on the res split; the table reports test metrics only.

## Hyperparameter selection protocol (K)
To preserve the iid calibration guarantee, cal is never used for selection:
1) Split data into res/cal/test (cal is held out for CI construction only).
2) For each candidate K, build uniform-mass bins using res scores.
3) Compute selection metric on res only:
   - Preferred: split res into res-train/res-val and select K by res-val ROC-AUC.
   - Alternative: use all res for selection (more optimistic; still does not touch cal).
4) With K fixed, rebuild bins on full res.
5) Build Hoeffding CIs on cal.
6) Evaluate final metrics on test only.

## Initial observations
- Uniform-mass binning reduces ROC-AUC and increases FPR@95 relative to the continuous score (CIFAR-10).
- Res ROC-AUC improves with larger K, but test ROC-AUC degrades sharply for K=500, indicating overfitting to res.
- CIFAR-100 shows a similar pattern: best res ROC-AUC at K=500, but weak test ROC-AUC and very high FPR@95.
- Cross-validation selection picks smaller K and improves test ROC-AUC and FPR@95 on both datasets.
- Test curves show different optima by metric (CIFAR-10: best ROC-AUC at K=20, best FPR at K=100; CIFAR-100: best ROC-AUC at K=10, best FPR at K=20).

## Interpretation guide for diagnostics plots
Use these plots to diagnose where binning hurts performance and why:
- CI vs score: confidence intervals and bin means vs score center. The x-axis uses quantile bin centers (uniform-mass), so near-uniform spacing is expected and does not imply a uniform score distribution.
- Width vs half-width: bin width vs CI half-width. Wide bins with large half-widths indicate poor resolution and high variance.
- Bin width histogram: distribution of bin widths. Heavy tails imply many coarse bins (often in score extremes).
- Count shift: cal vs test counts per bin. Large deviations suggest distribution shift that can inflate uncertainty.

## Evidence summary from plots (to read alongside the figures)
- The gap between continuous and binned ROC-AUC is consistent with fewer distinct score levels; check the bin width histogram and width vs half-width for wide bins with large CI half-widths.
- Larger K improves ROC-AUC on res, but this does not transfer to test; see the ROC-AUC vs K curves.

## Grid search highlights (res split)
Uniform-mass (res split, used only for selection; not a performance report):

### CIFAR-10 / ResNet-34
| n_clusters | roc_auc_res |
| --- | --- |
| 10 | 0.9147 |
| 20 | 0.9288 |
| 50 | 0.9404 |
| 100 | 0.9472 |
| 200 | 0.9543 |
| 300 | 0.9573 |
| 500 | 0.9655 |

### CIFAR-100 / ResNet-34
| n_clusters | roc_auc_res |
| --- | --- |
| 10 | 0.8820 |
| 20 | 0.8866 |
| 50 | 0.8891 |
| 100 | 0.8949 |
| 200 | 0.9010 |
| 300 | 0.9056 |
| 500 | 0.9112 |

## Figures (embedded)
These figures are rendered from the downloaded diagnostics CSVs for readability.

### Continuous score distribution
![Continuous score distribution (test)](partition_binning_assets/score_distribution_test.png)

### Uniform-mass (best grid)
![Uniform-mass ablation: CI vs score](partition_binning_assets/unif-mass-ablation_ci_vs_score.png)
![Uniform-mass ablation: width vs half-width](partition_binning_assets/unif-mass-ablation_width_vs_halfwidth.png)
![Uniform-mass ablation: bin width histogram](partition_binning_assets/unif-mass-ablation_bin_width_hist.png)
![Uniform-mass ablation: cal vs test counts](partition_binning_assets/unif-mass-ablation_count_shift.png)

### Grid search summaries (res split)
![Uniform-mass CIFAR-10: ROC-AUC](partition_binning_assets/unif_mass_grid_v2_cifar10_roc_auc_res.png)
![Uniform-mass CIFAR-10: FPR@95](partition_binning_assets/unif_mass_grid_v2_cifar10_fpr_res.png)
![Uniform-mass CIFAR-100: ROC-AUC](partition_binning_assets/unif_mass_grid_v2_cifar100_roc_auc_res.png)
![Uniform-mass CIFAR-100: FPR@95](partition_binning_assets/unif_mass_grid_v2_cifar100_fpr_res.png)

### Grid search summaries (cross-validation)
![Uniform-mass CIFAR-10: ROC-AUC (CV)](partition_binning_assets/unif_mass_grid_cv_cifar10_roc_auc_val_cross.png)
![Uniform-mass CIFAR-10: FPR@95 (CV)](partition_binning_assets/unif_mass_grid_cv_cifar10_fpr_val_cross.png)
![Uniform-mass CIFAR-100: ROC-AUC (CV)](partition_binning_assets/unif_mass_grid_cv_cifar100_roc_auc_val_cross.png)
![Uniform-mass CIFAR-100: FPR@95 (CV)](partition_binning_assets/unif_mass_grid_cv_cifar100_fpr_val_cross.png)

### Test performance vs K (bins on res, CIs on cal)
![Uniform-mass CIFAR-10: ROC-AUC (test) vs K](partition_binning_assets/unif_mass_test_curve_cifar10_roc_auc.png)
![Uniform-mass CIFAR-10: FPR@95 (test) vs K](partition_binning_assets/unif_mass_test_curve_cifar10_fpr.png)
![Uniform-mass CIFAR-100: ROC-AUC (test) vs K](partition_binning_assets/unif_mass_test_curve_cifar100_roc_auc.png)
![Uniform-mass CIFAR-100: FPR@95 (test) vs K](partition_binning_assets/unif_mass_test_curve_cifar100_fpr.png)

## Limitations of the current procedure
- Selection bias: `search_res` uses the same res split to build bins and to select K, which is optimistic and can overfit to res.
- Selection metric mismatch: ROC-AUC on res does not account for calibration tightness or the guarantee objective; it can prefer large K even when CIs become unstable.
- Finite-sample effects: with large K, per-bin counts shrink on cal, widening Hoeffding intervals and degrading FPR@95 on test.
- Single-seed evidence: results are for seed-split 9 only; variance across splits is unknown.
- Limited baselines: CIFAR-100 continuous-score baseline is not yet computed, so relative loss is not quantified there.

## Next steps (analysis plan)
1) Inspect per-bin diagnostics to see if loss is concentrated in tails (wide bins).
2) Compare bin-width distributions across K (see the bin width histogram).
3) Re-run selection with res-train/res-val (search_val) to reduce selection bias.
4) Add a continuous-score baseline for CIFAR-100.
5) Repeat for multiple seed splits to quantify variance.
