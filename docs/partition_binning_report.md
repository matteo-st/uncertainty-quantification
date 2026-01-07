# Partition Binning Investigation Report

Last updated: 2026-01-07

## Objective
Understand how 1D binning strategies (uniform-mass and quantile-merge) affect detection performance relative to the continuous score, and identify why performance drops.

## Experimental setup
- Dataset: CIFAR-10
- Model: ResNet-34 (preprocessor: `ce`)
- Score: Gini (doctor-style), temperature = 1.0
- Split: res = 3000, cal = 2000, test = 5000, seed-split = 9
- Metric used for selection: ROC-AUC on res (search_res)
- Guarantee: bins learned on res, confidence intervals on cal, evaluation on test

## Notation and parameters
- `K`: number of bins (clusters) before any merge.
- `n_min`: minimum per-bin count on res; bins below this are merged into neighbors (quantile-merge).
- res/cal/test: resolution split for binning, calibration split for CIs, and test for evaluation.
- FPR@95: false positive rate at 95% true positive rate on the test split.

## Methods compared
- Continuous score (no binning)
- Uniform-mass (quantile bins)
- Quantile-merge (uniform-mass + min count per bin)

## Runs and artifacts (server paths)
- Quantile-merge diagnostics (k=50, n_min=50):  
  `results/cifar10/resnet34_ce/partition/runs/quantile-merge-diagnostics/seed-split-9/diagnostics/`
- Quantile-merge ablation (grid search):  
  `results/cifar10/resnet34_ce/partition/runs/quantile-merge-ablation/seed-split-9/diagnostics/`
- Uniform-mass ablation (grid search):  
  `results/cifar10/resnet34_ce/partition/runs/unif-mass-ablation/seed-split-9/diagnostics/`
- Quantile-merge ablation (search grid saved):  
  `results/cifar10/resnet34_ce/partition/runs/quantile-merge-ablation-search/seed-split-9/diagnostics/`
- Uniform-mass ablation (search grid saved):  
  `results/cifar10/resnet34_ce/partition/runs/unif-mass-ablation-search/seed-split-9/diagnostics/`

## Summary metrics (test set)

| Method | Bin config | ROC-AUC | FPR@95 | AUPR_in | AUPR_out |
| --- | --- | --- | --- | --- | --- |
| Continuous | n/a | 0.9219 | 0.3560 | 0.4000 | 0.9952 |
| Quantile-merge | k=50, n_min=50 | 0.8990 | 0.5434 | 0.3623 | 0.9921 |
| Quantile-merge (best grid) | k=100, n_min=1 | 0.8956 | 0.4063 | 0.3445 | 0.9920 |
| Uniform-mass (best grid) | k=100 | 0.8956 | 0.4063 | 0.3445 | 0.9920 |

Notes:
- "Continuous" is the raw 1D score (no binning).
- "Best grid" refers to selection by ROC-AUC on the res split.

## Initial observations
- Binning reduces ROC-AUC and increases FPR@95 relative to the continuous score.
- Quantile-merge with n_min=50 trades ROC-AUC for a much worse FPR@95.
- When n_min=1, quantile-merge reduces to uniform-mass and yields the same metrics.
- Continuous scoring remains the strongest for ranking metrics.

## Interpretation guide for diagnostics plots
Use these plots to diagnose where binning hurts performance and why:
- `ci_vs_score.pdf`: confidence intervals and bin means vs score center. Look for wide CIs near the decision threshold (large uncertainty where it matters).
- `width_vs_halfwidth.pdf`: bin width vs CI half-width. Wide bins with large half-widths indicate poor resolution and high variance.
- `bin_width_hist.pdf`: distribution of bin widths. Heavy tails imply many coarse bins (often in score extremes).
- `count_shift.pdf`: cal vs test counts per bin. Large deviations suggest distribution shift that can inflate uncertainty.

## Evidence summary from plots (to read alongside the figures)
- The large FPR@95 jump at n_min=50 suggests the merge step likely removes resolution around the operating region; verify in `ci_vs_score.pdf` by checking how many merged bins span the threshold.
- The gap between continuous and binned ROC-AUC is consistent with fewer distinct score levels; check `bin_width_hist.pdf` for a heavy tail and `width_vs_halfwidth.pdf` for wide bins with large CI half-widths.
- The best K (100) improves ROC-AUC on res, indicating that finer binning helps ranking; see `grid_roc_auc_res.pdf` and `curve_roc_auc_res.pdf`.

## Grid search highlights (res split)
Quantile-merge grid (ROC-AUC on res):
- Best: n_clusters=100, n_min in {1, 20} with roc_auc_res ≈ 0.9472 and fpr_res ≈ 0.2561.
- Increasing n_min to 50 or 100 does not improve roc_auc_res; it usually shifts fpr_res upward for the same K.

Uniform-mass grid (ROC-AUC on res):
- ROC-AUC improves with K; best at K=100 with roc_auc_res ≈ 0.9472 and fpr_res ≈ 0.2561.
- K=50 is slightly worse (roc_auc_res ≈ 0.9404).

Top quantile-merge configurations (res split):
| n_clusters | n_min | roc_auc_res | fpr_res |
| --- | --- | --- | --- |
| 100 | 1 | 0.9472 | 0.2561 |
| 100 | 20 | 0.9472 | 0.2561 |
| 50 | 20 | 0.9404 | 0.3505 |
| 50 | 50 | 0.9404 | 0.3505 |

Uniform-mass (res split):
| n_clusters | roc_auc_res | fpr_res |
| --- | --- | --- |
| 100 | 0.9472 | 0.2561 |
| 50 | 0.9404 | 0.3505 |
| 20 | 0.9288 | 0.3211 |
| 10 | 0.9147 | 0.3732 |

## Plots generated (server paths)
Each diagnostics folder contains:
- `ci_vs_score.pdf`
- `width_vs_halfwidth.pdf`
- `bin_width_hist.pdf`
- `count_shift.pdf`

Grid plots (saved in the search diagnostics folders):
- `grid_roc_auc_res.pdf` and `grid_fpr_res.pdf` for quantile-merge
- `curve_roc_auc_res.pdf` and `curve_fpr_res.pdf` for uniform-mass

How to read the plots:
- `ci_vs_score.pdf`: lower/upper CI and bin means vs score center.
- `width_vs_halfwidth.pdf`: shows the resolution/variance trade-off.
- `bin_width_hist.pdf`: highlights very wide bins (often tails).
- `count_shift.pdf`: checks stability between cal and test counts.

## Next steps (analysis plan)
1) Inspect per-bin diagnostics to see if loss is concentrated in tails (wide bins).
2) Compare bin-width distributions across K and n_min (use `bin_width_hist.pdf`).
3) Extend the grid to K=200 for sensitivity (if runtime allows).
