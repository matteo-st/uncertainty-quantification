# Experimental Procedure: Binned Error Detectors With Guarantees

Last updated: 2026-01-12

## Goal
Transform an existing uncertainty/error score into a calibrated error detector with statistical guarantees, while minimizing performance loss.

## Core setup
- **Datasets / models (default):** CIFAR-10/100 with ResNet-34. Add ImageNet if time permits.
- **Scores (baseline):** max_proba, gini; margin (negative margin for error direction).
- **Splits:** res / cal / test
  - **res**: choose binning + hyperparameters (no guarantees needed).
  - **cal**: estimate per-bin error and confidence intervals (guarantee split).
  - **test**: final evaluation only.
- **Seeds:** 1–9, report mean ± std.

If res is unavailable (small data), run a **cal-only protocol** and document that guarantees are weaker (same data used for binning and CI).

## Methods to compare
1) **Continuous baseline:** raw score (no binning).
2) **Uniform width bins:** fixed width in score space.
3) **Uniform mass bins:** quantile bins (watch for ties / collapsed bins).
4) **Isotonic-style binning:**
   - Fit isotonic regression on **res** to enforce monotonicity.
   - Option A: bin the isotonic outputs (uniform width or mass).
   - Option B: merge bins based on monotonic plateaus (n_min constraint).
   - Optional preprocessing: temperature scaling on **res** only.

## Bin outputs
For each bin, compute:
- **Empirical mean error** (point estimate).
- **Upper bound** of a CI (Hoeffding or Bernstein).

Compare **mean vs upper** as the score fed to ROC/FPR.

## Hyperparameter selection (no leakage)
Select all hyperparameters **without using cal or test**:
- K (number of bins), n_min (min count), preprocessing (temperature), isotonic options.
- Use **res** only:
  - Preferred: split res into res-train/res-val; select by ROC-AUC or FPR@95 on res-val.
  - Alternative: k-fold CV on res.

## Diagnostics to include
For each method and selected K:
- **ROC-AUC / FPR@95 vs K** (mean ± std across seeds).
- **CI vs bin center** (or bin index).
- **Bin width histogram** and **width vs CI half-width**.
- **Cal vs test counts per bin** (shift).
- **Non-empty bins** vs K (especially for uniform mass and max_proba).

Track ties explicitly:
- unique score count
- non-empty bin count (K effective)
- max bin count

## Metrics
Primary:
- ROC-AUC (test)
- FPR@95 (test)
Secondary:
- AURC, AUPR_in/out
- **Guarantee check:** fraction of bins where test mean error exceeds upper bound (should be rare if independence holds).

## Recommended experimental blocks
1) **Baseline block (2–3 days):**
   - Continuous score performance.
   - Uniform mass + uniform width, wide K grid.
2) **Guarantee block (3–4 days):**
   - Compare mean vs upper bound.
   - Verify CI coverage on test.
3) **Isotonic block (3–4 days):**
   - Isotonic variants + merging strategies.
4) **Sensitivity block (2–3 days):**
   - Temperature vs score ties.
   - Float precision effects (float16/32/64).

## Status update (what is done vs pending)
### Done
- **Continuous scores (temperature + magnitude grid):** MSP/Doctor/Margin, seeds 1–9, n_cal=5000. Report and plots in `docs/temperature_magnitude_report.md`.
- **Uniform-mass cal-only (gini space):** CIFAR-10/100 curves vs K, CI vs score diagnostics. Report in `docs/partition_binning_report.md`.
- **Server uniform-mass sweep (max_proba space):** n_cal=5000, seeds 1–9, K grid 2–5000 (see `docs/partition_binning_report.md` for paths).

### Pending
- **Uniform-width binning** baseline (same K grid and protocol).
- **Isotonic / monotone merging** variants (uniform-mass + merging, PAV or similar).
- **Guarantee checks** on test (CI coverage rate and violation rate).
- **Unified selection protocol** using res-based validation for K/alpha (beyond cal-only).

## Reporting format
For each dataset/model:
- Table with best test ROC-AUC/FPR@95 per method (mean ± std).
- Figure: test ROC-AUC/FPR vs K (mean ± std).
- Figure: CI vs bin index (best K).
- Short paragraph interpreting failure modes (ties, empty bins, shift).

## Reproducibility checklist
- Record seed splits, K grid, and all preprocessing choices.
- Keep res/cal/test strictly separated for selection vs CI vs test.
- Save per-seed CSVs and plots.

## Existing results (snapshot)
### Continuous scores (best roc_auc_cal)
From `docs/temperature_magnitude_report.md` (CIFAR-10 / ResNet-34, n_cal=5000, seeds 1–9).

| method | selection | temp | mag | fpr_test | roc_auc_test |
|---|---|---|---|---|---|
| msp | best roc_auc_cal | 0.700 ± 0.000 | 0.0000 ± 0.0000 | 0.2849 ± 0.0411 | 0.9291 ± 0.0031 |
| doctor | best roc_auc_cal | 0.756 ± 0.068 | 0.0020 ± 0.0000 | 0.1931 ± 0.0149 | 0.9378 ± 0.0033 |
| margin | best roc_auc_cal | 0.722 ± 0.042 | 0.0020 ± 0.0000 | 0.1893 ± 0.0155 | 0.9371 ± 0.0032 |

### Uniform-mass - max_proba
Source: server sweep in `/home/lamsade/msammut/error_detection/error-estimation/results_hyperparams/partition_unif-mass/cifar10_resnet34/n_cal-5000/seed-split-*/results_opt-fpr_qunatiz-metric-fpr-ratio-None_n-split-val-1_weight-std-0.0_mode-evaluation.csv`.
- Dataset/model: CIFAR-10 / ResNet-34
- Protocol: cal-only (n_res=0, n_cal=5000, n_test=5000), seeds 1–9
- Space: max_proba, bound: Hoeffding, score: mean or upper
- K grid: 2–20 (step 1), then 30–5000 (step 10)
- Local summary CSV (computed from the server files): `diagnostics_server/unif_mass_server/n_cal-5000/summary_best_by_cal_alpha0.5.csv`

| method | selection | k | fpr_test | roc_auc_test |
|---|---|---|---|---|
| unif-mass (mean, alpha=0.5) | best fpr_cal | 484.4 ± 10.7 | 1.0000 ± 0.0000 | 0.8557 ± 0.0113 |
| unif-mass (upper, alpha=0.5) | best fpr_cal | 322.2 ± 117.0 | 0.3386 ± 0.0118 | 0.9158 ± 0.0034 |
| unif-mass (mean, alpha=0.5) | best roc_auc_cal | 482.2 ± 12.3 | 1.0000 ± 0.0000 | 0.8546 ± 0.0136 |
| unif-mass (upper, alpha=0.5) | best roc_auc_cal | 4954.4 ± 43.2 | 0.2970 ± 0.0336 | 0.8944 ± 0.0048 |

Note: the server sweep logs alpha=0.5 for nearly all entries; alpha=0.05 appears only once in the CSVs, so it is not summarized here.

### Uniform-mass (gini, cal-only) from `uncertainty-quantification/results`
Source folder: `/home/lamsade/msammut/error_detection/uncertainty-quantification/results/partition_binning/calonly_alpha/` with per-seed CSVs:
- CIFAR-10: `cifar10/unif_mass_calonly_alpha_cifar10_per_seed.csv`
- CIFAR-100: `cifar100/unif_mass_calonly_alpha_cifar100_per_seed.csv`
- Continuous baselines: `*_continuous_per_seed.csv`

Selection below is **best-on-test** per seed across K (post-hoc oracle, for comparison only).
Cal-based selection is **not available** for these runs because the saved CSVs only include test metrics (no `*_cal` columns). To add cal-based selection here, we need a rerun that logs cal metrics per K.

Cal-based selection (pending):
- Required inputs missing in current outputs (`fpr_cal`, `roc_auc_cal`).
- Rerun needed to populate a cal-based table for these uncertainty-quantification results.

**CIFAR-10 / ResNet-34 (gini, temp=1.0, n_cal=5000, seeds 1–9)**  
| curve | selection | k | fpr_test | roc_auc_test |
|---|---|---|---|---|
| continuous | n/a | n/a | 0.3426 ± 0.0521 | 0.9256 ± 0.0035 |
| mean | best fpr_test | 9.4 ± 4.4 | 0.3840 ± 0.0867 | 0.9028 ± 0.0156 |
| upper_alpha=0.05 | best fpr_test | 9.4 ± 4.4 | 0.3840 ± 0.0867 | 0.9029 ± 0.0157 |
| upper_alpha=0.1 | best fpr_test | 9.4 ± 4.4 | 0.3840 ± 0.0867 | 0.9029 ± 0.0157 |
| upper_alpha=0.5 | best fpr_test | 9.4 ± 4.4 | 0.3840 ± 0.0867 | 0.9029 ± 0.0157 |
| mean | best roc_auc_test | 24.4 ± 5.0 | 0.4629 ± 0.1273 | 0.9169 ± 0.0073 |
| upper_alpha=0.05 | best roc_auc_test | 33.3 ± 24.0 | 0.5072 ± 0.1242 | 0.9172 ± 0.0072 |
| upper_alpha=0.1 | best roc_auc_test | 33.3 ± 24.0 | 0.5072 ± 0.1242 | 0.9172 ± 0.0072 |
| upper_alpha=0.5 | best roc_auc_test | 33.3 ± 24.0 | 0.5072 ± 0.1242 | 0.9172 ± 0.0072 |

**CIFAR-100 / ResNet-34 (gini, temp=1.0, n_cal=5000, seeds 1–9)**  
| curve | selection | k | fpr_test | roc_auc_test |
|---|---|---|---|---|
| continuous | n/a | n/a | 0.4158 ± 0.0150 | 0.8772 ± 0.0038 |
| mean | best fpr_test | 57.2 ± 23.8 | 0.4282 ± 0.0155 | 0.8721 ± 0.0040 |
| upper_alpha=0.05 | best fpr_test | 57.2 ± 23.8 | 0.4282 ± 0.0155 | 0.8721 ± 0.0040 |
| upper_alpha=0.1 | best fpr_test | 57.2 ± 23.8 | 0.4282 ± 0.0155 | 0.8721 ± 0.0040 |
| upper_alpha=0.5 | best fpr_test | 57.2 ± 23.8 | 0.4282 ± 0.0155 | 0.8721 ± 0.0040 |
| mean | best roc_auc_test | 21.1 ± 3.1 | 0.4500 ± 0.0243 | 0.8752 ± 0.0035 |
| upper_alpha=0.05 | best roc_auc_test | 21.1 ± 3.1 | 0.4500 ± 0.0243 | 0.8752 ± 0.0035 |
| upper_alpha=0.1 | best roc_auc_test | 21.1 ± 3.1 | 0.4500 ± 0.0243 | 0.8752 ± 0.0035 |
| upper_alpha=0.5 | best roc_auc_test | 21.1 ± 3.1 | 0.4500 ± 0.0243 | 0.8752 ± 0.0035 |

### Uniform-mass grid search (gini, cal-only) from local `results/partition_binning`
Source folders:
- CIFAR-10: `results/partition_binning/cifar10/resnet34_ce/partition/runs/unif-mass-grid-20260112/`
- CIFAR-100: `results/partition_binning/cifar100/resnet34_ce/partition/runs/unif-mass-grid-20260112/`

Protocol: cal-only (n_res=0, n_cal=5000, n_test=5000), seeds 1–9.  
Space: gini, bound: Hoeffding, score in {mean, upper}, alpha in {0.05, 0.1, 0.5}.  
K grid: 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500.

Note: for score=mean, alpha is not used (mean errors ignore interval width). Results are identical across alpha, so mean rows are reported once with `alpha=any`.

Selection is per-seed best (by cal or test), then mean ± std reported for K, FPR, and ROC-AUC.

**CIFAR-10 / ResNet-34**

Cal-based selection:
| method | selection | k | fpr_test | roc_auc_test |
|---|---|---|---|---|
| raw-score (gini) | n/a | 1.0 ± 0.0 | 0.3426 ± 0.0553 | 0.9256 ± 0.0037 |
| unif-mass (mean, alpha=any) | best fpr_cal | 500.0 ± 0.0 | 1.0000 ± 0.0000 | 0.8681 ± 0.0123 |
| unif-mass (upper, alpha=0.05) | best fpr_cal | 500.0 ± 0.0 | 0.8314 ± 0.0096 | 0.8670 ± 0.0118 |
| unif-mass (upper, alpha=0.1) | best fpr_cal | 500.0 ± 0.0 | 0.8314 ± 0.0096 | 0.8670 ± 0.0118 |
| unif-mass (upper, alpha=0.5) | best fpr_cal | 500.0 ± 0.0 | 0.8319 ± 0.0092 | 0.8673 ± 0.0120 |
| unif-mass (mean, alpha=any) | best roc_auc_cal | 500.0 ± 0.0 | 1.0000 ± 0.0000 | 0.8681 ± 0.0123 |
| unif-mass (upper, alpha=0.05) | best roc_auc_cal | 500.0 ± 0.0 | 0.8314 ± 0.0096 | 0.8670 ± 0.0118 |
| unif-mass (upper, alpha=0.1) | best roc_auc_cal | 500.0 ± 0.0 | 0.8314 ± 0.0096 | 0.8670 ± 0.0118 |
| unif-mass (upper, alpha=0.5) | best roc_auc_cal | 500.0 ± 0.0 | 0.8319 ± 0.0092 | 0.8673 ± 0.0120 |

Test-based selection (oracle):
| method | selection | k | fpr_test | roc_auc_test |
|---|---|---|---|---|
| raw-score (gini) | n/a | 1.0 ± 0.0 | 0.3426 ± 0.0553 | 0.9256 ± 0.0037 |
| unif-mass (mean, alpha=any) | best fpr_test | 11.7 ± 8.3 | 0.3967 ± 0.1079 | 0.9033 ± 0.0181 |
| unif-mass (upper, alpha=0.05) | best fpr_test | 11.7 ± 8.3 | 0.3967 ± 0.1079 | 0.9033 ± 0.0180 |
| unif-mass (upper, alpha=0.1) | best fpr_test | 11.7 ± 8.3 | 0.3967 ± 0.1079 | 0.9033 ± 0.0180 |
| unif-mass (upper, alpha=0.5) | best fpr_test | 11.7 ± 8.3 | 0.3967 ± 0.1079 | 0.9033 ± 0.0180 |
| unif-mass (mean, alpha=any) | best roc_auc_test | 24.4 ± 5.3 | 0.4403 ± 0.1139 | 0.9173 ± 0.0082 |
| unif-mass (upper, alpha=0.05) | best roc_auc_test | 25.6 ± 5.3 | 0.4391 ± 0.1123 | 0.9174 ± 0.0080 |
| unif-mass (upper, alpha=0.1) | best roc_auc_test | 25.6 ± 5.3 | 0.4391 ± 0.1123 | 0.9174 ± 0.0080 |
| unif-mass (upper, alpha=0.5) | best roc_auc_test | 25.6 ± 5.3 | 0.4391 ± 0.1123 | 0.9174 ± 0.0080 |

**CIFAR-100 / ResNet-34**

Cal-based selection:
| method | selection | k | fpr_test | roc_auc_test |
|---|---|---|---|---|
| raw-score (gini) | n/a | 1.0 ± 0.0 | 0.4158 ± 0.0159 | 0.8772 ± 0.0040 |
| unif-mass (mean, alpha=any) | best fpr_cal | 258.3 ± 84.8 | 0.6155 ± 0.1529 | 0.8603 ± 0.0067 |
| unif-mass (upper, alpha=0.05) | best fpr_cal | 258.3 ± 84.8 | 0.5814 ± 0.0660 | 0.8602 ± 0.0065 |
| unif-mass (upper, alpha=0.1) | best fpr_cal | 258.3 ± 84.8 | 0.5814 ± 0.0660 | 0.8603 ± 0.0065 |
| unif-mass (upper, alpha=0.5) | best fpr_cal | 258.3 ± 84.8 | 0.5814 ± 0.0660 | 0.8606 ± 0.0064 |
| unif-mass (mean, alpha=any) | best roc_auc_cal | 500.0 ± 0.0 | 0.9472 ± 0.1585 | 0.8445 ± 0.0059 |
| unif-mass (upper, alpha=0.05) | best roc_auc_cal | 500.0 ± 0.0 | 0.9391 ± 0.1556 | 0.8421 ± 0.0061 |
| unif-mass (upper, alpha=0.1) | best roc_auc_cal | 500.0 ± 0.0 | 0.9391 ± 0.1556 | 0.8436 ± 0.0061 |
| unif-mass (upper, alpha=0.5) | best roc_auc_cal | 500.0 ± 0.0 | 0.9391 ± 0.1556 | 0.8443 ± 0.0058 |

Test-based selection (oracle):
| method | selection | k | fpr_test | roc_auc_test |
|---|---|---|---|---|
| raw-score (gini) | n/a | 1.0 ± 0.0 | 0.4158 ± 0.0159 | 0.8772 ± 0.0040 |
| unif-mass (mean, alpha=any) | best fpr_test | 57.2 ± 25.3 | 0.4282 ± 0.0165 | 0.8721 ± 0.0042 |
| unif-mass (upper, alpha=0.05) | best fpr_test | 57.2 ± 25.3 | 0.4282 ± 0.0165 | 0.8721 ± 0.0043 |
| unif-mass (upper, alpha=0.1) | best fpr_test | 57.2 ± 25.3 | 0.4282 ± 0.0165 | 0.8721 ± 0.0043 |
| unif-mass (upper, alpha=0.5) | best fpr_test | 57.2 ± 25.3 | 0.4282 ± 0.0165 | 0.8721 ± 0.0043 |
| unif-mass (mean, alpha=any) | best roc_auc_test | 21.1 ± 3.3 | 0.4500 ± 0.0258 | 0.8752 ± 0.0037 |
| unif-mass (upper, alpha=0.05) | best roc_auc_test | 21.1 ± 3.3 | 0.4500 ± 0.0258 | 0.8752 ± 0.0037 |
| unif-mass (upper, alpha=0.1) | best roc_auc_test | 21.1 ± 3.3 | 0.4500 ± 0.0258 | 0.8752 ± 0.0037 |
| unif-mass (upper, alpha=0.5) | best roc_auc_test | 21.1 ± 3.3 | 0.4500 ± 0.0258 | 0.8752 ± 0.0037 |
