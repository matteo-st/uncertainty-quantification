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

### Uniform-mass CV selection (gini, cal split) from server `results/partition_binning`
Source folders:
- CIFAR-10: `results/partition_binning/cifar10/resnet34_ce/partition/runs/unif-mass-cv-20260114c/`
- CIFAR-100: `results/partition_binning/cifar100/resnet34_ce/partition/runs/unif-mass-cv-20260114c/`

Protocol: cal split (n_res=0, n_cal=5000, n_test=5000), seeds 1–9.  
Space: gini, bound: Hoeffding, score in {mean, upper}, alpha in {0.05, 0.1, 0.5}.  
K grid: 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500.

Reported values are cross-validated metrics from `search.jsonl` (val splits). Test metrics are taken from `unif-mass-grid-20260112/seed-split-*/grid_results.csv` by matching seed + selected K (and score/alpha); for score=mean the test metrics are averaged across alphas. Selection is per-seed best by `fpr_val_cross` or `roc_auc_val_cross`, then mean ± std for K, FPR, and ROC-AUC.

Note: for score=mean, alpha is not used (mean errors ignore interval width). Results are identical across alpha, so mean rows are reported once with `alpha=any`.
Raw-score baseline uses `raw-score-20260112` with gini, score=upper, temperature=1.0, magnitude=0.0.

**CIFAR-10 / ResNet-34**

CV-based selection:
| method | selection | k | fpr_val_cross | roc_auc_val_cross | fpr_test | roc_auc_test |
|---|---|---|---|---|---|---|
| raw-score (gini, upper, temp=1.0, mag=0.0) | n/a | 1.0 ± 0.0 | n/a | n/a | 0.3426 ± 0.0521 | 0.9256 ± 0.0035 |
| unif-mass (mean, alpha=any) | best fpr_val_cross | 11.7 ± 4.7 | 0.4170 ± 0.0567 | 0.9081 ± 0.0116 | 0.4931 ± 0.1232 | 0.9059 ± 0.0097 |
| unif-mass (mean, alpha=any) | best roc_auc_val_cross | 24.4 ± 5.0 | 0.4816 ± 0.0747 | 0.9154 ± 0.0082 | 0.5502 ± 0.2549 | 0.9134 ± 0.0123 |
| unif-mass (upper, alpha=0.05) | best fpr_val_cross | 17.2 ± 12.7 | 0.4128 ± 0.0573 | 0.9084 ± 0.0116 | 0.5194 ± 0.1749 | 0.9072 ± 0.0105 |
| unif-mass (upper, alpha=0.05) | best roc_auc_val_cross | 23.3 ± 4.7 | 0.4499 ± 0.0607 | 0.9153 ± 0.0076 | 0.4991 ± 0.1920 | 0.9132 ± 0.0122 |
| unif-mass (upper, alpha=0.1) | best fpr_val_cross | 17.2 ± 12.7 | 0.4128 ± 0.0573 | 0.9084 ± 0.0116 | 0.5194 ± 0.1749 | 0.9072 ± 0.0105 |
| unif-mass (upper, alpha=0.1) | best roc_auc_val_cross | 23.3 ± 4.7 | 0.4499 ± 0.0607 | 0.9153 ± 0.0076 | 0.4991 ± 0.1920 | 0.9132 ± 0.0122 |
| unif-mass (upper, alpha=0.5) | best fpr_val_cross | 17.2 ± 12.7 | 0.4128 ± 0.0573 | 0.9084 ± 0.0116 | 0.5194 ± 0.1749 | 0.9072 ± 0.0105 |
| unif-mass (upper, alpha=0.5) | best roc_auc_val_cross | 23.3 ± 4.7 | 0.4499 ± 0.0607 | 0.9153 ± 0.0076 | 0.4991 ± 0.1920 | 0.9132 ± 0.0122 |

**CIFAR-100 / ResNet-34**

CV-based selection:
| method | selection | k | fpr_val_cross | roc_auc_val_cross | fpr_test | roc_auc_test |
|---|---|---|---|---|---|---|
| raw-score (gini, upper, temp=1.0, mag=0.0) | n/a | 1.0 ± 0.0 | n/a | n/a | 0.4158 ± 0.0150 | 0.8772 ± 0.0038 |
| unif-mass (mean, alpha=any) | best fpr_val_cross | 31.1 ± 11.0 | 0.4543 ± 0.0278 | 0.8722 ± 0.0036 | 0.4444 ± 0.0253 | 0.8738 ± 0.0042 |
| unif-mass (mean, alpha=any) | best roc_auc_val_cross | 23.3 ± 4.7 | 0.4680 ± 0.0265 | 0.8733 ± 0.0035 | 0.4500 ± 0.0238 | 0.8744 ± 0.0036 |
| unif-mass (upper, alpha=0.05) | best fpr_val_cross | 31.1 ± 11.0 | 0.4543 ± 0.0278 | 0.8722 ± 0.0036 | 0.4444 ± 0.0253 | 0.8738 ± 0.0042 |
| unif-mass (upper, alpha=0.05) | best roc_auc_val_cross | 23.3 ± 4.7 | 0.4680 ± 0.0265 | 0.8732 ± 0.0035 | 0.4500 ± 0.0238 | 0.8744 ± 0.0036 |
| unif-mass (upper, alpha=0.1) | best fpr_val_cross | 31.1 ± 11.0 | 0.4543 ± 0.0278 | 0.8722 ± 0.0036 | 0.4444 ± 0.0253 | 0.8738 ± 0.0042 |
| unif-mass (upper, alpha=0.1) | best roc_auc_val_cross | 23.3 ± 4.7 | 0.4680 ± 0.0265 | 0.8732 ± 0.0035 | 0.4500 ± 0.0238 | 0.8744 ± 0.0036 |
| unif-mass (upper, alpha=0.5) | best fpr_val_cross | 31.1 ± 11.0 | 0.4543 ± 0.0278 | 0.8722 ± 0.0036 | 0.4444 ± 0.0253 | 0.8738 ± 0.0042 |
| unif-mass (upper, alpha=0.5) | best roc_auc_val_cross | 23.3 ± 4.7 | 0.4680 ± 0.0265 | 0.8732 ± 0.0035 | 0.4500 ± 0.0238 | 0.8744 ± 0.0036 |

### Rule-of-thumb K selection (cube-root, Rice) from local `results/partition_binning`
Source folders:
- CIFAR-10: `results/partition_binning/cifar10/resnet34_ce/partition/runs/unif-mass-grid-20260112/`
- CIFAR-100: `results/partition_binning/cifar100/resnet34_ce/partition/runs/unif-mass-grid-20260112/`

Rules computed from n_cal=5000. Targets: K_cube = n^(1/3) = 17.1 and K_Rice = 2 n^(1/3) = 34.2.  
Nearest grid K values used: 20 for cube-root and 30 for Rice. Test metrics are taken from `grid_results.csv` at the selected K; for score=mean the test metrics are averaged across alpha (identical across alpha in practice).

**CIFAR-10 / ResNet-34**

| rule | method | k | fpr_test | roc_auc_test |
|---|---|---|---|---|
| n/a | raw-score (gini, upper, temp=1.0, mag=0.0) | 1.0 ± 0.0 | 0.3426 ± 0.0521 | 0.9256 ± 0.0035 |
| cube-root (K = n^(1/3)) | unif-mass (mean, alpha=any) | 20.0 ± 0.0 | 0.4934 ± 0.1400 | 0.9152 ± 0.0074 |
| cube-root (K = n^(1/3)) | unif-mass (upper, alpha=0.05) | 20.0 ± 0.0 | 0.4934 ± 0.1400 | 0.9151 ± 0.0076 |
| cube-root (K = n^(1/3)) | unif-mass (upper, alpha=0.1) | 20.0 ± 0.0 | 0.4934 ± 0.1400 | 0.9151 ± 0.0076 |
| cube-root (K = n^(1/3)) | unif-mass (upper, alpha=0.5) | 20.0 ± 0.0 | 0.4934 ± 0.1400 | 0.9151 ± 0.0076 |
| Rice (K = 2 n^(1/3)) | unif-mass (mean, alpha=any) | 30.0 ± 0.0 | 0.5889 ± 0.2397 | 0.9144 ± 0.0119 |
| Rice (K = 2 n^(1/3)) | unif-mass (upper, alpha=0.05) | 30.0 ± 0.0 | 0.5365 ± 0.1824 | 0.9144 ± 0.0118 |
| Rice (K = 2 n^(1/3)) | unif-mass (upper, alpha=0.1) | 30.0 ± 0.0 | 0.5365 ± 0.1824 | 0.9144 ± 0.0118 |
| Rice (K = 2 n^(1/3)) | unif-mass (upper, alpha=0.5) | 30.0 ± 0.0 | 0.5365 ± 0.1824 | 0.9144 ± 0.0118 |

**CIFAR-100 / ResNet-34**

| rule | method | k | fpr_test | roc_auc_test |
|---|---|---|---|---|
| n/a | raw-score (gini, upper, temp=1.0, mag=0.0) | 1.0 ± 0.0 | 0.4158 ± 0.0150 | 0.8772 ± 0.0038 |
| cube-root (K = n^(1/3)) | unif-mass (mean, alpha=any) | 20.0 ± 0.0 | 0.4528 ± 0.0236 | 0.8751 ± 0.0035 |
| cube-root (K = n^(1/3)) | unif-mass (upper, alpha=0.05) | 20.0 ± 0.0 | 0.4528 ± 0.0236 | 0.8751 ± 0.0035 |
| cube-root (K = n^(1/3)) | unif-mass (upper, alpha=0.1) | 20.0 ± 0.0 | 0.4528 ± 0.0236 | 0.8751 ± 0.0035 |
| cube-root (K = n^(1/3)) | unif-mass (upper, alpha=0.5) | 20.0 ± 0.0 | 0.4528 ± 0.0236 | 0.8751 ± 0.0035 |
| Rice (K = 2 n^(1/3)) | unif-mass (mean, alpha=any) | 30.0 ± 0.0 | 0.4416 ± 0.0222 | 0.8737 ± 0.0040 |
| Rice (K = 2 n^(1/3)) | unif-mass (upper, alpha=0.05) | 30.0 ± 0.0 | 0.4416 ± 0.0222 | 0.8737 ± 0.0040 |
| Rice (K = 2 n^(1/3)) | unif-mass (upper, alpha=0.1) | 30.0 ± 0.0 | 0.4416 ± 0.0222 | 0.8737 ± 0.0040 |
| Rice (K = 2 n^(1/3)) | unif-mass (upper, alpha=0.5) | 30.0 ± 0.0 | 0.4416 ± 0.0222 | 0.8737 ± 0.0040 |

### Res-selected doctor raw-score + rule-of-thumb K by n_res (CIFAR-10)
Source folders:
- Unif-mass grids: `results/partition_binning/cifar10/resnet34_ce/partition/runs/unif-mass-grid-nres{500,1000,2000,3000,4000}-20260114/`
- Doctor res-grid searches: `results/cifar10/resnet34_ce/doctor/runs/doctor-res-grid-nres{500,1000,2000,3000,4000}-20260114c/`

Raw-score hyperparameters are selected on the res split (metric fpr_res). The tables below report test performance only; for score=mean, alpha is not used.
Note: large K does not rescue FPR@95 for uniform-mass. With fixed n_cal, increasing K yields small per-bin counts, noisy bin estimates, and a stepwise score that makes the 95% TPR threshold jump across large bin masses. For skewed 1D scores (e.g., gini), many quantile bins sit in a narrow score range, so the effective resolution stays coarse even for large K.

Related work (monotone calibration and binning):
- Zadrozny & Elkan (2002), "Transforming classifier scores into accurate multiclass probability estimates" (isotonic regression calibration). DOI: 10.1145/775047.775151.
- Niculescu-Mizil & Caruana (2005), "Predicting Good Probabilities with Supervised Learning" (compares isotonic and histogram binning for calibration).
- Vovk et al. (2020), "Detecting adversarial manipulation using inductive Venn-Abers predictors" (isotonic-based probability intervals). DOI: 10.1016/j.neucom.2019.11.113.
- Brümmer & du Preez (2009), "Similarity-Binning Averaging: A Generalisation of Binning Calibration" (binning-based calibration). DOI: 10.1007/978-3-642-04394-9_42.
- Shi et al. (2023), "Calibration Error Estimation Using Fuzzy Binning" (binning variants). DOI: 10.1007/978-3-031-46778-3_9.

#### n_res=500 (n_cal=4500, K_cube=16.5, K_Rice=33.0)
| rule | method | k | fpr_test | roc_auc_test |
|---|---|---|---|---|
| n/a | raw-score (doctor, res-selected) | 1 | 0.2703 | 0.9257 |
| cube-root (K=16.5) | unif-mass (mean, alpha=any) | 20 | 0.4681 | 0.9049 |
| cube-root (K=16.5) | unif-mass (upper, alpha=0.05) | 20 | 0.4451 | 0.9080 |
| cube-root (K=16.5) | unif-mass (upper, alpha=0.10) | 20 | 0.4451 | 0.9081 |
| cube-root (K=16.5) | unif-mass (upper, alpha=0.50) | 20 | 0.4451 | 0.9090 |
| Rice (K=33.0) | unif-mass (mean, alpha=any) | 30 | 0.3990 | 0.9097 |
| Rice (K=33.0) | unif-mass (upper, alpha=0.05) | 30 | 0.4394 | 0.9108 |
| Rice (K=33.0) | unif-mass (upper, alpha=0.10) | 30 | 0.4394 | 0.9110 |
| Rice (K=33.0) | unif-mass (upper, alpha=0.50) | 30 | 0.4394 | 0.9120 |
| oracle best fpr_test | unif-mass (mean, alpha=any) | 5 | 0.3598 | 0.8804 |
| oracle best roc_auc_test | unif-mass (mean, alpha=any) | 30 | 0.3990 | 0.9097 |
| oracle best fpr_test | unif-mass (upper, alpha=0.05) | 300 | 0.3181 | 0.9159 |
| oracle best roc_auc_test | unif-mass (upper, alpha=0.05) | 150 | 0.3680 | 0.9172 |
| oracle best fpr_test | unif-mass (upper, alpha=0.10) | 300 | 0.3181 | 0.9160 |
| oracle best roc_auc_test | unif-mass (upper, alpha=0.10) | 150 | 0.3680 | 0.9173 |
| oracle best fpr_test | unif-mass (upper, alpha=0.50) | 300 | 0.3233 | 0.9167 |
| oracle best roc_auc_test | unif-mass (upper, alpha=0.50) | 150 | 0.3780 | 0.9171 |

#### n_res=1000 (n_cal=4000, K_cube=15.9, K_Rice=31.7)
| rule | method | k | fpr_test | roc_auc_test |
|---|---|---|---|---|
| n/a | raw-score (doctor, res-selected) | 1 | 0.2244 | 0.9351 |
| cube-root (K=15.9) | unif-mass (mean, alpha=any) | 20 | 0.4495 | 0.9084 |
| cube-root (K=15.9) | unif-mass (upper, alpha=0.05) | 20 | 0.4495 | 0.9106 |
| cube-root (K=15.9) | unif-mass (upper, alpha=0.10) | 20 | 0.4495 | 0.9106 |
| cube-root (K=15.9) | unif-mass (upper, alpha=0.50) | 20 | 0.4495 | 0.9106 |
| Rice (K=31.7) | unif-mass (mean, alpha=any) | 30 | 0.5304 | 0.9101 |
| Rice (K=31.7) | unif-mass (upper, alpha=0.05) | 30 | 0.5304 | 0.9088 |
| Rice (K=31.7) | unif-mass (upper, alpha=0.10) | 30 | 0.5304 | 0.9088 |
| Rice (K=31.7) | unif-mass (upper, alpha=0.50) | 30 | 0.5304 | 0.9088 |
| oracle best fpr_test | unif-mass (mean, alpha=any) | 20 | 0.4495 | 0.9084 |
| oracle best roc_auc_test | unif-mass (mean, alpha=any) | 30 | 0.5304 | 0.9101 |
| oracle best fpr_test | unif-mass (upper, alpha=0.05) | 20 | 0.4495 | 0.9106 |
| oracle best roc_auc_test | unif-mass (upper, alpha=0.05) | 20 | 0.4495 | 0.9106 |
| oracle best fpr_test | unif-mass (upper, alpha=0.10) | 20 | 0.4495 | 0.9106 |
| oracle best roc_auc_test | unif-mass (upper, alpha=0.10) | 20 | 0.4495 | 0.9106 |
| oracle best fpr_test | unif-mass (upper, alpha=0.50) | 20 | 0.4495 | 0.9106 |
| oracle best roc_auc_test | unif-mass (upper, alpha=0.50) | 20 | 0.4495 | 0.9106 |

#### n_res=2000 (n_cal=3000, K_cube=14.4, K_Rice=28.8)
| rule | method | k | fpr_test | roc_auc_test |
|---|---|---|---|---|
| n/a | raw-score (doctor, res-selected) | 1 | 0.2330 | 0.9153 |
| cube-root (K=14.4) | unif-mass (mean, alpha=any) | 10 | 0.5654 | 0.8999 |
| cube-root (K=14.4) | unif-mass (upper, alpha=0.05) | 10 | 0.5654 | 0.9012 |
| cube-root (K=14.4) | unif-mass (upper, alpha=0.10) | 10 | 0.5654 | 0.9012 |
| cube-root (K=14.4) | unif-mass (upper, alpha=0.50) | 10 | 0.5654 | 0.9012 |
| Rice (K=28.8) | unif-mass (mean, alpha=any) | 30 | 1.0000 | 0.9033 |
| Rice (K=28.8) | unif-mass (upper, alpha=0.05) | 30 | 0.5532 | 0.9073 |
| Rice (K=28.8) | unif-mass (upper, alpha=0.10) | 30 | 0.5532 | 0.9073 |
| Rice (K=28.8) | unif-mass (upper, alpha=0.50) | 30 | 0.5532 | 0.9073 |
| oracle best fpr_test | unif-mass (mean, alpha=any) | 5 | 0.3621 | 0.8809 |
| oracle best roc_auc_test | unif-mass (mean, alpha=any) | 20 | 0.4470 | 0.9092 |
| oracle best fpr_test | unif-mass (upper, alpha=0.05) | 5 | 0.3621 | 0.8809 |
| oracle best roc_auc_test | unif-mass (upper, alpha=0.05) | 20 | 0.4470 | 0.9086 |
| oracle best fpr_test | unif-mass (upper, alpha=0.10) | 5 | 0.3621 | 0.8809 |
| oracle best roc_auc_test | unif-mass (upper, alpha=0.10) | 20 | 0.4470 | 0.9086 |
| oracle best fpr_test | unif-mass (upper, alpha=0.50) | 5 | 0.3621 | 0.8809 |
| oracle best roc_auc_test | unif-mass (upper, alpha=0.50) | 20 | 0.4470 | 0.9086 |

#### n_res=3000 (n_cal=2000, K_cube=12.6, K_Rice=25.2)
| rule | method | k | fpr_test | roc_auc_test |
|---|---|---|---|---|
| n/a | raw-score (doctor, res-selected) | 1 | 0.2399 | 0.9169 |
| cube-root (K=12.6) | unif-mass (mean, alpha=any) | 10 | 0.5631 | 0.8928 |
| cube-root (K=12.6) | unif-mass (upper, alpha=0.05) | 10 | 0.5631 | 0.8928 |
| cube-root (K=12.6) | unif-mass (upper, alpha=0.10) | 10 | 0.5631 | 0.8928 |
| cube-root (K=12.6) | unif-mass (upper, alpha=0.50) | 10 | 0.5631 | 0.8928 |
| Rice (K=25.2) | unif-mass (mean, alpha=any) | 30 | 1.0000 | 0.8918 |
| Rice (K=25.2) | unif-mass (upper, alpha=0.05) | 30 | 0.9518 | 0.8962 |
| Rice (K=25.2) | unif-mass (upper, alpha=0.10) | 30 | 0.9518 | 0.8962 |
| Rice (K=25.2) | unif-mass (upper, alpha=0.50) | 30 | 0.9518 | 0.8962 |
| oracle best fpr_test | unif-mass (mean, alpha=any) | 10 | 0.5631 | 0.8928 |
| oracle best roc_auc_test | unif-mass (mean, alpha=any) | 20 | 1.0000 | 0.8986 |
| oracle best fpr_test | unif-mass (upper, alpha=0.05) | 10 | 0.5631 | 0.8928 |
| oracle best roc_auc_test | unif-mass (upper, alpha=0.05) | 20 | 1.0000 | 0.8986 |
| oracle best fpr_test | unif-mass (upper, alpha=0.10) | 10 | 0.5631 | 0.8928 |
| oracle best roc_auc_test | unif-mass (upper, alpha=0.10) | 20 | 1.0000 | 0.8986 |
| oracle best fpr_test | unif-mass (upper, alpha=0.50) | 10 | 0.5631 | 0.8928 |
| oracle best roc_auc_test | unif-mass (upper, alpha=0.50) | 20 | 1.0000 | 0.8986 |

#### n_res=4000 (n_cal=1000, K_cube=10.0, K_Rice=20.0)
| rule | method | k | fpr_test | roc_auc_test |
|---|---|---|---|---|
| n/a | raw-score (doctor, res-selected) | 1 | 0.2257 | 0.9354 |
| cube-root (K=10.0) | unif-mass (mean, alpha=any) | 10 | 1.0000 | 0.8943 |
| cube-root (K=10.0) | unif-mass (upper, alpha=0.05) | 10 | 0.6691 | 0.9013 |
| cube-root (K=10.0) | unif-mass (upper, alpha=0.10) | 10 | 0.6691 | 0.9013 |
| cube-root (K=10.0) | unif-mass (upper, alpha=0.50) | 10 | 0.6691 | 0.9013 |
| Rice (K=20.0) | unif-mass (mean, alpha=any) | 20 | 1.0000 | 0.9005 |
| Rice (K=20.0) | unif-mass (upper, alpha=0.05) | 20 | 0.8441 | 0.9012 |
| Rice (K=20.0) | unif-mass (upper, alpha=0.10) | 20 | 0.8441 | 0.9012 |
| Rice (K=20.0) | unif-mass (upper, alpha=0.50) | 20 | 0.8441 | 0.9012 |
| oracle best fpr_test | unif-mass (mean, alpha=any) | 5 | 0.7839 | 0.8663 |
| oracle best roc_auc_test | unif-mass (mean, alpha=any) | 20 | 1.0000 | 0.9005 |
| oracle best fpr_test | unif-mass (upper, alpha=0.05) | 500 | 0.5637 | 0.7876 |
| oracle best roc_auc_test | unif-mass (upper, alpha=0.05) | 10 | 0.6691 | 0.9013 |
| oracle best fpr_test | unif-mass (upper, alpha=0.10) | 500 | 0.5637 | 0.7876 |
| oracle best roc_auc_test | unif-mass (upper, alpha=0.10) | 10 | 0.6691 | 0.9013 |
| oracle best fpr_test | unif-mass (upper, alpha=0.50) | 500 | 0.5662 | 0.8093 |
| oracle best roc_auc_test | unif-mass (upper, alpha=0.50) | 10 | 0.6691 | 0.9013 |

### Soft-kmeans res-fit with init selection (partition on res, binning on cal) — CIFAR-10, n_res=500
Source:
- `results/partition_binning/cifar10/resnet34_ce/partition/runs/soft-kmeans-grid-nres500-fpr-20260115f/`

Experiment definition (precise):
- Dataset/model: CIFAR-10 / ResNet-34 (preprocessor `ce`), seed split 9.
- Splits: n_res=500, n_cal=4500, n_test=5000.
- Postprocessor: `partition` with `method=soft-kmeans_torch`.
- Embedding space: full class probits (`space=probits`), reordered (`reorder_embs=True`).
- Other key args: `temperature=2`, `pred_weights=0`, `bound=bernstein`, `n_init=5`, `max_iter=300`.
- Grid: `n_clusters` in {5,10,20,30,50,75,100,150,200,300}, `score` in {mean, upper}, `alpha` in {0.05,0.10,0.50} (alpha ignored when score=mean).
- Fitting protocol: fit the partition (resolution function) on res only, then compute bin statistics on cal, then evaluate on test.

Selection protocol:
- K chosen by rule-of-thumb from n_cal=4500: cube-root -> K=20, Rice -> K=30 (nearest grid value).
- For each method (score/alpha), select the init by a res criterion (`fpr_res`, `roc_auc_res`, or `inertia_res`).
- Report test metrics from the selected init only.
- Raw-score baseline uses res-selected doctor (from the earlier res-grid).

| rule | init_select | method | k | fpr_test | roc_auc_test |
|---|---|---|---|---|---|
| n/a | n/a | raw-score (doctor, res-selected) | 1 | 0.2703 | 0.9257 |
| cube-root (K=20) | fpr_res | soft-kmeans (mean, alpha=any) | 20 | 0.7945 | 0.8970 |
| cube-root (K=20) | roc_auc_res | soft-kmeans (mean, alpha=any) | 20 | 0.5860 | 0.9156 |
| cube-root (K=20) | inertia_res | soft-kmeans (mean, alpha=any) | 20 | 0.5662 | 0.9024 |
| cube-root (K=20) | fpr_res | soft-kmeans (upper, alpha=0.05) | 20 | 0.6813 | 0.9021 |
| cube-root (K=20) | roc_auc_res | soft-kmeans (upper, alpha=0.05) | 20 | 0.5660 | 0.9087 |
| cube-root (K=20) | inertia_res | soft-kmeans (upper, alpha=0.05) | 20 | 0.7120 | 0.9000 |
| cube-root (K=20) | fpr_res | soft-kmeans (upper, alpha=0.10) | 20 | 0.6813 | 0.9014 |
| cube-root (K=20) | roc_auc_res | soft-kmeans (upper, alpha=0.10) | 20 | 0.5660 | 0.9098 |
| cube-root (K=20) | inertia_res | soft-kmeans (upper, alpha=0.10) | 20 | 0.7120 | 0.9000 |
| cube-root (K=20) | fpr_res | soft-kmeans (upper, alpha=0.50) | 20 | 0.6813 | 0.9006 |
| cube-root (K=20) | roc_auc_res | soft-kmeans (upper, alpha=0.50) | 20 | 0.5660 | 0.9063 |
| cube-root (K=20) | inertia_res | soft-kmeans (upper, alpha=0.50) | 20 | 0.7120 | 0.9032 |
| Rice (K=30) | fpr_res | soft-kmeans (mean, alpha=any) | 30 | 0.6862 | 0.8954 |
| Rice (K=30) | roc_auc_res | soft-kmeans (mean, alpha=any) | 30 | 0.7983 | 0.9000 |
| Rice (K=30) | inertia_res | soft-kmeans (mean, alpha=any) | 30 | 0.7378 | 0.8991 |
| Rice (K=30) | fpr_res | soft-kmeans (upper, alpha=0.05) | 30 | 0.6871 | 0.8874 |
| Rice (K=30) | roc_auc_res | soft-kmeans (upper, alpha=0.05) | 30 | 0.8371 | 0.8869 |
| Rice (K=30) | inertia_res | soft-kmeans (upper, alpha=0.05) | 30 | 0.8360 | 0.8870 |
| Rice (K=30) | fpr_res | soft-kmeans (upper, alpha=0.10) | 30 | 0.6871 | 0.8865 |
| Rice (K=30) | roc_auc_res | soft-kmeans (upper, alpha=0.10) | 30 | 0.8371 | 0.8892 |
| Rice (K=30) | inertia_res | soft-kmeans (upper, alpha=0.10) | 30 | 0.8360 | 0.8871 |
| Rice (K=30) | fpr_res | soft-kmeans (upper, alpha=0.50) | 30 | 0.6871 | 0.8851 |
| Rice (K=30) | roc_auc_res | soft-kmeans (upper, alpha=0.50) | 30 | 0.8371 | 0.8872 |
| Rice (K=30) | inertia_res | soft-kmeans (upper, alpha=0.50) | 30 | 0.8360 | 0.8858 |

Oracle best test per method (diagnostic only; selection uses test):

| select | method | k | init | fpr_test | roc_auc_test |
|---|---|---|---|---|---|
| best fpr_test | soft-kmeans (mean, alpha=any) | 50 | 4 | 0.4371 | 0.9072 |
| best roc_auc_test | soft-kmeans (mean, alpha=any) | 30 | 4 | 0.4604 | 0.9218 |
| best fpr_test | soft-kmeans (upper, alpha=0.05) | 100 | 2 | 0.2252 | 0.8799 |
| best roc_auc_test | soft-kmeans (upper, alpha=0.05) | 50 | 1 | 0.2423 | 0.9184 |
| best fpr_test | soft-kmeans (upper, alpha=0.10) | 100 | 2 | 0.2252 | 0.8852 |
| best roc_auc_test | soft-kmeans (upper, alpha=0.10) | 50 | 1 | 0.2423 | 0.9188 |
| best fpr_test | soft-kmeans (upper, alpha=0.50) | 100 | 2 | 0.2302 | 0.9007 |
| best roc_auc_test | soft-kmeans (upper, alpha=0.50) | 50 | 3 | 0.2308 | 0.9218 |

Best mean test performance per method (mean ± std over inits for each K):

| select | method | k | fpr_test | roc_auc_test |
|---|---|---|---|---|
| best mean fpr_test | soft-kmeans (mean, alpha=any) | 50 | 0.5023 ± 0.0713 | 0.9104 ± 0.0057 |
| best mean roc_auc_test | soft-kmeans (mean, alpha=any) | 50 | 0.5023 ± 0.0713 | 0.9104 ± 0.0057 |
| best mean fpr_test | soft-kmeans (upper, alpha=0.05) | 50 | 0.2475 ± 0.0111 | 0.9149 ± 0.0032 |
| best mean roc_auc_test | soft-kmeans (upper, alpha=0.05) | 50 | 0.2475 ± 0.0111 | 0.9149 ± 0.0032 |
| best mean fpr_test | soft-kmeans (upper, alpha=0.10) | 50 | 0.2475 ± 0.0111 | 0.9154 ± 0.0033 |
| best mean roc_auc_test | soft-kmeans (upper, alpha=0.10) | 50 | 0.2475 ± 0.0111 | 0.9154 ± 0.0033 |
| best mean fpr_test | soft-kmeans (upper, alpha=0.50) | 100 | 0.2449 ± 0.0143 | 0.9107 ± 0.0088 |
| best mean roc_auc_test | soft-kmeans (upper, alpha=0.50) | 50 | 0.2475 ± 0.0111 | 0.9188 ± 0.0026 |

### Soft-kmeans with doctor res-selection (fit on res, binning on cal) — CIFAR-10, n_res=1000
Source:
- `results/partition_binning/cifar10/resnet34_ce/partition/runs/soft-kmeans-grid-nres1000-doctor-20260115d/`

Doctor hyperparameters (temperature, magnitude, normalize) are selected on res from `doctor-res-grid-nres1000-20260114c` and applied to the gini score.  
K chosen by rule-of-thumb from n_cal=4000 (cube-root -> K=20, Rice -> K=30).  
Init is selected by res metric (`fpr_res`, `roc_auc_res`, or `inertia_res`); report test metrics only.

| rule | init_select | method | k | fpr_test | roc_auc_test |
|---|---|---|---|---|---|
| n/a | n/a | raw-score (doctor, res-selected) | 1 | 0.2244 | 0.9351 |
| cube-root (K=20) | fpr_res | soft-kmeans-doctor (score=mean, alpha=any) | 20 | 0.9996 | 0.9037 |
| cube-root (K=20) | roc_auc_res | soft-kmeans-doctor (score=mean, alpha=any) | 20 | 0.9994 | 0.8985 |
| cube-root (K=20) | inertia_res | soft-kmeans-doctor (score=mean, alpha=any) | 20 | 0.9998 | 0.9040 |
| cube-root (K=20) | fpr_res | soft-kmeans-doctor (score=upper, alpha=0.05) | 20 | 1.0000 | 0.9040 |
| cube-root (K=20) | roc_auc_res | soft-kmeans-doctor (score=upper, alpha=0.05) | 20 | 1.0000 | 0.9036 |
| cube-root (K=20) | inertia_res | soft-kmeans-doctor (score=upper, alpha=0.05) | 20 | 1.0000 | 0.9066 |
| cube-root (K=20) | fpr_res | soft-kmeans-doctor (score=upper, alpha=0.1) | 20 | 1.0000 | 0.9040 |
| cube-root (K=20) | roc_auc_res | soft-kmeans-doctor (score=upper, alpha=0.1) | 20 | 1.0000 | 0.9036 |
| cube-root (K=20) | inertia_res | soft-kmeans-doctor (score=upper, alpha=0.1) | 20 | 1.0000 | 0.9066 |
| cube-root (K=20) | fpr_res | soft-kmeans-doctor (score=upper, alpha=0.5) | 20 | 1.0000 | 0.9045 |
| cube-root (K=20) | roc_auc_res | soft-kmeans-doctor (score=upper, alpha=0.5) | 20 | 1.0000 | 0.9044 |
| cube-root (K=20) | inertia_res | soft-kmeans-doctor (score=upper, alpha=0.5) | 20 | 1.0000 | 0.9061 |
| Rice (K=30) | fpr_res | soft-kmeans-doctor (score=mean, alpha=any) | 30 | 0.9998 | 0.8996 |
| Rice (K=30) | roc_auc_res | soft-kmeans-doctor (score=mean, alpha=any) | 30 | 0.9990 | 0.8627 |
| Rice (K=30) | inertia_res | soft-kmeans-doctor (score=mean, alpha=any) | 30 | 1.0000 | 0.8945 |
| Rice (K=30) | fpr_res | soft-kmeans-doctor (score=upper, alpha=0.05) | 30 | 1.0000 | 0.9036 |
| Rice (K=30) | roc_auc_res | soft-kmeans-doctor (score=upper, alpha=0.05) | 30 | 1.0000 | 0.9046 |
| Rice (K=30) | inertia_res | soft-kmeans-doctor (score=upper, alpha=0.05) | 30 | 1.0000 | 0.9032 |
| Rice (K=30) | fpr_res | soft-kmeans-doctor (score=upper, alpha=0.1) | 30 | 1.0000 | 0.9036 |
| Rice (K=30) | roc_auc_res | soft-kmeans-doctor (score=upper, alpha=0.1) | 30 | 1.0000 | 0.9059 |
| Rice (K=30) | inertia_res | soft-kmeans-doctor (score=upper, alpha=0.1) | 30 | 1.0000 | 0.9037 |
| Rice (K=30) | fpr_res | soft-kmeans-doctor (score=upper, alpha=0.5) | 30 | 1.0000 | 0.9036 |
| Rice (K=30) | roc_auc_res | soft-kmeans-doctor (score=upper, alpha=0.5) | 30 | 1.0000 | 0.9041 |
| Rice (K=30) | inertia_res | soft-kmeans-doctor (score=upper, alpha=0.5) | 30 | 1.0000 | 0.9041 |

Oracle best test per method (diagnostic only; selection uses test):

| select | method | k | init | fpr_test | roc_auc_test |
|---|---|---|---|---|---|
| best fpr_test | soft-kmeans-doctor (score=mean, alpha=any) | 300 | 1 | 0.9969 | 0.8617 |
| best roc_auc_test | soft-kmeans-doctor (score=mean, alpha=any) | 20 | 3 | 0.9998 | 0.9040 |
| best fpr_test | soft-kmeans-doctor (score=upper, alpha=0.05) | 20 | 3 | 1.0000 | 0.9066 |
| best roc_auc_test | soft-kmeans-doctor (score=upper, alpha=0.05) | 20 | 3 | 1.0000 | 0.9066 |
| best fpr_test | soft-kmeans-doctor (score=upper, alpha=0.1) | 20 | 3 | 1.0000 | 0.9066 |
| best roc_auc_test | soft-kmeans-doctor (score=upper, alpha=0.1) | 20 | 3 | 1.0000 | 0.9066 |
| best fpr_test | soft-kmeans-doctor (score=upper, alpha=0.5) | 200 | 1 | 1.0000 | 0.9079 |
| best roc_auc_test | soft-kmeans-doctor (score=upper, alpha=0.5) | 200 | 1 | 1.0000 | 0.9079 |

Best mean test performance per method (mean ± std over inits for each K):

| select | method | k | fpr_test | roc_auc_test |
|---|---|---|---|---|
| best mean fpr_test | soft-kmeans-doctor (score=mean, alpha=any) | 50 | 0.9988 ± 0.0006 | 0.8680 ± 0.0092 |
| best mean roc_auc_test | soft-kmeans-doctor (score=mean, alpha=any) | 10 | 1.0000 ± 0.0000 | 0.9004 ± 0.0016 |
| best mean fpr_test | soft-kmeans-doctor (score=upper, alpha=0.05) | 20 | 1.0000 ± 0.0000 | 0.9049 ± 0.0013 |
| best mean roc_auc_test | soft-kmeans-doctor (score=upper, alpha=0.05) | 20 | 1.0000 ± 0.0000 | 0.9049 ± 0.0013 |
| best mean fpr_test | soft-kmeans-doctor (score=upper, alpha=0.1) | 200 | 1.0000 ± 0.0000 | 0.9060 ± 0.0006 |
| best mean roc_auc_test | soft-kmeans-doctor (score=upper, alpha=0.1) | 200 | 1.0000 ± 0.0000 | 0.9060 ± 0.0006 |
| best mean fpr_test | soft-kmeans-doctor (score=upper, alpha=0.5) | 200 | 1.0000 ± 0.0000 | 0.9066 ± 0.0011 |
| best mean roc_auc_test | soft-kmeans-doctor (score=upper, alpha=0.5) | 200 | 1.0000 ± 0.0000 | 0.9066 ± 0.0011 |
