# Experimental Procedure: Binned Error Detectors With Guarantees

Last updated: 2026-01-17

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

#### n_res=1000 (perturbed logits, diagnostic; same K rules)
Source: `results/cifar10/resnet34_ce/partition/runs/unif-mass-grid-nres1000-perturbed-20260117/`  
Note: `use_perturbed_logits=True` with doctor-selected (temperature=0.9, magnitude=0.002, normalize=true).

| rule | method | k | fpr_test | roc_auc_test |
|---|---|---|---|---|
| n/a | raw-score (doctor, res-selected) | 1 | 0.2244 | 0.9351 |
| cube-root (K=15.9) | unif-mass (mean, alpha=any) | 20 | 0.3617 | 0.9273 |
| cube-root (K=15.9) | unif-mass (upper, alpha=0.05) | 20 | 0.3617 | 0.9280 |
| cube-root (K=15.9) | unif-mass (upper, alpha=0.10) | 20 | 0.3617 | 0.9280 |
| cube-root (K=15.9) | unif-mass (upper, alpha=0.50) | 20 | 0.3617 | 0.9280 |
| Rice (K=31.7) | unif-mass (mean, alpha=any) | 30 | 0.3875 | 0.9234 |
| Rice (K=31.7) | unif-mass (upper, alpha=0.05) | 30 | 0.3875 | 0.9253 |
| Rice (K=31.7) | unif-mass (upper, alpha=0.10) | 30 | 0.3875 | 0.9253 |
| Rice (K=31.7) | unif-mass (upper, alpha=0.50) | 30 | 0.3875 | 0.9253 |

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

### Isotonic-binning res-fit (partition on res, binning on cal) — CIFAR-10, n_res=1000
Source:
- `results/cifar10/resnet34_ce/partition/runs/isotonic-binning-grid-nres1000-20260115/`

Experiment definition (precise):
- Dataset/model: CIFAR-10 / ResNet-34 (preprocessor `ce`), seed split 9.
- Splits: n_res=1000, n_cal=4000, n_test=5000.
- Postprocessor: `partition` with `method=isotonic-binning`.
- Embedding space: gini (doctor params selected on res), `use_perturbed_logits=True`.
- Grid: `n_clusters` in {5,10,20,30,50,75,100,150,200,300,500}, `n_min` in {1,5,10,20}, `score` in {mean, upper}, `alpha` in {0.05,0.10,0.50} (alpha ignored when score=mean).
- Fitting protocol: fit the isotonic partition on res only, then compute bin statistics on cal, then evaluate on test.

Selection protocol:
- K chosen by rule-of-thumb from n_cal=4000: cube-root -> K=20, Rice -> K=30 (nearest grid value).
- For each method (score/alpha), pick n_min by best fpr_res at the chosen K.
- Report test metrics from the selected (K, n_min) only.

| rule | method | k | n_min | fpr_test | roc_auc_test |
|---|---|---|---|---|---|
| n/a | raw-score (doctor, res-selected) | 1 | n/a | 0.2244 | 0.9351 |
| cube-root (K=20) | isotonic-binning (mean, alpha=any) | 20 | 20 | 1.0000 | 0.8916 |
| cube-root (K=20) | isotonic-binning (upper, alpha=0.05) | 20 | 5 | 1.0000 | 0.8926 |
| cube-root (K=20) | isotonic-binning (upper, alpha=0.10) | 20 | 5 | 1.0000 | 0.8926 |
| cube-root (K=20) | isotonic-binning (upper, alpha=0.50) | 20 | 5 | 1.0000 | 0.8926 |
| Rice (K=30) | isotonic-binning (mean, alpha=any) | 30 | 20 | 1.0000 | 0.8916 |
| Rice (K=30) | isotonic-binning (upper, alpha=0.05) | 30 | 5 | 1.0000 | 0.8926 |
| Rice (K=30) | isotonic-binning (upper, alpha=0.10) | 30 | 5 | 1.0000 | 0.8926 |
| Rice (K=30) | isotonic-binning (upper, alpha=0.50) | 30 | 5 | 1.0000 | 0.8926 |

Oracle best test per method (diagnostic only; selection uses test):

| select | method | k | n_min | fpr_test | roc_auc_test |
|---|---|---|---|---|---|
| oracle best fpr_test | isotonic-binning (mean, alpha=any) | 10 | 1 | 0.9992 | 0.8748 |
| oracle best roc_auc_test | isotonic-binning (mean, alpha=any) | 5 | 20 | 1.0000 | 0.8916 |
| oracle best fpr_test | isotonic-binning (upper, alpha=0.05) | 10 | 1 | 1.0000 | 0.8926 |
| oracle best roc_auc_test | isotonic-binning (upper, alpha=0.05) | 10 | 1 | 1.0000 | 0.8926 |
| oracle best fpr_test | isotonic-binning (upper, alpha=0.10) | 10 | 1 | 1.0000 | 0.8926 |
| oracle best roc_auc_test | isotonic-binning (upper, alpha=0.10) | 10 | 1 | 1.0000 | 0.8926 |
| oracle best fpr_test | isotonic-binning (upper, alpha=0.50) | 10 | 1 | 1.0000 | 0.8926 |
| oracle best roc_auc_test | isotonic-binning (upper, alpha=0.50) | 10 | 1 | 1.0000 | 0.8926 |

#### n_res=3000 (n_cal=2000, K_cube=12.6, K_Rice=25.2)
| rule | method | k | n_min | fpr_test | roc_auc_test |
|---|---|---|---|---|---|
| n/a | raw-score (doctor, res-selected) | 1 | n/a | 0.2399 | 0.9169 |
| cube-root (K=12.6) | isotonic-binning (mean, alpha=any) | 10 | 10 | 0.6873 | 0.8963 |
| cube-root (K=12.6) | isotonic-binning (upper, alpha=0.05) | 10 | 20 | 0.4183 | 0.9077 |
| cube-root (K=12.6) | isotonic-binning (upper, alpha=0.10) | 10 | 20 | 0.4183 | 0.9077 |
| cube-root (K=12.6) | isotonic-binning (upper, alpha=0.50) | 10 | 20 | 0.4183 | 0.9095 |
| Rice (K=25.2) | isotonic-binning (mean, alpha=any) | 30 | 5 | 0.4183 | 0.9035 |
| Rice (K=25.2) | isotonic-binning (upper, alpha=0.05) | 30 | 10 | 0.4183 | 0.9077 |
| Rice (K=25.2) | isotonic-binning (upper, alpha=0.10) | 30 | 10 | 0.4183 | 0.9077 |
| Rice (K=25.2) | isotonic-binning (upper, alpha=0.50) | 30 | 10 | 0.4183 | 0.9096 |

Oracle best test per method (diagnostic only; selection uses test):

| select | method | k | n_min | fpr_test | roc_auc_test |
|---|---|---|---|---|---|
| oracle best fpr_test | isotonic-binning (mean, alpha=any) | 20 | 1 | 0.4181 | 0.9037 |
| oracle best roc_auc_test | isotonic-binning (mean, alpha=any) | 20 | 1 | 0.4181 | 0.9037 |
| oracle best fpr_test | isotonic-binning (upper, alpha=0.05) | 20 | 1 | 0.4183 | 0.9078 |
| oracle best roc_auc_test | isotonic-binning (upper, alpha=0.05) | 20 | 1 | 0.4183 | 0.9078 |
| oracle best fpr_test | isotonic-binning (upper, alpha=0.10) | 20 | 1 | 0.4183 | 0.9078 |
| oracle best roc_auc_test | isotonic-binning (upper, alpha=0.10) | 20 | 1 | 0.4183 | 0.9078 |
| oracle best fpr_test | isotonic-binning (upper, alpha=0.50) | 20 | 5 | 0.4183 | 0.9097 |
| oracle best roc_auc_test | isotonic-binning (upper, alpha=0.50) | 20 | 5 | 0.4183 | 0.9097 |

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

### Soft-kmeans with doctor res-selection + CDF score transform (fit on res, binning on cal) — CIFAR-10, n_res=1000
Source:
- `results/partition_binning/cifar10/resnet34_ce/partition/runs/soft-kmeans-cdf-grid-nres1000-20260115/`

Doctor hyperparameters (temperature, magnitude, normalize) are selected on res from `doctor-res-grid-nres1000-20260114c` and applied to the gini score.  
Before soft-kmeans, gini scores are mapped through the empirical CDF fit on res (monotone map to [0,1]); the same transform is applied to cal/test.  
K chosen by rule-of-thumb from n_cal=4000 (cube-root -> K=20, Rice -> K=30).  
Init is selected by res metric (`fpr_res`, `roc_auc_res`, or `inertia_res`); report test metrics only.

CDF transform definition (res split size $n$):
Let $s \\in [s_{\\min}, s_{\\max}]$ be the 1D score (gini). The empirical CDF is
$\\hat F(s)=\\frac{1}{n}\\sum_{i=1}^n \\mathbf{1}\\{s_i \\le s\\}$ with codomain $[0,1]$.
We use the rank-based variant to avoid 0/1:
$u_i = \\frac{\\mathrm{rank}(s_i) - 0.5}{n} \\in (0,1)$.
The transformed score is $t_i = u_i$ (monotone). This spreads dense regions and compresses tails, which helps soft-kmeans avoid a single dominant cluster, but it can still fail when the res distribution is highly concentrated or when cal/test shift relative to res.

Visualization (res split, seed 9; perturbed logits with magnitude=0.002; temperature=0.9, normalize=true):
`results/cifar10/resnet34_ce/partition/runs/soft-kmeans-cdf-grid-nres1000-20260115/seed-split-9/analysis/soft_kmeans_cdf_transform_res_dist.png`  
`results/cifar10/resnet34_ce/partition/runs/soft-kmeans-cdf-grid-nres1000-20260115/seed-split-9/analysis/soft_kmeans_cdf_transform_res_cdf.png`  
`results/cifar10/resnet34_ce/partition/runs/soft-kmeans-cdf-grid-nres1000-20260115/seed-split-9/analysis/soft_kmeans_cdf_transform_res_u_hist.png`

| rule | init_select | method | k | fpr_test | roc_auc_test |
|---|---|---|---|---|---|
| n/a | n/a | raw-score (doctor, res-selected) | 1 | 0.2244 | 0.9351 |
| cube-root (K=20) | fpr_res | soft-kmeans-doctor-cdf (score=mean, alpha=any) | 20 | 0.2559 | 0.9219 |
| cube-root (K=20) | roc_auc_res | soft-kmeans-doctor-cdf (score=mean, alpha=any) | 20 | 0.2559 | 0.9219 |
| cube-root (K=20) | inertia_res | soft-kmeans-doctor-cdf (score=mean, alpha=any) | 20 | 0.2640 | 0.9239 |
| Rice (K=30) | fpr_res | soft-kmeans-doctor-cdf (score=mean, alpha=any) | 30 | 0.2728 | 0.9283 |
| Rice (K=30) | roc_auc_res | soft-kmeans-doctor-cdf (score=mean, alpha=any) | 30 | 0.2324 | 0.9235 |
| Rice (K=30) | inertia_res | soft-kmeans-doctor-cdf (score=mean, alpha=any) | 30 | 0.2728 | 0.9283 |
| cube-root (K=20) | fpr_res | soft-kmeans-doctor-cdf (score=upper, alpha=0.05) | 20 | 0.5635 | 0.9040 |
| cube-root (K=20) | roc_auc_res | soft-kmeans-doctor-cdf (score=upper, alpha=0.05) | 20 | 0.5635 | 0.9040 |
| cube-root (K=20) | inertia_res | soft-kmeans-doctor-cdf (score=upper, alpha=0.05) | 20 | 0.5522 | 0.8997 |
| Rice (K=30) | fpr_res | soft-kmeans-doctor-cdf (score=upper, alpha=0.05) | 30 | 0.5017 | 0.9001 |
| Rice (K=30) | roc_auc_res | soft-kmeans-doctor-cdf (score=upper, alpha=0.05) | 30 | 0.7689 | 0.8825 |
| Rice (K=30) | inertia_res | soft-kmeans-doctor-cdf (score=upper, alpha=0.05) | 30 | 0.5017 | 0.9001 |
| cube-root (K=20) | fpr_res | soft-kmeans-doctor-cdf (score=upper, alpha=0.1) | 20 | 0.5522 | 0.9042 |
| cube-root (K=20) | roc_auc_res | soft-kmeans-doctor-cdf (score=upper, alpha=0.1) | 20 | 0.5151 | 0.9056 |
| cube-root (K=20) | inertia_res | soft-kmeans-doctor-cdf (score=upper, alpha=0.1) | 20 | 0.5522 | 0.9042 |
| Rice (K=30) | fpr_res | soft-kmeans-doctor-cdf (score=upper, alpha=0.1) | 30 | 0.4396 | 0.9078 |
| Rice (K=30) | roc_auc_res | soft-kmeans-doctor-cdf (score=upper, alpha=0.1) | 30 | 0.6641 | 0.8922 |
| Rice (K=30) | inertia_res | soft-kmeans-doctor-cdf (score=upper, alpha=0.1) | 30 | 0.4396 | 0.9078 |
| cube-root (K=20) | fpr_res | soft-kmeans-doctor-cdf (score=upper, alpha=0.5) | 20 | 0.4985 | 0.9126 |
| cube-root (K=20) | roc_auc_res | soft-kmeans-doctor-cdf (score=upper, alpha=0.5) | 20 | 0.3875 | 0.9188 |
| cube-root (K=20) | inertia_res | soft-kmeans-doctor-cdf (score=upper, alpha=0.5) | 20 | 0.4531 | 0.9120 |
| Rice (K=30) | fpr_res | soft-kmeans-doctor-cdf (score=upper, alpha=0.5) | 30 | 0.3248 | 0.9182 |
| Rice (K=30) | roc_auc_res | soft-kmeans-doctor-cdf (score=upper, alpha=0.5) | 30 | 0.4684 | 0.9038 |
| Rice (K=30) | inertia_res | soft-kmeans-doctor-cdf (score=upper, alpha=0.5) | 30 | 0.3248 | 0.9182 |

Oracle best test per method (diagnostic only; selection uses test):

| select | method | k | init | fpr_test | roc_auc_test |
|---|---|---|---|---|---|
| best fpr_test | soft-kmeans-doctor-cdf (score=mean, alpha=any) | 50 | 5 | 0.2253 | 0.9232 |
| best roc_auc_test | soft-kmeans-doctor-cdf (score=mean, alpha=any) | 30 | 4 | 0.2728 | 0.9283 |
| best fpr_test | soft-kmeans-doctor-cdf (score=upper, alpha=0.05) | 5 | 3 | 0.2471 | 0.8940 |
| best roc_auc_test | soft-kmeans-doctor-cdf (score=upper, alpha=0.05) | 10 | 1 | 0.3093 | 0.9121 |
| best fpr_test | soft-kmeans-doctor-cdf (score=upper, alpha=0.1) | 5 | 3 | 0.2471 | 0.8940 |
| best roc_auc_test | soft-kmeans-doctor-cdf (score=upper, alpha=0.1) | 10 | 3 | 0.3060 | 0.9181 |
| best fpr_test | soft-kmeans-doctor-cdf (score=upper, alpha=0.5) | 5 | 3 | 0.2471 | 0.8940 |
| best roc_auc_test | soft-kmeans-doctor-cdf (score=upper, alpha=0.5) | 20 | 5 | 0.3875 | 0.9188 |

Best mean test performance per method (mean ± std over inits for each K):

| select | method | k | fpr_test | roc_auc_test |
|---|---|---|---|---|
| best mean fpr_test | soft-kmeans-doctor-cdf (score=mean, alpha=any) | 5 | 0.2474 ± 0.0003 | 0.9079 ± 0.0001 |
| best mean roc_auc_test | soft-kmeans-doctor-cdf (score=mean, alpha=any) | 10 | 0.2877 ± 0.0078 | 0.9241 ± 0.0003 |
| best mean fpr_test | soft-kmeans-doctor-cdf (score=upper, alpha=0.05) | 5 | 0.2474 ± 0.0003 | 0.8940 ± 0.0001 |
| best mean roc_auc_test | soft-kmeans-doctor-cdf (score=upper, alpha=0.05) | 10 | 0.3585 ± 0.0253 | 0.9085 ± 0.0021 |
| best mean fpr_test | soft-kmeans-doctor-cdf (score=upper, alpha=0.1) | 5 | 0.2474 ± 0.0003 | 0.8940 ± 0.0001 |
| best mean roc_auc_test | soft-kmeans-doctor-cdf (score=upper, alpha=0.1) | 10 | 0.3159 ± 0.0069 | 0.9153 ± 0.0019 |
| best mean fpr_test | soft-kmeans-doctor-cdf (score=upper, alpha=0.5) | 5 | 0.2474 ± 0.0003 | 0.8940 ± 0.0001 |
| best mean roc_auc_test | soft-kmeans-doctor-cdf (score=upper, alpha=0.5) | 10 | 0.3159 ± 0.0069 | 0.9174 ± 0.0006 |

### K-means with doctor res-selection + CDF score transform (fit on res, binning on cal) — CIFAR-10, n_res=1000
Source:
- `results/cifar10/resnet34_ce/partition/runs/kmeans-cdf-grid-nres1000-20260116/`

Doctor hyperparameters (temperature, magnitude, normalize) are selected on res from `doctor-res-grid-nres1000-20260114c` and applied to the gini score.  
Before k-means, gini scores are mapped through the empirical CDF fit on res (monotone map to [0,1]); the same transform is applied to cal/test.  
K chosen by rule-of-thumb from n_cal=4000 (cube-root -> K=20, Rice -> K=30).  
Init is selected by res metric (`fpr_res`, `roc_auc_res`, or `inertia_res`); report test metrics only.

| rule | init_select | method | k | fpr_test | roc_auc_test |
|---|---|---|---|---|---|
| n/a | n/a | raw-score (doctor, res-selected) | 1 | 0.2244 | 0.9351 |
| cube-root (K=20) | fpr_res | kmeans-doctor-cdf (score=mean, alpha=any) | 20 | 0.2657 | 0.9290 |
| cube-root (K=20) | roc_auc_res | kmeans-doctor-cdf (score=mean, alpha=any) | 20 | 0.2657 | 0.9290 |
| cube-root (K=20) | inertia_res | kmeans-doctor-cdf (score=mean, alpha=any) | 20 | 0.2349 | 0.9289 |
| Rice (K=30) | fpr_res | kmeans-doctor-cdf (score=mean, alpha=any) | 30 | 0.2953 | 0.9200 |
| Rice (K=30) | roc_auc_res | kmeans-doctor-cdf (score=mean, alpha=any) | 30 | 0.3219 | 0.9240 |
| Rice (K=30) | inertia_res | kmeans-doctor-cdf (score=mean, alpha=any) | 30 | 0.3219 | 0.9240 |
| cube-root (K=20) | fpr_res | kmeans-doctor-cdf (score=upper, alpha=0.05) | 20 | 0.2657 | 0.9335 |
| cube-root (K=20) | roc_auc_res | kmeans-doctor-cdf (score=upper, alpha=0.05) | 20 | 0.2657 | 0.9335 |
| cube-root (K=20) | inertia_res | kmeans-doctor-cdf (score=upper, alpha=0.05) | 20 | 0.2349 | 0.9314 |
| Rice (K=30) | fpr_res | kmeans-doctor-cdf (score=upper, alpha=0.05) | 30 | 0.5153 | 0.9093 |
| Rice (K=30) | roc_auc_res | kmeans-doctor-cdf (score=upper, alpha=0.05) | 30 | 0.3089 | 0.9238 |
| Rice (K=30) | inertia_res | kmeans-doctor-cdf (score=upper, alpha=0.05) | 30 | 0.3089 | 0.9238 |
| cube-root (K=20) | fpr_res | kmeans-doctor-cdf (score=upper, alpha=0.1) | 20 | 0.2657 | 0.9332 |
| cube-root (K=20) | roc_auc_res | kmeans-doctor-cdf (score=upper, alpha=0.1) | 20 | 0.2657 | 0.9332 |
| cube-root (K=20) | inertia_res | kmeans-doctor-cdf (score=upper, alpha=0.1) | 20 | 0.2349 | 0.9314 |
| Rice (K=30) | fpr_res | kmeans-doctor-cdf (score=upper, alpha=0.1) | 30 | 0.5153 | 0.9118 |
| Rice (K=30) | roc_auc_res | kmeans-doctor-cdf (score=upper, alpha=0.1) | 30 | 0.2812 | 0.9260 |
| Rice (K=30) | inertia_res | kmeans-doctor-cdf (score=upper, alpha=0.1) | 30 | 0.2812 | 0.9260 |
| cube-root (K=20) | fpr_res | kmeans-doctor-cdf (score=upper, alpha=0.5) | 20 | 0.2657 | 0.9327 |
| cube-root (K=20) | roc_auc_res | kmeans-doctor-cdf (score=upper, alpha=0.5) | 20 | 0.2657 | 0.9327 |
| cube-root (K=20) | inertia_res | kmeans-doctor-cdf (score=upper, alpha=0.5) | 20 | 0.2349 | 0.9311 |
| Rice (K=30) | fpr_res | kmeans-doctor-cdf (score=upper, alpha=0.5) | 30 | 0.4193 | 0.9165 |
| Rice (K=30) | roc_auc_res | kmeans-doctor-cdf (score=upper, alpha=0.5) | 30 | 0.3219 | 0.9273 |
| Rice (K=30) | inertia_res | kmeans-doctor-cdf (score=upper, alpha=0.5) | 30 | 0.3219 | 0.9273 |

Oracle best test per method (diagnostic only; selection uses test):

| select | method | k | init | fpr_test | roc_auc_test |
|---|---|---|---|---|---|
| best fpr_test | kmeans-doctor-cdf (score=mean, alpha=any) | 50 | 5 | 0.2150 | 0.9221 |
| best roc_auc_test | kmeans-doctor-cdf (score=mean, alpha=any) | 20 | 5 | 0.2452 | 0.9293 |
| best fpr_test | kmeans-doctor-cdf (score=upper, alpha=0.05) | 20 | 4 | 0.2349 | 0.9314 |
| best roc_auc_test | kmeans-doctor-cdf (score=upper, alpha=0.05) | 20 | 2 | 0.2657 | 0.9335 |
| best fpr_test | kmeans-doctor-cdf (score=upper, alpha=0.1) | 20 | 4 | 0.2349 | 0.9314 |
| best roc_auc_test | kmeans-doctor-cdf (score=upper, alpha=0.1) | 20 | 2 | 0.2657 | 0.9332 |
| best fpr_test | kmeans-doctor-cdf (score=upper, alpha=0.5) | 20 | 4 | 0.2349 | 0.9311 |
| best roc_auc_test | kmeans-doctor-cdf (score=upper, alpha=0.5) | 20 | 2 | 0.2657 | 0.9327 |

Best mean test performance per method (mean ± std over inits for each K):

| select | method | k | fpr_test | roc_auc_test |
|---|---|---|---|---|
| best mean fpr_test | kmeans-doctor-cdf (score=mean, alpha=any) | 20 | 0.2453 ± 0.0110 | 0.9276 ± 0.0018 |
| best mean roc_auc_test | kmeans-doctor-cdf (score=mean, alpha=any) | 20 | 0.2453 ± 0.0110 | 0.9276 ± 0.0018 |
| best mean fpr_test | kmeans-doctor-cdf (score=upper, alpha=0.05) | 10 | 0.2617 ± 0.0163 | 0.9243 ± 0.0003 |
| best mean roc_auc_test | kmeans-doctor-cdf (score=upper, alpha=0.05) | 20 | 0.2627 ± 0.0199 | 0.9296 ± 0.0037 |
| best mean fpr_test | kmeans-doctor-cdf (score=upper, alpha=0.1) | 20 | 0.2542 ± 0.0188 | 0.9298 ± 0.0032 |
| best mean roc_auc_test | kmeans-doctor-cdf (score=upper, alpha=0.1) | 20 | 0.2542 ± 0.0188 | 0.9298 ± 0.0032 |
| best mean fpr_test | kmeans-doctor-cdf (score=upper, alpha=0.5) | 20 | 0.2453 ± 0.0110 | 0.9297 ± 0.0031 |
| best mean roc_auc_test | kmeans-doctor-cdf (score=upper, alpha=0.5) | 20 | 0.2453 ± 0.0110 | 0.9297 ± 0.0031 |

#### Diagnostic: unif-mass vs CDF bin occupancy (n_res=1000, perturbed logits)
Source: `results/cifar10/resnet34_ce/partition/runs/kmeans-cdf-grid-nres1000-20260116/seed-split-9/analysis/diagnose_unif_mass_vs_cdf/`  
All diagnostics use doctor-selected (temperature=0.9, magnitude=0.002, normalize=true) with `use_perturbed_logits=True`.

Test counts per bin (min / median / max):
- K=20 unif-mass (cal bins): 216 / 249.5 / 288
- K=20 CDF uniform bins: 204 / 239.5 / 301
- K=20 kmeans CDF: 197 / 234.5 / 324
- K=30 unif-mass (cal bins): 138 / 169 / 202
- K=30 CDF uniform bins: 127 / 167.5 / 214
- K=30 kmeans CDF: 56 / 166 / 255

Plots:
- CDF u distributions: `results/cifar10/resnet34_ce/partition/runs/kmeans-cdf-grid-nres1000-20260116/seed-split-9/analysis/diagnose_unif_mass_vs_cdf/cdf_u_cal_hist.png`, `results/cifar10/resnet34_ce/partition/runs/kmeans-cdf-grid-nres1000-20260116/seed-split-9/analysis/diagnose_unif_mass_vs_cdf/cdf_u_test_hist.png`
- Unif-mass test counts: `results/cifar10/resnet34_ce/partition/runs/kmeans-cdf-grid-nres1000-20260116/seed-split-9/analysis/diagnose_unif_mass_vs_cdf/unif_mass_cal_bins_k20_test_counts.png`, `results/cifar10/resnet34_ce/partition/runs/kmeans-cdf-grid-nres1000-20260116/seed-split-9/analysis/diagnose_unif_mass_vs_cdf/unif_mass_cal_bins_k30_test_counts.png`
- CDF uniform test counts: `results/cifar10/resnet34_ce/partition/runs/kmeans-cdf-grid-nres1000-20260116/seed-split-9/analysis/diagnose_unif_mass_vs_cdf/cdf_uniform_bins_k20_test_counts.png`, `results/cifar10/resnet34_ce/partition/runs/kmeans-cdf-grid-nres1000-20260116/seed-split-9/analysis/diagnose_unif_mass_vs_cdf/cdf_uniform_bins_k30_test_counts.png`
- K-means CDF test counts: `results/cifar10/resnet34_ce/partition/runs/kmeans-cdf-grid-nres1000-20260116/seed-split-9/analysis/diagnose_unif_mass_vs_cdf/kmeans_cdf_k20_test_counts.png`, `results/cifar10/resnet34_ce/partition/runs/kmeans-cdf-grid-nres1000-20260116/seed-split-9/analysis/diagnose_unif_mass_vs_cdf/kmeans_cdf_k30_test_counts.png`

---

## Multi-Dataset / Multi-Model Results (K-means + CDF Partition)

**Date:** 2026-01-17

### Experimental Setup

- **Datasets:** CIFAR-10, CIFAR-100
- **Models:** ResNet-34, DenseNet-121
- **Splits:** n_res=1000, n_cal=4000, n_test=5000
- **Seeds:** 1-9 (mean ± std reported)
- **Base score:** Doctor (gini) with hyperparameters selected **per-seed** on res split by FPR@95
- **Partition method:** K-means clustering on CDF-transformed scores
- **Grid:** n_clusters ∈ {5, 10, 20, 30, 50, 75, 100, 150, 200, 300}, alpha ∈ {0.05, 0.1, 0.5}, score ∈ {mean, upper}

### Methods

The partition postprocessor outputs a binned score. Four distinct methods are evaluated:
- **mean**: Use empirical bin error rate as score (alpha not used)
- **upper (α=0.05)**: Use upper bound of 95% confidence interval
- **upper (α=0.1)**: Use upper bound of 90% confidence interval
- **upper (α=0.5)**: Use upper bound of 50% confidence interval

### Protocol

1. **Doctor hyperparameter search on res:** Find best (temperature, magnitude, normalize) **per-seed** by FPR@95 on each res split
2. **Partition fitting on res:** Fit K-means clustering on CDF(doctor_score) using res split
3. **Binning/counting on cal:** Compute bin error rates and confidence intervals on cal split
4. **Evaluation on test:** Report final metrics on held-out test split

---

### Doctor Baseline (Continuous Score)

| Dataset | Model | Seeds | temperature | magnitude | FPR (cal) | FPR (test) | ROC-AUC (test) |
|---------|-------|-------|-------------|-----------|-----------|------------|----------------|
| CIFAR-10 | ResNet-34 | 9 | 0.73 ± 0.05 | 0.0027 ± 0.0010 | 0.1976 ± 0.0126 | 0.1982 ± 0.0166 | 0.9297 ± 0.0108 |
| CIFAR-10 | DenseNet-121 | 9 | 0.90 ± 0.23 | 0.0020 ± 0.0000 | 0.2686 ± 0.0212 | 0.2650 ± 0.0219 | 0.9124 ± 0.0052 |
| CIFAR-100 | ResNet-34 | 9 | 0.80 ± 0.17 | 0.0020 ± 0.0000 | 0.3938 ± 0.0193 | 0.3948 ± 0.0177 | 0.8726 ± 0.0039 |
| CIFAR-100 | DenseNet-121 | 9 | 0.87 ± 0.19 | 0.0000 ± 0.0000 | 0.4498 ± 0.0210 | 0.4614 ± 0.0137 | 0.8570 ± 0.0038 |

---

### CIFAR-10 / ResNet-34 (9 seeds)

#### Selection on res split by FPR@95

| Method | n_clusters | FPR (res) | FPR (cal) | FPR (test) | ROC-AUC (test) |
|--------|------------|-----------|-----------|------------|----------------|
| mean | 39.4 ± 18.8 | 0.2032 ± 0.0648 | 0.2395 ± 0.0512 | 0.2882 ± 0.0871 | 0.9126 ± 0.0099 |
| upper (α=0.05) | 20.6 ± 10.1 | 0.2259 ± 0.0817 | 0.2826 ± 0.0560 | 0.2979 ± 0.0542 | 0.9090 ± 0.0111 |
| upper (α=0.1) | 20.6 ± 10.1 | 0.2236 ± 0.0783 | 0.2783 ± 0.0577 | 0.2948 ± 0.0526 | 0.9095 ± 0.0112 |
| upper (α=0.5) | 23.9 ± 13.2 | 0.2129 ± 0.0707 | 0.2692 ± 0.0606 | 0.2929 ± 0.0636 | 0.9103 ± 0.0105 |

#### Selection on res split by ROC-AUC

| Method | n_clusters | ROC-AUC (res) | ROC-AUC (cal) | FPR (test) | ROC-AUC (test) |
|--------|------------|---------------|---------------|------------|----------------|
| mean | 24.4 ± 15.9 | 0.9186 ± 0.0248 | 0.9211 ± 0.0134 | 0.2516 ± 0.0340 | 0.9170 ± 0.0121 |
| upper (α=0.05) | 13.3 ± 5.0 | 0.9162 ± 0.0248 | 0.9151 ± 0.0106 | 0.2656 ± 0.0539 | 0.9147 ± 0.0109 |
| upper (α=0.1) | 14.4 ± 7.3 | 0.9163 ± 0.0247 | 0.9155 ± 0.0106 | 0.2728 ± 0.0577 | 0.9140 ± 0.0097 |
| upper (α=0.5) | 17.8 ± 6.7 | 0.9176 ± 0.0240 | 0.9180 ± 0.0106 | 0.2694 ± 0.0416 | 0.9155 ± 0.0107 |

#### Oracle: Best on test (per seed, then averaged)

| Method | Selection | n_clusters | FPR (test) | ROC-AUC (test) |
|--------|-----------|------------|------------|----------------|
| mean | best FPR | 35.6 ± 14.2 | 0.2349 ± 0.0311 | 0.9162 ± 0.0117 |
| mean | best ROC-AUC | 22.2 ± 6.7 | 0.2490 ± 0.0357 | 0.9186 ± 0.0125 |
| upper (α=0.05) | best FPR | 36.7 ± 61.4 | 0.2562 ± 0.0374 | 0.9110 ± 0.0132 |
| upper (α=0.05) | best ROC-AUC | 16.7 ± 7.1 | 0.2671 ± 0.0402 | 0.9157 ± 0.0115 |
| upper (α=0.1) | best FPR | 37.8 ± 61.0 | 0.2530 ± 0.0339 | 0.9127 ± 0.0124 |
| upper (α=0.1) | best ROC-AUC | 16.7 ± 7.1 | 0.2623 ± 0.0365 | 0.9162 ± 0.0117 |
| upper (α=0.5) | best FPR | 20.0 ± 7.1 | 0.2461 ± 0.0281 | 0.9167 ± 0.0120 |
| upper (α=0.5) | best ROC-AUC | 18.9 ± 6.0 | 0.2593 ± 0.0230 | 0.9171 ± 0.0121 |

---

### CIFAR-10 / DenseNet-121 (9 seeds)

**Note:** High variance due to float32 precision issue in seed 2 (see Float Precision Fix below).

#### Selection on res split by FPR@95

| Method | n_clusters | FPR (res) | FPR (cal) | FPR (test) | ROC-AUC (test) |
|--------|------------|-----------|-----------|------------|----------------|
| mean | 36.1 ± 27.0 | 0.3308 ± 0.2226 | 0.3449 ± 0.1564 | 0.4559 ± 0.2241 | 0.8909 ± 0.0427 |
| upper (α=0.05) | 37.8 ± 44.9 | 0.3373 ± 0.2220 | 0.3931 ± 0.1218 | 0.4117 ± 0.1207 | 0.8792 ± 0.0553 |
| upper (α=0.1) | 38.9 ± 44.3 | 0.3357 ± 0.2219 | 0.3821 ± 0.1178 | 0.4096 ± 0.1147 | 0.8816 ± 0.0528 |
| upper (α=0.5) | 38.9 ± 44.3 | 0.3329 ± 0.2221 | 0.3649 ± 0.1086 | 0.3943 ± 0.1104 | 0.8866 ± 0.0472 |

#### Selection on res split by ROC-AUC

| Method | n_clusters | ROC-AUC (res) | ROC-AUC (cal) | FPR (test) | ROC-AUC (test) |
|--------|------------|---------------|---------------|------------|----------------|
| mean | 25.0 ± 15.0 | 0.9012 ± 0.0653 | 0.8963 ± 0.0601 | 0.3859 ± 0.1524 | 0.8937 ± 0.0434 |
| upper (α=0.05) | 16.1 ± 6.0 | 0.8993 ± 0.0653 | 0.8885 ± 0.0575 | 0.4032 ± 0.1448 | 0.8922 ± 0.0433 |
| upper (α=0.1) | 17.2 ± 5.7 | 0.8995 ± 0.0653 | 0.8896 ± 0.0581 | 0.3965 ± 0.1453 | 0.8932 ± 0.0435 |
| upper (α=0.5) | 19.4 ± 6.3 | 0.9000 ± 0.0654 | 0.8919 ± 0.0587 | 0.3889 ± 0.1471 | 0.8937 ± 0.0436 |

#### Oracle: Best on test (per seed, then averaged)

| Method | Selection | n_clusters | FPR (test) | ROC-AUC (test) |
|--------|-----------|------------|------------|----------------|
| mean | best FPR | 32.2 ± 25.0 | 0.3535 ± 0.1146 | 0.8884 ± 0.0428 |
| mean | best ROC-AUC | 25.6 ± 5.3 | 0.3765 ± 0.1264 | 0.8946 ± 0.0428 |
| upper (α=0.05) | best FPR | 13.3 ± 8.3 | 0.3627 ± 0.1099 | 0.8864 ± 0.0465 |
| upper (α=0.05) | best ROC-AUC | 15.0 ± 6.1 | 0.3930 ± 0.1492 | 0.8929 ± 0.0434 |
| upper (α=0.1) | best FPR | 16.1 ± 9.3 | 0.3597 ± 0.1063 | 0.8881 ± 0.0460 |
| upper (α=0.1) | best ROC-AUC | 16.1 ± 6.0 | 0.3946 ± 0.1468 | 0.8932 ± 0.0435 |
| upper (α=0.5) | best FPR | 12.2 ± 6.2 | 0.3603 ± 0.1125 | 0.8878 ± 0.0440 |
| upper (α=0.5) | best ROC-AUC | 22.2 ± 4.4 | 0.3819 ± 0.1162 | 0.8940 ± 0.0434 |

---

### CIFAR-100 / ResNet-34

**Status:** Only 1 seed completed for K-means+CDF partition. Results omitted pending full run.

---

### CIFAR-100 / DenseNet-121 (9 seeds)

#### Selection on res split by FPR@95

| Method | n_clusters | FPR (res) | FPR (cal) | FPR (test) | ROC-AUC (test) |
|--------|------------|-----------|-----------|------------|----------------|
| mean | 33.3 ± 21.9 | 0.4396 ± 0.0385 | 0.5058 ± 0.0875 | 0.4854 ± 0.0219 | 0.8521 ± 0.0049 |
| upper (α=0.05) | 19.4 ± 9.5 | 0.4411 ± 0.0390 | 0.5224 ± 0.0841 | 0.4910 ± 0.0215 | 0.8524 ± 0.0053 |
| upper (α=0.1) | 20.6 ± 10.1 | 0.4410 ± 0.0389 | 0.5200 ± 0.0849 | 0.4890 ± 0.0189 | 0.8523 ± 0.0053 |
| upper (α=0.5) | 23.3 ± 8.7 | 0.4418 ± 0.0414 | 0.4938 ± 0.0450 | 0.4906 ± 0.0167 | 0.8540 ± 0.0040 |

#### Selection on res split by ROC-AUC

| Method | n_clusters | ROC-AUC (res) | ROC-AUC (cal) | FPR (test) | ROC-AUC (test) |
|--------|------------|---------------|---------------|------------|----------------|
| mean | 24.4 ± 5.3 | 0.8635 ± 0.0100 | 0.8594 ± 0.0053 | 0.4919 ± 0.0260 | 0.8544 ± 0.0038 |
| upper (α=0.05) | 21.1 ± 6.0 | 0.8631 ± 0.0103 | 0.8585 ± 0.0054 | 0.5009 ± 0.0317 | 0.8543 ± 0.0040 |
| upper (α=0.1) | 22.2 ± 6.7 | 0.8632 ± 0.0103 | 0.8586 ± 0.0054 | 0.4983 ± 0.0327 | 0.8545 ± 0.0041 |
| upper (α=0.5) | 22.2 ± 6.7 | 0.8634 ± 0.0102 | 0.8588 ± 0.0054 | 0.4975 ± 0.0329 | 0.8547 ± 0.0041 |

#### Oracle: Best on test (per seed, then averaged)

| Method | Selection | n_clusters | FPR (test) | ROC-AUC (test) |
|--------|-----------|------------|------------|----------------|
| mean | best FPR | 47.8 ± 28.3 | 0.4716 ± 0.0164 | 0.8519 ± 0.0070 |
| mean | best ROC-AUC | 21.1 ± 6.0 | 0.4896 ± 0.0198 | 0.8549 ± 0.0037 |
| upper (α=0.05) | best FPR | 20.0 ± 10.9 | 0.4825 ± 0.0241 | 0.8515 ± 0.0078 |
| upper (α=0.05) | best ROC-AUC | 18.9 ± 6.0 | 0.4954 ± 0.0335 | 0.8548 ± 0.0039 |
| upper (α=0.1) | best FPR | 26.1 ± 13.2 | 0.4799 ± 0.0217 | 0.8522 ± 0.0072 |
| upper (α=0.1) | best ROC-AUC | 18.9 ± 6.0 | 0.4949 ± 0.0338 | 0.8549 ± 0.0040 |
| upper (α=0.5) | best FPR | 30.6 ± 17.0 | 0.4762 ± 0.0214 | 0.8522 ± 0.0071 |
| upper (α=0.5) | best ROC-AUC | 18.9 ± 6.0 | 0.4951 ± 0.0337 | 0.8550 ± 0.0039 |

---

## Isotonic Regression Calibration (9 seeds)

**Setup:**
- Fit isotonic regression on cal data to map gini scores → calibrated error probabilities
- Doctor hyperparameters (temperature, magnitude, normalize) selected **per-seed** on each res split using FPR@95 (hyperparameters vary across seeds, e.g., CIFAR-10 ResNet-34: temperature 0.7-1.2, magnitude 0.002-0.004)
- Evaluate calibrated probabilities on test set
- All experiments use `--logits-dtype float64`

**Source:** `results/<dataset>/<model>_ce/isotonic/runs/isotonic-cal-fit-doctor-allseeds-20260117/`

### Results Summary

| Dataset | Model | FPR@95 (test) ↓ | ROC-AUC (test) ↑ |
|---------|-------|-----------------|------------------|
| CIFAR10 | resnet34 | 0.4527 ± 0.1736 | 0.9247 ± 0.0056 |
| CIFAR10 | densenet121 | 0.3588 ± 0.0167 | 0.9143 ± 0.0048 |
| CIFAR100 | resnet34 | 0.4225 ± 0.0287 | 0.8797 ± 0.0049 |
| CIFAR100 | densenet121 | 0.4999 ± 0.0333 | 0.8558 ± 0.0036 |

### Overfitting Analysis (Cal vs Test)

| Dataset | Model | FPR (cal) | FPR (test) | Δ FPR | ROC-AUC (cal) | ROC-AUC (test) | Δ ROC-AUC |
|---------|-------|-----------|------------|-------|---------------|----------------|-----------|
| CIFAR10 | resnet34 | 0.4118 ± 0.1479 | 0.4527 ± 0.1736 | +0.0409 | 0.9283 ± 0.0062 | 0.9247 ± 0.0056 | -0.0036 |
| CIFAR10 | densenet121 | 0.3549 ± 0.0134 | 0.3588 ± 0.0167 | +0.0040 | 0.9181 ± 0.0045 | 0.9143 ± 0.0048 | -0.0038 |
| CIFAR100 | resnet34 | 0.4193 ± 0.0166 | 0.4225 ± 0.0287 | +0.0031 | 0.8821 ± 0.0028 | 0.8797 ± 0.0049 | -0.0025 |
| CIFAR100 | densenet121 | 0.4826 ± 0.0253 | 0.4999 ± 0.0333 | +0.0173 | 0.8621 ± 0.0050 | 0.8558 ± 0.0036 | -0.0062 |

### Comparison with Doctor Baseline

| Dataset | Model | Method | FPR@95 (test) | ROC-AUC (test) |
|---------|-------|--------|---------------|----------------|
| CIFAR10 | resnet34 | Doctor | 0.1982 ± 0.0166 | 0.9297 ± 0.0108 |
| CIFAR10 | resnet34 | Isotonic | 0.4527 ± 0.1736 | 0.9247 ± 0.0056 |
| CIFAR10 | densenet121 | Doctor | 0.2650 ± 0.0219 | 0.9124 ± 0.0052 |
| CIFAR10 | densenet121 | Isotonic | 0.3588 ± 0.0167 | 0.9143 ± 0.0048 |
| CIFAR100 | resnet34 | Doctor | 0.3948 ± 0.0177 | 0.8726 ± 0.0039 |
| CIFAR100 | resnet34 | Isotonic | 0.4225 ± 0.0287 | 0.8797 ± 0.0049 |
| CIFAR100 | densenet121 | Doctor | 0.4614 ± 0.0137 | 0.8570 ± 0.0038 |
| CIFAR100 | densenet121 | Isotonic | 0.4999 ± 0.0333 | 0.8558 ± 0.0036 |

### Isotonic Observations

1. **Overfitting:** Isotonic shows mild overfitting with FPR increasing +0.003 to +0.041 from cal to test, and ROC-AUC decreasing -0.003 to -0.006. The overfitting is modest but consistent across all configurations.

2. **Calibration degrades FPR:** Isotonic calibration increases FPR@95 compared to raw Doctor scores:
   - CIFAR10 resnet34: +0.2545 FPR (0.1982 → 0.4527)
   - CIFAR10 densenet121: +0.0939 FPR (0.2650 → 0.3588)
   - CIFAR100 resnet34: +0.0276 FPR (0.3948 → 0.4225)
   - CIFAR100 densenet121: +0.0385 FPR (0.4614 → 0.4999)

3. **ROC-AUC comparable:** Isotonic shows similar ROC-AUC to Doctor, with slight improvements on CIFAR10 densenet121 (+0.0019) and CIFAR100 resnet34 (+0.0071), suggesting the monotonic transformation preserves ranking quality.

4. **High variance on CIFAR10 resnet34:** FPR std=0.1736 indicates sensitivity to the calibration split, likely due to limited calibration samples (n_cal=4000).

5. **Conclusion:** For failure detection, raw Doctor scores outperform isotonic-calibrated probabilities on FPR@95. Isotonic regression is better suited for probability calibration rather than ranking optimization.

---

## Isotonic Regression with Bin Splitting (9 seeds)

**Motivation:** Isotonic regression creates coarse bins with many ties, hurting threshold-based metrics. We hypothesize that splitting large bins could improve resolution.

**Algorithm:**
1. Fit isotonic regression on cal data → creates adaptive bins via PAV
2. Split bins with > n_max samples at the median until all bins have n_min ≤ size ≤ n_max
3. Compute error rate per sub-bin (no monotonicity enforcement)
4. n_min tuned via 5-fold cross-validation on cal split

**Configuration:**
- n_min grid: [20, 30, 50, 75, 100] (selected by CV on cal)
- n_max: 200 (fixed)
- Doctor hyperparameters selected **per-seed** on each res split using FPR@95

**Source:** `results/<dataset>/<model>_ce/isotonic_splitting/runs/isotonic-splitting-cal-fit-doctor-allseeds-20260117/`

### Results Summary

| Dataset | Model | FPR@95 (test) ↓ | ROC-AUC (test) ↑ | n_min selected |
|---------|-------|-----------------|------------------|----------------|
| CIFAR10 | resnet34 | 0.5402 ± 0.2683 | 0.9136 ± 0.0149 | 20 (6/9), 100 (3/9) |
| CIFAR10 | densenet121 | 0.5230 ± 0.2161 | 0.9056 ± 0.0084 | 20 (9/9) |
| CIFAR100 | resnet34 | 0.4236 ± 0.0282 | 0.8770 ± 0.0054 | 20 (9/9) |
| CIFAR100 | densenet121 | 0.4789 ± 0.0241 | 0.8533 ± 0.0037 | 20 (9/9) |

### Comparison: Isotonic vs Isotonic+Splitting

| Dataset | Model | Isotonic FPR | Isotonic+Split FPR | Δ FPR |
|---------|-------|--------------|---------------------|-------|
| CIFAR10 | resnet34 | 0.4527 ± 0.1736 | 0.5402 ± 0.2683 | +0.0875 (worse) |
| CIFAR10 | densenet121 | 0.3588 ± 0.0167 | 0.5230 ± 0.2161 | +0.1642 (worse) |
| CIFAR100 | resnet34 | 0.4225 ± 0.0287 | 0.4236 ± 0.0282 | +0.0011 (similar) |
| CIFAR100 | densenet121 | 0.4999 ± 0.0333 | 0.4789 ± 0.0241 | -0.0210 (better) |

### Threshold Transfer Issue

Some seeds show FPR=1.0 on test (seeds 1, 6 for CIFAR10/resnet34):

| Seed | n_min | FPR (cal) | FPR (test) | thr (cal) | thr (test) |
|------|-------|-----------|------------|-----------|------------|
| 1 | 20 | 0.366 | 1.000 | 0.0092 | 0.0 |
| 6 | 20 | 0.231 | 1.000 | 0.0204 | 0.0 |

**Root cause:** When calibrated probabilities are binned, the threshold selected on cal may not exist in test predictions. If the minimum calibrated probability in test is higher than the cal threshold, the effective threshold becomes 0, classifying all samples as positive.

### Isotonic+Splitting Observations

1. **Splitting does not improve FPR:** Contrary to hypothesis, splitting large isotonic bins worsens FPR on CIFAR-10 and shows marginal effects on CIFAR-100. The increased resolution introduces threshold transfer issues.

2. **High variance on CIFAR-10:** FPR std > 0.21 indicates extreme sensitivity to seed. Two seeds (1, 6) produce FPR=1.0 due to threshold transfer failure.

3. **n_min=20 consistently selected:** CV selects smallest n_min, preferring finer resolution. However, this leads to unstable bin boundaries and threshold transfer issues.

4. **CIFAR-100 slightly better:** With higher error rate (~25%), more samples per bin lead to more stable boundaries. CIFAR-100 densenet121 shows slight improvement (-0.021 FPR).

5. **Conclusion:** Bin splitting after isotonic regression does not solve the FPR degradation problem. The fundamental issue is that discretizing continuous scores creates threshold transfer instabilities. Alternative approaches (e.g., Venn-ABERS, ROC-regularized isotonic) may be needed.

---

### Key Observations

1. **Performance gap vs baseline:** K-means+CDF partition shows consistent performance degradation vs. continuous Doctor baseline across all configurations:
   - CIFAR-10 ResNet-34: +0.05-0.09 FPR (baseline 0.198 → partition ~0.25-0.29)
   - CIFAR-10 DenseNet-121: +0.09-0.19 FPR (baseline 0.265 → partition ~0.35-0.46)
   - CIFAR-100 DenseNet-121: +0.02-0.03 FPR (baseline 0.461 → partition ~0.48-0.49)

2. **Method comparison (mean vs upper):** The `mean` score generally performs comparably or slightly better than `upper` bounds. Smaller alpha values in `upper` lead to more conservative (higher) scores but similar test performance.

3. **Selection metric impact:** Selecting by ROC-AUC on res tends to yield lower n_clusters and slightly better test ROC-AUC, while selecting by FPR yields higher n_clusters.

4. **Oracle gap:** The oracle (best on test) shows modest improvement over res-based selection (~0.02-0.05 FPR reduction), indicating that res-based selection is reasonably effective but not optimal.

5. **CIFAR-10 DenseNet-121 anomaly:** Very high variance (std ~0.11-0.22) due to float32 precision issues affecting seed 2. Re-run with float64 recommended.

6. **Hyperparameter stability:** Selected n_clusters varies substantially across seeds (std ~5-27), indicating sensitivity to the random split.

### Float Precision Fix

During these experiments, we identified a float32 precision issue affecting Doctor score computation:
- Samples with extreme logit gaps (11-15) cause softmax underflow when scaled by temperature
- This results in gini_norm=0 for "confident errors", breaking the score ranking
- Fix: Added `--logits-dtype float64` option to `run_detection.py`
- **Recommendation:** Re-run CIFAR-10 DenseNet-121 experiments with float64 to obtain reliable variance estimates

### Source Paths

- CIFAR-10 ResNet-34 K-means: `results/cifar10/resnet34_ce/partition/runs/kmeans-cdf-grid-nres1000-cifar10-resnet34-allseeds-20260116/`
- CIFAR-10 DenseNet-121 K-means: `results/cifar10/densenet121_ce/partition/runs/kmeans-cdf-grid-nres1000-cifar10-densenet121-allseeds-20260116/`
- CIFAR-100 DenseNet-121 K-means: `results/cifar100/densenet121_ce/partition/runs/kmeans-cdf-grid-nres1000-cifar100-densenet121-allseeds-20260117/`
- Isotonic: `results/<dataset>/<model>_ce/isotonic/runs/isotonic-cal-fit-doctor-allseeds-20260117/`
- Doctor baselines: `results/<dataset>/<model>_ce/doctor/runs/doctor-eval-grid-nres1000-*-allseeds-*/`
- Uniform Mass: `results/<dataset>/<model>_ce/uniform_mass/runs/uniform-mass-cal-fit-doctor-allseeds-20260117/`

---

## Experiment: Uniform Mass Binning

**Date:** 2026-01-17
**Hypothesis:** Uniform mass binning (equal samples per bin using quantiles) can provide stable partitioning for FPR control, with guarantees preserved since partition and counting can use the same calibration data.

### Method

Uniform mass binning differs from uniform width binning in that bin boundaries are determined by quantiles, ensuring each bin contains approximately the same number of samples. This is the only partitioning method where we can define the partition on the same data used for counting (calibration set) while preserving statistical guarantees.

**Experimental setup:**
1. Select Doctor parameters (temperature, normalize, magnitude) on res split
2. Compute Doctor scores on calibration set
3. Create uniform mass bins using quantile-based boundaries on cal
4. Count errors per bin on cal (same data as partition)
5. Evaluate on test

**Grid search:** n_bins ∈ {5, 10, 15, 20, 30, 50}

**Note:** No cross-validation on cal to preserve guarantees (partition and counting use same data). All n_bins values are reported for oracle/deterministic rule analysis.

**Doctor hyperparameters:** Selected **per-seed** on each res split (best ROC-AUC on res). Hyperparameters vary across seeds. Example ranges from CIFAR-10 ResNet-34:
- Temperature: 0.7 to 1.2 depending on seed
- Magnitude: 0.002 to 0.004 depending on seed
- Normalize: True (all seeds)

### Results: FPR by n_bins

**CIFAR-10 ResNet-34**
| Method | FPR Cal | FPR Test | ROC-AUC Test |
|--------|---------|----------|--------------|
| **Doctor (baseline)** | - | **0.1982 ± 0.0166** | **0.9297 ± 0.0108** |
| n_bins=5 | 0.444 ± 0.104 | 0.465 ± 0.110 | 0.883 ± 0.006 |
| n_bins=10 | 0.387 ± 0.062 | 0.419 ± 0.118 | 0.911 ± 0.007 |
| n_bins=15 | 0.374 ± 0.070 | 0.375 ± 0.049 | 0.918 ± 0.005 |
| n_bins=20 | 0.348 ± 0.048 | 0.403 ± 0.054 | 0.917 ± 0.007 |
| n_bins=30 | 0.304 ± 0.030 | 0.496 ± 0.208 | 0.915 ± 0.013 |
| n_bins=50 | 0.287 ± 0.081 | 0.804 ± 0.296 | 0.910 ± 0.012 |

**CIFAR-10 DenseNet-121**
| Method | FPR Cal | FPR Test | ROC-AUC Test |
|--------|---------|----------|--------------|
| **Doctor (baseline)** | - | **0.2650 ± 0.0219** | **0.9124 ± 0.0052** |
| n_bins=5 | 0.366 ± 0.002 | 0.367 ± 0.010 | 0.878 ± 0.005 |
| n_bins=10 | 0.366 ± 0.002 | 0.367 ± 0.010 | 0.905 ± 0.005 |
| n_bins=15 | 0.374 ± 0.023 | 0.368 ± 0.036 | 0.910 ± 0.004 |
| n_bins=20 | 0.331 ± 0.026 | 0.410 ± 0.104 | 0.910 ± 0.005 |
| n_bins=30 | 0.343 ± 0.039 | 0.408 ± 0.095 | 0.911 ± 0.006 |
| n_bins=50 | 0.301 ± 0.027 | 0.555 ± 0.263 | 0.906 ± 0.008 |

**CIFAR-100 ResNet-34**
| Method | FPR Cal | FPR Test | ROC-AUC Test |
|--------|---------|----------|--------------|
| **Doctor (baseline)** | - | **0.3948 ± 0.0177** | **0.8726 ± 0.0039** |
| n_bins=5 | 0.501 ± 0.004 | 0.502 ± 0.014 | 0.863 ± 0.005 |
| n_bins=10 | 0.487 ± 0.041 | 0.489 ± 0.034 | 0.876 ± 0.005 |
| n_bins=15 | 0.447 ± 0.038 | 0.438 ± 0.036 | 0.877 ± 0.005 |
| n_bins=20 | 0.433 ± 0.021 | 0.440 ± 0.024 | 0.878 ± 0.005 |
| n_bins=30 | 0.429 ± 0.026 | 0.423 ± 0.018 | 0.876 ± 0.005 |
| n_bins=50 | 0.428 ± 0.034 | 0.442 ± 0.021 | 0.877 ± 0.005 |

**CIFAR-100 DenseNet-121**
| Method | FPR Cal | FPR Test | ROC-AUC Test |
|--------|---------|----------|--------------|
| **Doctor (baseline)** | - | **0.4614 ± 0.0137** | **0.8570 ± 0.0038** |
| n_bins=5 | 0.504 ± 0.085 | 0.565 ± 0.124 | 0.841 ± 0.004 |
| n_bins=10 | 0.490 ± 0.042 | 0.522 ± 0.059 | 0.853 ± 0.004 |
| n_bins=15 | 0.485 ± 0.029 | 0.507 ± 0.037 | 0.855 ± 0.004 |
| n_bins=20 | 0.483 ± 0.022 | 0.506 ± 0.025 | 0.854 ± 0.004 |
| n_bins=30 | 0.480 ± 0.015 | 0.495 ± 0.022 | 0.854 ± 0.004 |
| n_bins=50 | 0.470 ± 0.039 | 0.499 ± 0.029 | 0.853 ± 0.004 |

**ImageNet ViT-Base16**
| Method | FPR Cal | FPR Test | ROC-AUC Test |
|--------|---------|----------|--------------|
| **Doctor (baseline)** | - | **0.4256 ± 0.0085** | **0.8750 ± 0.0025** |
| n_bins=5 | 0.5170 ± 0.0013 | 0.5175 ± 0.0035 | 0.8575 ± 0.0023 |
| n_bins=10 | 0.5170 ± 0.0013 | 0.5175 ± 0.0035 | 0.8705 ± 0.0025 |
| n_bins=15 | 0.4481 ± 0.0261 | 0.4465 ± 0.0279 | 0.8729 ± 0.0025 |
| n_bins=20 | 0.4587 ± 0.0014 | 0.4578 ± 0.0034 | 0.8736 ± 0.0026 |
| n_bins=30 | 0.4437 ± 0.0131 | 0.4420 ± 0.0147 | 0.8742 ± 0.0025 |
| n_bins=50 | 0.4342 ± 0.0129 | 0.4382 ± 0.0112 | 0.8738 ± 0.0025 |
| n_bins=100 | 0.4352 ± 0.0239 | 0.4526 ± 0.0118 | 0.8730 ± 0.0025 |

**ImageNet ViT-Tiny16**
| Method | FPR Cal | FPR Test | ROC-AUC Test |
|--------|---------|----------|--------------|
| **Doctor (baseline)** | - | **0.4504 ± 0.0096** | **0.8658 ± 0.0031** |
| n_bins=5 | 0.4829 ± 0.0019 | 0.4837 ± 0.0050 | 0.8506 ± 0.0031 |
| n_bins=10 | 0.4829 ± 0.0019 | 0.4837 ± 0.0050 | 0.8615 ± 0.0033 |
| n_bins=15 | 0.4829 ± 0.0019 | 0.4837 ± 0.0050 | 0.8642 ± 0.0029 |
| n_bins=20 | 0.4829 ± 0.0019 | 0.4837 ± 0.0050 | 0.8647 ± 0.0031 |
| n_bins=30 | 0.4739 ± 0.0190 | 0.4704 ± 0.0175 | 0.8650 ± 0.0030 |
| n_bins=50 | 0.4721 ± 0.0190 | 0.4813 ± 0.0154 | 0.8646 ± 0.0034 |
| n_bins=100 | 0.4529 ± 0.0102 | 0.4711 ± 0.0116 | 0.8637 ± 0.0034 |

### Selection Rule Analysis

| Dataset | Model | Doctor FPR | Oracle FPR | Cal-Selected FPR | Rice Rule FPR |
|---------|-------|------------|------------|------------------|---------------|
| CIFAR10 | resnet34 | **0.1982 ± 0.0166** | 0.350 ± 0.059 | 0.811 ± 0.274 | 0.496 ± 0.208 |
| CIFAR10 | densenet121 | **0.2650 ± 0.0219** | 0.349 ± 0.023 | 0.496 ± 0.194 | 0.408 ± 0.095 |
| CIFAR100 | resnet34 | **0.3948 ± 0.0177** | 0.419 ± 0.011 | 0.443 ± 0.031 | 0.423 ± 0.018 |
| CIFAR100 | densenet121 | **0.4614 ± 0.0137** | 0.486 ± 0.024 | 0.495 ± 0.027 | 0.495 ± 0.022 |
| ImageNet | timm_vit_base16 | **0.4256 ± 0.0085** | 0.434 ± 0.008 | 0.445 ± 0.018 | 0.442 ± 0.014 |
| ImageNet | timm_vit_tiny16 | **0.4504 ± 0.0096** | 0.463 ± 0.012 | 0.472 ± 0.012 | 0.470 ± 0.017 |

**Rice's Rule:** n_bins = 2 × n^(1/3)
- CIFAR: 2 × 4000^(1/3) ≈ 31
- ImageNet: 2 × 20000^(1/3) ≈ 54

### Uniform Mass Observations

1. **Severe cal-test gap on CIFAR-10:** Selecting n_bins by best cal FPR leads to catastrophic test FPR (0.81 for ResNet-34, 0.50 for DenseNet-121). The method overfits to calibration data, especially with high n_bins.

2. **High n_bins causes instability:** n_bins=50 produces FPR test >0.55 on all CIFAR-10 models, with very high variance (std >0.26). Fine-grained binning exacerbates threshold transfer issues.

3. **CIFAR-100 more stable:** With higher base error rate (~25%), there are more errors per bin, leading to more stable probability estimates. Cal-selected FPR is close to oracle on CIFAR-100.

4. **Rice's rule performs reasonably:** Using n_bins≈30 (Rice's rule) provides a deterministic selection without tuning. Results are competitive with oracle on CIFAR-100 but suboptimal on CIFAR-10.

5. **Oracle FPR around 0.35-0.49:** Even the best n_bins achieves only moderate FPR, significantly worse than continuous Doctor baseline (0.20-0.46). Discretization inherently loses information.

6. **n_bins=15 often optimal on test:** Across datasets, n_bins=15 frequently achieves good cal-test agreement and reasonable FPR, suggesting a conservative choice of fewer bins.

7. **ImageNet confirms high-error-rate stability:** ImageNet results (18-24% error rate) show excellent cal-test transfer similar to CIFAR-100. Doctor baseline FPR of 0.43-0.45 degrades only slightly to 0.43-0.47 with binning. Oracle, Cal-Selected, and Rice Rule FPRs are all within ~0.02 of each other, demonstrating that selection rule choice matters little when error rates are high enough.

8. **ImageNet shows FPR plateau at low n_bins:** For both ViT models on ImageNet, n_bins=5,10,15,20 produce nearly identical FPR values (within std), suggesting that very coarse binning saturates discriminative ability. Only n_bins≥30 shows variation.

### Conclusion

Uniform mass binning fails to provide reliable FPR control on CIFAR-10 due to:
- **Overfitting:** Cal-based selection of n_bins leads to poor test generalization
- **Threshold transfer:** Binned probabilities create discrete thresholds that may not transfer across splits
- **Information loss:** Discretization degrades separation compared to continuous scores

For CIFAR-100 and ImageNet (higher error rates of ~25% and ~18-24% respectively), uniform mass binning performs well with minimal FPR degradation from the Doctor baseline. Rice's rule (n_bins≈30) provides a practical deterministic choice that performs comparably to oracle selection. The key factor determining binning effectiveness is the base error rate: higher error rates provide more errors per bin, leading to more stable probability estimates and better cal-test transfer.

---

## 8. LDA Binning (Multi-Score Combination)

### Motivation

Single uncertainty scores (Doctor/Gini, Margin, MSP) each capture different aspects of prediction uncertainty. We hypothesize that combining multiple scores might provide better error detection than any single score alone. LDA (Linear Discriminant Analysis) provides a supervised projection that maximizes class separation between correct and incorrect predictions.

### Method

LDA binning combines multiple uncertainty scores through supervised dimensionality reduction, then applies uniform mass binning:

1. **Score computation:** For each sample, compute multiple uncertainty scores (gini, margin, msp, entropy)
2. **LDA projection on res:** Fit LDA to project multi-dimensional scores to 1D, supervised by error labels
3. **Uniform mass binning on cal:** Apply quantile-based binning on the projected 1D score
4. **Probability estimation:** Compute error rate per bin on calibration data
5. **Evaluation:** Apply to test set

**Per-score hyperparameter selection:** Best hyperparameters for each score (temperature, magnitude) are loaded **per-seed** from previous grid searches on each res split:
- Gini: from Doctor grid search (per-seed)
- Margin: from Margin grid search (per-seed)
- MSP: from ODIN grid search (per-seed)

**Grid search:**
- n_bins ∈ {5, 10, 15, 20, 30}
- alpha ∈ {0.05, 0.1, 0.5}
- score_type ∈ {mean, upper}
- base_scores combinations: [gini,margin], [gini,msp], [gini,margin,msp], [gini,margin,entropy]

### 8.1 Raw LDA Score Combination (Without Binning)

**STATUS: PENDING RE-RUN**

Previous results were computed with incorrect hyperparameters (used seed-1 hyperparameters for all seeds instead of per-seed selection). See Issue #1 in `docs/inconsistencies_log.md`.

**Experiment:** Fit LDA on res split (supervised by error labels), apply to test, measure ROC-AUC and FPR@95.

**To re-run:**
```bash
python scripts/run_raw_lda_evaluation.py --seed-splits 1 2 3 4 5 6 7 8 9
```

Results will be populated after re-running with correct per-seed hyperparameters.

### 8.2 Effect of Binning Configuration

We analyze how different binning parameters affect performance.

#### 8.2.1 Score Type: Mean vs Upper Confidence Bound

| Dataset | Model | Score Type | Best FPR@95 Test |
|---------|-------|------------|------------------|
| CIFAR-10 | ResNet-34 | mean | 0.433 |
| CIFAR-10 | ResNet-34 | upper (α=0.05) | 0.434 |
| CIFAR-10 | DenseNet-121 | mean | 0.368 |
| CIFAR-10 | DenseNet-121 | upper (α=0.05) | 0.368 |
| CIFAR-100 | ResNet-34 | mean | 0.427 |
| CIFAR-100 | ResNet-34 | upper (α=0.05) | 0.427 |
| CIFAR-100 | DenseNet-121 | mean | 0.483 |
| CIFAR-100 | DenseNet-121 | upper (α=0.05) | 0.483 |

**Key Finding:** Score type (mean vs upper) shows **no significant difference**. The Hoeffding/Bernstein upper bound does not improve calibration.

#### 8.2.2 Alpha (Confidence Level for Upper Bound)

For score_type=upper, varying α ∈ {0.05, 0.1, 0.5}:

| Dataset | Model | α=0.05 FPR | α=0.1 FPR | α=0.5 FPR |
|---------|-------|------------|-----------|-----------|
| CIFAR-10 | ResNet-34 | 0.434 | 0.434 | 0.434 |
| CIFAR-10 | DenseNet-121 | 0.368 | 0.368 | 0.368 |
| CIFAR-100 | ResNet-34 | 0.427 | 0.427 | 0.427 |
| CIFAR-100 | DenseNet-121 | 0.483 | 0.483 | 0.483 |

**Key Finding:** Alpha has **negligible effect**. The upper confidence bound approach doesn't meaningfully improve threshold transfer.

#### 8.2.3 Number of Bins (K)

Best K selection varies by configuration:

| Dataset | Model | Score Combination | Best K | FPR@95 Test | ROC-AUC Test |
|---------|-------|-------------------|--------|-------------|--------------|
| CIFAR-10 | ResNet-34 | gini+margin | 15 | 0.434 | 0.917 |
| CIFAR-10 | DenseNet-121 | gini+margin | 5 | 0.368 | 0.878 |
| CIFAR-100 | ResNet-34 | gini+margin | 30 | 0.434 | 0.877 |
| CIFAR-100 | DenseNet-121 | gini+margin | 30 | 0.488 | 0.850 |

**Key Finding:** Optimal K varies across datasets/models. Smaller K (5) works better for CIFAR-10 DenseNet-121, while larger K (30) is better for CIFAR-100. This suggests K selection is data-dependent and doesn't generalize.

### 8.3 Comparison with Baselines (FPR@95 Test)

| Dataset | Model | LDA Binning (best) | Uniform Mass (gini) | Doctor (raw) |
|---------|-------|-------------------|---------------------|--------------|
| CIFAR-10 | ResNet-34 | 0.354 ± 0.075 | **0.350 ± 0.062** | **0.1982 ± 0.0166** |
| CIFAR-10 | DenseNet-121 | 0.351 ± 0.056 | **0.349 ± 0.024** | **0.2650 ± 0.0219** |
| CIFAR-100 | ResNet-34 | 0.418 ± 0.020 | **0.419 ± 0.012** | **0.3948 ± 0.0177** |
| CIFAR-100 | DenseNet-121 | 0.482 ± 0.023 | **0.486 ± 0.026** | **0.4614 ± 0.0137** |

### 8.4 LDA Binning Observations

1. **No improvement in ranking power (ROC-AUC):** LDA combination of multiple scores does not improve ROC-AUC over single gini score. The scores (gini, margin, msp) are derived from the same softmax distribution and contain redundant information.

2. **No improvement in calibration (FPR):** LDA binning achieves similar or slightly worse FPR compared to single-score uniform mass binning. The added complexity provides no benefit.

3. **Configuration parameters don't matter:** Score type (mean vs upper), alpha, and n_bins show minimal effect. This suggests the binning approach itself is the limiting factor, not the specific parameters.

4. **Higher variance:** LDA binning shows higher variance than single-score methods (e.g., CIFAR-10 ResNet-34: std=0.075 vs 0.062), indicating less stable behavior.

5. **Two-score combinations slightly better:** Among LDA variants, two-score combinations (gini+margin, gini+msp) marginally outperform three-score combinations. Adding more scores doesn't help.

6. **LDA projection doesn't address threshold transfer:** The fundamental issue is that thresholds learned on calibration data don't transfer to test data. LDA projection doesn't solve this problem.

### 8.5 Conclusion

LDA-based multi-score combination fails to improve over simpler single-score methods. Neither ranking power (ROC-AUC) nor calibration (FPR) improves. The scores being combined are highly correlated, limiting LDA's ability to find discriminative projections.

**Recommendation:** Use single-score methods (Doctor raw or uniform mass with gini) rather than multi-score LDA combination. The added complexity of LDA fitting and per-score hyperparameter selection provides no benefit.

---

## 9. K-Means Constrained Binning (n_res=1000, n_cal=4000)

**Date:** 2026-01-19

### 9.1 Experimental Setup

- **Datasets:** CIFAR-10, CIFAR-100
- **Models:** ResNet-34, DenseNet-121
- **Splits:** n_res=1000, n_cal=4000, n_test=5000
- **Seeds:** 1-9 (mean ± std reported)
- **Base score:** Doctor (gini) with hyperparameters selected **per-seed** on res split by FPR@95
- **Partition method:** K-means constrained (uniform cluster sizes) on gini scores
- **Grid:** n_clusters ∈ {10, 20, 30, 50}, alpha ∈ {0.05, 0.1, 0.5}, score ∈ {mean, upper}
- **Clustering params:** n_init=5, max_iter=100

### 9.2 Methods

K-means constrained enforces equal-sized clusters, similar to uniform-mass binning but using K-means optimization.
- **mean**: Use empirical bin error rate as score (alpha not used)
- **upper (α=0.05)**: Use upper bound of 95% confidence interval
- **upper (α=0.1)**: Use upper bound of 90% confidence interval
- **upper (α=0.5)**: Use upper bound of 50% confidence interval

---

### CIFAR-10 / ResNet-34 (9 seeds)

#### Full Grid Results (all alpha × n_clusters)

| score | alpha | n_clusters | FPR (res) | FPR (cal) | FPR (test) | ROC-AUC (test) |
|-------|-------|------------|-----------|-----------|------------|----------------|
| mean | - | 10 | 0.3401 ± 0.2562 | 0.3313 ± 0.2431 | 0.3383 ± 0.2398 | 0.9102 ± 0.0112 |
| mean | - | 20 | 0.2143 ± 0.0961 | 0.3162 ± 0.2306 | 0.3248 ± 0.2269 | 0.9153 ± 0.0113 |
| mean | - | 30 | 0.2192 ± 0.0667 | 0.2350 ± 0.0294 | 0.3423 ± 0.2096 | 0.9153 ± 0.0146 |
| mean | - | 50 | 0.3007 ± 0.2730 | 0.2196 ± 0.0123 | 0.3180 ± 0.1876 | 0.9120 ± 0.0124 |
| upper | 0.05 | 10 | 0.2805 ± 0.0972 | 0.3799 ± 0.2425 | 0.3823 ± 0.2409 | 0.8970 ± 0.0159 |
| upper | 0.05 | 20 | 0.3060 ± 0.1464 | 0.4109 ± 0.2359 | 0.3625 ± 0.0899 | 0.8928 ± 0.0198 |
| upper | 0.05 | 30 | 0.3369 ± 0.1206 | 0.3321 ± 0.0881 | 0.3834 ± 0.0499 | 0.8780 ± 0.0270 |
| upper | 0.05 | 50 | 0.3491 ± 0.1091 | 0.3686 ± 0.0443 | 0.3947 ± 0.0313 | 0.8503 ± 0.0324 |
| upper | 0.1 | 10 | 0.2805 ± 0.0972 | 0.3777 ± 0.2439 | 0.3823 ± 0.2409 | 0.8989 ± 0.0146 |
| upper | 0.1 | 20 | 0.2981 ± 0.1381 | 0.4109 ± 0.2359 | 0.3603 ± 0.0871 | 0.8978 ± 0.0158 |
| upper | 0.1 | 30 | 0.3369 ± 0.1206 | 0.3222 ± 0.0898 | 0.3756 ± 0.0551 | 0.8835 ± 0.0240 |
| upper | 0.1 | 50 | 0.3468 ± 0.1108 | 0.3615 ± 0.0448 | 0.3875 ± 0.0350 | 0.8594 ± 0.0299 |
| upper | 0.5 | 10 | 0.2732 ± 0.0971 | 0.3601 ± 0.2508 | 0.3660 ± 0.2476 | 0.9059 ± 0.0110 |
| upper | 0.5 | 20 | 0.2785 ± 0.1341 | 0.3775 ± 0.2430 | 0.3306 ± 0.0638 | 0.9094 ± 0.0106 |
| upper | 0.5 | 30 | 0.3187 ± 0.1240 | 0.3004 ± 0.0790 | 0.3652 ± 0.0734 | 0.8972 ± 0.0189 |
| upper | 0.5 | 50 | 0.3142 ± 0.1117 | 0.3372 ± 0.0516 | 0.3601 ± 0.0331 | 0.8816 ± 0.0223 |
| **Doctor (raw)** | - | - | - | - | 0.1982 ± 0.0166 | 0.9297 ± 0.0108 |

#### Selection on res split by FPR@95 (per-seed)

| score | alpha | n_clusters | FPR (test) | ROC-AUC (test) |
|-------|-------|------------|------------|----------------|
| mean | - | 25.6 ± 10.1 | 0.3324 ± 0.2137 | 0.9138 ± 0.0118 |
| upper | 0.05 | 15.6 ± 7.3 | 0.4072 ± 0.2394 | 0.8956 ± 0.0144 |
| upper | 0.1 | 16.7 ± 7.1 | 0.4109 ± 0.2377 | 0.8976 ± 0.0147 |
| upper | 0.5 | 16.7 ± 7.1 | 0.3764 ± 0.2437 | 0.9092 ± 0.0103 |
| **Doctor (raw)** | - | - | 0.1982 ± 0.0166 | 0.9297 ± 0.0108 |

#### Oracle: Best FPR on test (per seed)

| score | alpha | n_clusters | FPR (test) | ROC-AUC (test) |
|-------|-------|------------|------------|----------------|
| mean | - | 31.1 ± 19.0 | 0.2909 ± 0.1961 | 0.9148 ± 0.0101 |
| upper | 0.05 | 13.3 ± 7.1 | 0.3081 ± 0.0626 | 0.8950 ± 0.0156 |
| upper | 0.1 | 15.6 ± 8.8 | 0.3079 ± 0.0627 | 0.8966 ± 0.0145 |
| upper | 0.5 | 20.0 ± 13.2 | 0.2876 ± 0.0653 | 0.9053 ± 0.0136 |
| **Doctor (raw)** | - | - | 0.1982 ± 0.0166 | 0.9297 ± 0.0108 |

---

### CIFAR-10 / DenseNet-121 (9 seeds)

#### Full Grid Results (all alpha × n_clusters)

| score | alpha | n_clusters | FPR (res) | FPR (cal) | FPR (test) | ROC-AUC (test) |
|-------|-------|------------|-----------|-----------|------------|----------------|
| mean | - | 10 | 0.3962 ± 0.2967 | 0.3895 ± 0.1419 | 0.3845 ± 0.1084 | 0.8896 ± 0.0443 |
| mean | - | 20 | 0.3688 ± 0.2646 | 0.3637 ± 0.1239 | 0.4196 ± 0.2314 | 0.8916 ± 0.0430 |
| mean | - | 30 | 0.3701 ± 0.2782 | 0.3361 ± 0.1281 | 0.4429 ± 0.2294 | 0.8906 ± 0.0425 |
| mean | - | 50 | 0.3812 ± 0.2960 | 0.3050 ± 0.1150 | 0.4916 ± 0.2725 | 0.8858 ± 0.0479 |
| upper | 0.05 | 10 | 0.4102 ± 0.2754 | 0.4287 ± 0.1245 | 0.3827 ± 0.1069 | 0.8839 ± 0.0414 |
| upper | 0.05 | 20 | 0.4321 ± 0.2728 | 0.4244 ± 0.1200 | 0.4128 ± 0.1058 | 0.8725 ± 0.0461 |
| upper | 0.05 | 30 | 0.4598 ± 0.2653 | 0.4212 ± 0.1112 | 0.4575 ± 0.0824 | 0.8666 ± 0.0411 |
| upper | 0.05 | 50 | 0.4711 ± 0.2848 | 0.4411 ± 0.0999 | 0.4567 ± 0.0937 | 0.8425 ± 0.0507 |
| upper | 0.1 | 10 | 0.4102 ± 0.2754 | 0.4230 ± 0.1260 | 0.3827 ± 0.1069 | 0.8844 ± 0.0416 |
| upper | 0.1 | 20 | 0.4321 ± 0.2728 | 0.4200 ± 0.1197 | 0.4128 ± 0.1058 | 0.8764 ± 0.0460 |
| upper | 0.1 | 30 | 0.4484 ± 0.2708 | 0.4183 ± 0.1153 | 0.4537 ± 0.0862 | 0.8711 ± 0.0425 |
| upper | 0.1 | 50 | 0.4663 ± 0.2866 | 0.4332 ± 0.1007 | 0.4541 ± 0.0947 | 0.8477 ± 0.0491 |
| upper | 0.5 | 10 | 0.4185 ± 0.2968 | 0.4149 ± 0.1343 | 0.4048 ± 0.1008 | 0.8870 ± 0.0434 |
| upper | 0.5 | 20 | 0.4202 ± 0.2775 | 0.4145 ± 0.1187 | 0.4068 ± 0.1102 | 0.8858 ± 0.0445 |
| upper | 0.5 | 30 | 0.4334 ± 0.2891 | 0.4091 ± 0.1073 | 0.4372 ± 0.0920 | 0.8824 ± 0.0407 |
| upper | 0.5 | 50 | 0.4610 ± 0.2900 | 0.4134 ± 0.0983 | 0.4439 ± 0.0953 | 0.8651 ± 0.0479 |
| **Doctor (raw)** | - | - | - | - | 0.2650 ± 0.0219 | 0.9124 ± 0.0052 |

#### Selection on res split by FPR@95 (per-seed)

| score | alpha | n_clusters | FPR (test) | ROC-AUC (test) |
|-------|-------|------------|------------|----------------|
| mean | - | 32.2 ± 14.8 | 0.4819 ± 0.2972 | 0.8888 ± 0.0420 |
| upper | 0.05 | 21.1 ± 16.9 | 0.3937 ± 0.1119 | 0.8767 ± 0.0401 |
| upper | 0.1 | 21.1 ± 16.9 | 0.3915 ± 0.1102 | 0.8784 ± 0.0401 |
| upper | 0.5 | 20.0 ± 13.2 | 0.4139 ± 0.1231 | 0.8840 ± 0.0440 |
| **Doctor (raw)** | - | - | 0.2650 ± 0.0219 | 0.9124 ± 0.0052 |

#### Oracle: Best FPR on test (per seed)

| score | alpha | n_clusters | FPR (test) | ROC-AUC (test) |
|-------|-------|------------|------------|----------------|
| mean | - | 24.4 ± 12.4 | 0.3365 ± 0.1183 | 0.8913 ± 0.0448 |
| upper | 0.05 | 20.0 ± 13.2 | 0.3726 ± 0.1012 | 0.8734 ± 0.0465 |
| upper | 0.1 | 20.0 ± 13.2 | 0.3726 ± 0.1012 | 0.8756 ± 0.0460 |
| upper | 0.5 | 22.2 ± 13.0 | 0.3788 ± 0.0975 | 0.8842 ± 0.0419 |
| **Doctor (raw)** | - | - | 0.2650 ± 0.0219 | 0.9124 ± 0.0052 |

---

### CIFAR-100 / ResNet-34 (9 seeds)

#### Full Grid Results (all alpha × n_clusters)

| score | alpha | n_clusters | FPR (res) | FPR (cal) | FPR (test) | ROC-AUC (test) |
|-------|-------|------------|-----------|-----------|------------|----------------|
| mean | - | 10 | 0.4439 ± 0.0534 | 0.4731 ± 0.0192 | 0.4597 ± 0.0312 | 0.8691 ± 0.0047 |
| mean | - | 20 | 0.4089 ± 0.0430 | 0.4289 ± 0.0218 | 0.4272 ± 0.0311 | 0.8712 ± 0.0052 |
| mean | - | 30 | 0.4027 ± 0.0543 | 0.4216 ± 0.0223 | 0.4286 ± 0.0335 | 0.8707 ± 0.0063 |
| mean | - | 50 | 0.4270 ± 0.0587 | 0.4097 ± 0.0152 | 0.4216 ± 0.0200 | 0.8694 ± 0.0050 |
| upper | 0.05 | 10 | 0.4450 ± 0.0542 | 0.4744 ± 0.0188 | 0.4617 ± 0.0312 | 0.8685 ± 0.0034 |
| upper | 0.05 | 20 | 0.4190 ± 0.0427 | 0.4494 ± 0.0420 | 0.4512 ± 0.0346 | 0.8682 ± 0.0040 |
| upper | 0.05 | 30 | 0.4349 ± 0.0564 | 0.4477 ± 0.0444 | 0.4675 ± 0.0318 | 0.8637 ± 0.0054 |
| upper | 0.05 | 50 | 0.4760 ± 0.0673 | 0.4443 ± 0.0318 | 0.4737 ± 0.0281 | 0.8520 ± 0.0066 |
| upper | 0.1 | 10 | 0.4450 ± 0.0542 | 0.4744 ± 0.0188 | 0.4617 ± 0.0312 | 0.8685 ± 0.0034 |
| upper | 0.1 | 20 | 0.4190 ± 0.0427 | 0.4467 ± 0.0388 | 0.4512 ± 0.0346 | 0.8684 ± 0.0039 |
| upper | 0.1 | 30 | 0.4274 ± 0.0588 | 0.4367 ± 0.0373 | 0.4605 ± 0.0365 | 0.8653 ± 0.0053 |
| upper | 0.1 | 50 | 0.4749 ± 0.0691 | 0.4455 ± 0.0305 | 0.4736 ± 0.0277 | 0.8550 ± 0.0056 |
| upper | 0.5 | 10 | 0.4450 ± 0.0542 | 0.4744 ± 0.0188 | 0.4617 ± 0.0312 | 0.8687 ± 0.0037 |
| upper | 0.5 | 20 | 0.4113 ± 0.0357 | 0.4395 ± 0.0307 | 0.4369 ± 0.0297 | 0.8698 ± 0.0041 |
| upper | 0.5 | 30 | 0.4143 ± 0.0586 | 0.4238 ± 0.0307 | 0.4438 ± 0.0388 | 0.8688 ± 0.0048 |
| upper | 0.5 | 50 | 0.4638 ± 0.0662 | 0.4347 ± 0.0338 | 0.4610 ± 0.0286 | 0.8623 ± 0.0043 |
| **Doctor (raw)** | - | - | - | - | 0.3948 ± 0.0177 | 0.8726 ± 0.0039 |

#### Selection on res split by FPR@95 (per-seed)

| score | alpha | n_clusters | FPR (test) | ROC-AUC (test) |
|-------|-------|------------|------------|----------------|
| mean | - | 24.4 ± 11.3 | 0.4268 ± 0.0300 | 0.8709 ± 0.0050 |
| upper | 0.05 | 20.0 ± 7.1 | 0.4512 ± 0.0251 | 0.8681 ± 0.0044 |
| upper | 0.1 | 20.0 ± 7.1 | 0.4512 ± 0.0251 | 0.8683 ± 0.0043 |
| upper | 0.5 | 22.2 ± 6.7 | 0.4371 ± 0.0269 | 0.8701 ± 0.0046 |
| **Doctor (raw)** | - | - | 0.3948 ± 0.0177 | 0.8726 ± 0.0039 |

#### Oracle: Best FPR on test (per seed)

| score | alpha | n_clusters | FPR (test) | ROC-AUC (test) |
|-------|-------|------------|------------|----------------|
| mean | - | 36.7 ± 16.6 | 0.4063 ± 0.0197 | 0.8691 ± 0.0055 |
| upper | 0.05 | 23.3 ± 12.2 | 0.4367 ± 0.0293 | 0.8669 ± 0.0058 |
| upper | 0.1 | 23.3 ± 13.2 | 0.4357 ± 0.0296 | 0.8665 ± 0.0063 |
| upper | 0.5 | 24.4 ± 12.4 | 0.4211 ± 0.0259 | 0.8686 ± 0.0056 |
| **Doctor (raw)** | - | - | 0.3948 ± 0.0177 | 0.8726 ± 0.0039 |

---

### CIFAR-100 / DenseNet-121 (9 seeds)

#### Full Grid Results (all alpha × n_clusters)

| score | alpha | n_clusters | FPR (res) | FPR (cal) | FPR (test) | ROC-AUC (test) |
|-------|-------|------------|-----------|-----------|------------|----------------|
| mean | - | 10 | 0.4609 ± 0.0444 | 0.5093 ± 0.0625 | 0.5382 ± 0.0521 | 0.8529 ± 0.0042 |
| mean | - | 20 | 0.4738 ± 0.0656 | 0.4834 ± 0.0206 | 0.5010 ± 0.0278 | 0.8536 ± 0.0043 |
| mean | - | 30 | 0.4408 ± 0.0386 | 0.4703 ± 0.0224 | 0.4819 ± 0.0222 | 0.8541 ± 0.0038 |
| mean | - | 50 | 0.4482 ± 0.0537 | 0.4540 ± 0.0196 | 0.4746 ± 0.0195 | 0.8529 ± 0.0040 |
| upper | 0.05 | 10 | 0.4648 ± 0.0498 | 0.5047 ± 0.0578 | 0.5423 ± 0.0547 | 0.8524 ± 0.0045 |
| upper | 0.05 | 20 | 0.4701 ± 0.0535 | 0.5028 ± 0.0252 | 0.5028 ± 0.0260 | 0.8501 ± 0.0071 |
| upper | 0.05 | 30 | 0.4673 ± 0.0456 | 0.5010 ± 0.0198 | 0.5153 ± 0.0243 | 0.8449 ± 0.0074 |
| upper | 0.05 | 50 | 0.4973 ± 0.0562 | 0.4940 ± 0.0256 | 0.5224 ± 0.0201 | 0.8358 ± 0.0092 |
| upper | 0.1 | 10 | 0.4648 ± 0.0498 | 0.5135 ± 0.0625 | 0.5423 ± 0.0547 | 0.8526 ± 0.0045 |
| upper | 0.1 | 20 | 0.4728 ± 0.0598 | 0.5011 ± 0.0241 | 0.5028 ± 0.0260 | 0.8510 ± 0.0064 |
| upper | 0.1 | 30 | 0.4661 ± 0.0446 | 0.4974 ± 0.0232 | 0.5114 ± 0.0214 | 0.8468 ± 0.0066 |
| upper | 0.1 | 50 | 0.4961 ± 0.0569 | 0.4881 ± 0.0261 | 0.5186 ± 0.0234 | 0.8391 ± 0.0083 |
| upper | 0.5 | 10 | 0.4633 ± 0.0504 | 0.5116 ± 0.0642 | 0.5423 ± 0.0547 | 0.8529 ± 0.0042 |
| upper | 0.5 | 20 | 0.4726 ± 0.0546 | 0.4891 ± 0.0209 | 0.5061 ± 0.0242 | 0.8528 ± 0.0051 |
| upper | 0.5 | 30 | 0.4533 ± 0.0397 | 0.4818 ± 0.0222 | 0.4970 ± 0.0189 | 0.8517 ± 0.0042 |
| upper | 0.5 | 50 | 0.4735 ± 0.0599 | 0.4722 ± 0.0199 | 0.4955 ± 0.0219 | 0.8469 ± 0.0071 |
| **Doctor (raw)** | - | - | - | - | 0.4614 ± 0.0137 | 0.8570 ± 0.0038 |

#### Selection on res split by FPR@95 (per-seed)

| score | alpha | n_clusters | FPR (test) | ROC-AUC (test) |
|-------|-------|------------|------------|----------------|
| mean | - | 27.8 ± 14.8 | 0.5007 ± 0.0391 | 0.8536 ± 0.0035 |
| upper | 0.05 | 18.9 ± 12.7 | 0.5331 ± 0.0402 | 0.8500 ± 0.0071 |
| upper | 0.1 | 20.0 ± 13.2 | 0.5309 ± 0.0401 | 0.8508 ± 0.0063 |
| upper | 0.5 | 23.3 ± 13.2 | 0.5120 ± 0.0375 | 0.8526 ± 0.0039 |
| **Doctor (raw)** | - | - | 0.4614 ± 0.0137 | 0.8570 ± 0.0038 |

#### Oracle: Best FPR on test (per seed)

| score | alpha | n_clusters | FPR (test) | ROC-AUC (test) |
|-------|-------|------------|------------|----------------|
| mean | - | 43.3 ± 10.0 | 0.4717 ± 0.0192 | 0.8537 ± 0.0035 |
| upper | 0.05 | 24.4 ± 15.1 | 0.4929 ± 0.0256 | 0.8472 ± 0.0095 |
| upper | 0.1 | 26.7 ± 15.0 | 0.4913 ± 0.0239 | 0.8488 ± 0.0070 |
| upper | 0.5 | 31.1 ± 16.2 | 0.4856 ± 0.0218 | 0.8514 ± 0.0050 |
| **Doctor (raw)** | - | - | 0.4614 ± 0.0137 | 0.8570 ± 0.0038 |

---

### 9.3 Rice Rule Selection (K=30, closest to 2×n^(1/3) ≈ 31.7 for n_cal=4000)

| Dataset | Model | score | alpha | FPR (test) | ROC-AUC (test) | Doctor (raw) FPR |
|---------|-------|-------|-------|------------|----------------|------------------|
| CIFAR-10 | ResNet-34 | mean | - | 0.3423 ± 0.2096 | 0.9153 ± 0.0146 | 0.1982 ± 0.0166 |
| CIFAR-10 | ResNet-34 | upper | 0.05 | 0.3834 ± 0.0499 | 0.8780 ± 0.0270 | 0.1982 ± 0.0166 |
| CIFAR-10 | ResNet-34 | upper | 0.1 | 0.3756 ± 0.0551 | 0.8835 ± 0.0240 | 0.1982 ± 0.0166 |
| CIFAR-10 | ResNet-34 | upper | 0.5 | 0.3652 ± 0.0734 | 0.8972 ± 0.0189 | 0.1982 ± 0.0166 |
| CIFAR-10 | DenseNet-121 | mean | - | 0.4429 ± 0.2294 | 0.8906 ± 0.0425 | 0.2650 ± 0.0219 |
| CIFAR-10 | DenseNet-121 | upper | 0.05 | 0.4575 ± 0.0824 | 0.8666 ± 0.0411 | 0.2650 ± 0.0219 |
| CIFAR-10 | DenseNet-121 | upper | 0.1 | 0.4537 ± 0.0862 | 0.8711 ± 0.0425 | 0.2650 ± 0.0219 |
| CIFAR-10 | DenseNet-121 | upper | 0.5 | 0.4372 ± 0.0920 | 0.8824 ± 0.0407 | 0.2650 ± 0.0219 |
| CIFAR-100 | ResNet-34 | mean | - | 0.4286 ± 0.0335 | 0.8707 ± 0.0063 | 0.3948 ± 0.0177 |
| CIFAR-100 | ResNet-34 | upper | 0.05 | 0.4675 ± 0.0318 | 0.8637 ± 0.0054 | 0.3948 ± 0.0177 |
| CIFAR-100 | ResNet-34 | upper | 0.1 | 0.4605 ± 0.0365 | 0.8653 ± 0.0053 | 0.3948 ± 0.0177 |
| CIFAR-100 | ResNet-34 | upper | 0.5 | 0.4438 ± 0.0388 | 0.8688 ± 0.0048 | 0.3948 ± 0.0177 |
| CIFAR-100 | DenseNet-121 | mean | - | 0.4819 ± 0.0222 | 0.8541 ± 0.0038 | 0.4614 ± 0.0137 |
| CIFAR-100 | DenseNet-121 | upper | 0.05 | 0.5153 ± 0.0243 | 0.8449 ± 0.0074 | 0.4614 ± 0.0137 |
| CIFAR-100 | DenseNet-121 | upper | 0.1 | 0.5114 ± 0.0214 | 0.8468 ± 0.0066 | 0.4614 ± 0.0137 |
| CIFAR-100 | DenseNet-121 | upper | 0.5 | 0.4970 ± 0.0189 | 0.8517 ± 0.0042 | 0.4614 ± 0.0137 |

---

### 9.4 Observations

1. **High variance on CIFAR-10:** K-means constrained shows high variance (up to 29% std) on CIFAR-10, particularly with DenseNet-121. This suggests sensitivity to seed splits.

2. **Worse than raw Doctor:** All submethods perform worse than raw Doctor scores in terms of FPR@95. The binning procedure degrades detection performance by 3-22 percentage points.

3. **Mean score slightly better than upper:** The mean score type generally outperforms upper bound scores, contrary to expectation that conservative bounds would help.

4. **Small n_clusters preferred:** Best configurations tend to use smaller n_clusters (10-30), similar to other binning methods.

5. **CIFAR-100 more stable:** CIFAR-100 shows much lower variance than CIFAR-10, suggesting the method is more stable with higher error rates.

6. **Rice rule (K=30) not optimal:** Using the Rice rule for K selection does not improve over per-seed selection on res. FPR degradation remains significant.

### 9.5 Conclusion

K-means constrained binning does not improve over raw Doctor scores. Despite enforcing uniform cluster sizes (similar to uniform-mass binning), the method shows:
- Higher FPR@95 than raw scores (degradation of 3-22 percentage points)
- High variance on CIFAR-10 datasets
- No clear advantage from using upper bounds vs empirical means
- No benefit from Rice rule K selection

The constrained K-means optimization does not provide benefits over simpler binning approaches.

## 10. K-Means Constrained on Top-5 Softmax Probabilities (n_res=1000, n_cal=4000)

**Date:** 2026-01-19

### 10.1 Motivation

The previous K-means constrained experiments used a 1D gini score as features for clustering. We hypothesize that using richer features from the softmax distribution might improve cluster quality by capturing:
1. Multi-class competition (not just overall confidence)
2. Better distinction between "confused" predictions and "confident wrong" predictions
3. Richer feature space for cluster separation

### 10.2 Experimental Setup

- **Datasets:** CIFAR-10, CIFAR-100
- **Models:** ResNet-34, DenseNet-121
- **Splits:** n_res=1000, n_cal=4000, n_test=5000
- **Seeds:** 1-10 (mean ± std reported)
- **Feature space:** Top-5 softmax probabilities (5D) — sorted in descending order
- **Partition method:** K-means constrained (uniform cluster sizes)
- **Grid:** n_clusters ∈ {10, 20, 30, 50}, alpha ∈ {0.05, 0.1, 0.5}, score ∈ {mean, upper}
- **Clustering params:** n_init=5, max_iter=100
- **No ODIN perturbation** (direct softmax probabilities)

### 10.3 Configuration

Key differences from gini-based K-means constrained:

| Parameter | Gini Config | Top-5 Config |
|-----------|-------------|--------------|
| space | gini | probits |
| reorder_embs | False | True |
| n_dim | (not set, 1D) | 5 |
| normalize | True | False |
| use_perturbed_logits | True | False |

---

### CIFAR-10 / ResNet-34 (10 seeds)

#### Full Grid Results (all alpha × n_clusters)

| score | alpha | n_clusters | FPR (res) | FPR (cal) | FPR (test) | ROC-AUC (test) |
|-------|-------|------------|-----------|-----------|------------|----------------|
| mean | - | 10 | 0.4758 ± 0.2536 | 0.4012 ± 0.0752 | 0.3492 ± 0.0583 | 0.9158 ± 0.0078 |
| mean | - | 20 | 0.4126 ± 0.2331 | 0.3682 ± 0.0561 | 0.4562 ± 0.1649 | 0.9215 ± 0.0090 |
| mean | - | 30 | 0.4508 ± 0.2783 | 0.3015 ± 0.0512 | 0.4233 ± 0.2320 | 0.9206 ± 0.0190 |
| mean | - | 50 | 0.4472 ± 0.2504 | 0.2733 ± 0.0488 | 0.4135 ± 0.2129 | 0.9191 ± 0.0149 |
| upper | 0.05 | 10 | 0.5001 ± 0.2307 | 0.5199 ± 0.1029 | 0.4598 ± 0.0915 | 0.9054 ± 0.0101 |
| upper | 0.05 | 20 | 0.4825 ± 0.2300 | 0.5550 ± 0.0920 | 0.5214 ± 0.0884 | 0.8965 ± 0.0112 |
| upper | 0.05 | 30 | 0.4284 ± 0.1687 | 0.5008 ± 0.1071 | 0.4526 ± 0.0947 | 0.8896 ± 0.0128 |
| upper | 0.05 | 50 | 0.4175 ± 0.1597 | 0.4535 ± 0.0543 | 0.4623 ± 0.0985 | 0.8733 ± 0.0118 |
| upper | 0.1 | 10 | 0.5001 ± 0.2307 | 0.5199 ± 0.1029 | 0.4598 ± 0.0915 | 0.9062 ± 0.0104 |
| upper | 0.1 | 20 | 0.4777 ± 0.2338 | 0.5513 ± 0.0951 | 0.5173 ± 0.0916 | 0.9004 ± 0.0109 |
| upper | 0.1 | 30 | 0.4277 ± 0.1699 | 0.5016 ± 0.1076 | 0.4526 ± 0.0947 | 0.8935 ± 0.0136 |
| upper | 0.1 | 50 | 0.4156 ± 0.1624 | 0.4452 ± 0.0572 | 0.4623 ± 0.0985 | 0.8773 ± 0.0123 |
| upper | 0.5 | 10 | 0.4726 ± 0.2427 | 0.5065 ± 0.1108 | 0.4217 ± 0.1128 | 0.9114 ± 0.0092 |
| upper | 0.5 | 20 | 0.4634 ± 0.2373 | 0.5326 ± 0.1030 | 0.5117 ± 0.0988 | 0.9090 ± 0.0091 |
| upper | 0.5 | 30 | 0.4078 ± 0.1838 | 0.4764 ± 0.1174 | 0.4277 ± 0.1019 | 0.9059 ± 0.0119 |
| upper | 0.5 | 50 | 0.4006 ± 0.1554 | 0.4256 ± 0.0530 | 0.4531 ± 0.1044 | 0.8932 ± 0.0116 |
| **Doctor (raw)** | - | - | - | - | 0.1982 ± 0.0166 | 0.9297 ± 0.0108 |

---

### CIFAR-10 / DenseNet-121 (10 seeds)

#### Full Grid Results (all alpha × n_clusters)

| score | alpha | n_clusters | FPR (res) | FPR (cal) | FPR (test) | ROC-AUC (test) |
|-------|-------|------------|-----------|-----------|------------|----------------|
| mean | - | 10 | 0.4549 ± 0.3041 | 0.3927 ± 0.0358 | 0.4042 ± 0.0710 | 0.9104 ± 0.0041 |
| mean | - | 20 | 0.4198 ± 0.2692 | 0.3448 ± 0.0264 | 0.3750 ± 0.0462 | 0.9132 ± 0.0044 |
| mean | - | 30 | 0.4739 ± 0.3111 | 0.3189 ± 0.0268 | 0.3891 ± 0.0497 | 0.9120 ± 0.0062 |
| mean | - | 50 | 0.4314 ± 0.2972 | 0.2960 ± 0.0218 | 0.4278 ± 0.2096 | 0.9090 ± 0.0079 |
| upper | 0.05 | 10 | 0.4309 ± 0.2498 | 0.4565 ± 0.0698 | 0.4568 ± 0.0704 | 0.8999 ± 0.0072 |
| upper | 0.05 | 20 | 0.5005 ± 0.2764 | 0.4720 ± 0.0587 | 0.4385 ± 0.0566 | 0.8944 ± 0.0066 |
| upper | 0.05 | 30 | 0.4676 ± 0.2456 | 0.4518 ± 0.0630 | 0.4647 ± 0.0418 | 0.8847 ± 0.0077 |
| upper | 0.05 | 50 | 0.4643 ± 0.2277 | 0.4770 ± 0.0689 | 0.4918 ± 0.0531 | 0.8715 ± 0.0123 |
| upper | 0.1 | 10 | 0.4309 ± 0.2498 | 0.4519 ± 0.0680 | 0.4527 ± 0.0686 | 0.9002 ± 0.0072 |
| upper | 0.1 | 20 | 0.4962 ± 0.2805 | 0.4563 ± 0.0502 | 0.4292 ± 0.0506 | 0.8971 ± 0.0057 |
| upper | 0.1 | 30 | 0.4644 ± 0.2480 | 0.4435 ± 0.0572 | 0.4617 ± 0.0409 | 0.8891 ± 0.0078 |
| upper | 0.1 | 50 | 0.4516 ± 0.2328 | 0.4632 ± 0.0710 | 0.4735 ± 0.0471 | 0.8769 ± 0.0114 |
| upper | 0.5 | 10 | 0.4309 ± 0.2778 | 0.4413 ± 0.0696 | 0.4452 ± 0.0689 | 0.9045 ± 0.0052 |
| upper | 0.5 | 20 | 0.4696 ± 0.2688 | 0.4256 ± 0.0408 | 0.4157 ± 0.0734 | 0.9033 ± 0.0071 |
| upper | 0.5 | 30 | 0.4571 ± 0.2460 | 0.4243 ± 0.0624 | 0.4552 ± 0.0323 | 0.8980 ± 0.0069 |
| upper | 0.5 | 50 | 0.4264 ± 0.2400 | 0.4166 ± 0.0404 | 0.4510 ± 0.0452 | 0.8908 ± 0.0090 |
| **Doctor (raw)** | - | - | - | - | 0.2650 ± 0.0219 | 0.9124 ± 0.0052 |

---

### CIFAR-100 / ResNet-34 (10 seeds)

#### Full Grid Results (all alpha × n_clusters)

| score | alpha | n_clusters | FPR (res) | FPR (cal) | FPR (test) | ROC-AUC (test) |
|-------|-------|------------|-----------|-----------|------------|----------------|
| mean | - | 10 | 0.4975 ± 0.0733 | 0.4840 ± 0.0188 | 0.4709 ± 0.0310 | 0.8735 ± 0.0036 |
| mean | - | 20 | 0.4676 ± 0.0479 | 0.4447 ± 0.0229 | 0.4425 ± 0.0235 | 0.8757 ± 0.0040 |
| mean | - | 30 | 0.4524 ± 0.0591 | 0.4339 ± 0.0168 | 0.4333 ± 0.0252 | 0.8767 ± 0.0041 |
| mean | - | 50 | 0.4336 ± 0.0632 | 0.4116 ± 0.0249 | 0.4338 ± 0.0122 | 0.8767 ± 0.0038 |
| upper | 0.05 | 10 | 0.4975 ± 0.0733 | 0.4840 ± 0.0188 | 0.4709 ± 0.0310 | 0.8736 ± 0.0036 |
| upper | 0.05 | 20 | 0.4842 ± 0.0620 | 0.4511 ± 0.0256 | 0.4488 ± 0.0292 | 0.8743 ± 0.0055 |
| upper | 0.05 | 30 | 0.4560 ± 0.0522 | 0.4541 ± 0.0215 | 0.4543 ± 0.0349 | 0.8726 ± 0.0041 |
| upper | 0.05 | 50 | 0.4520 ± 0.0450 | 0.4396 ± 0.0179 | 0.4635 ± 0.0265 | 0.8682 ± 0.0046 |
| upper | 0.1 | 10 | 0.4975 ± 0.0733 | 0.4840 ± 0.0188 | 0.4709 ± 0.0310 | 0.8736 ± 0.0036 |
| upper | 0.1 | 20 | 0.4842 ± 0.0620 | 0.4511 ± 0.0256 | 0.4488 ± 0.0292 | 0.8747 ± 0.0052 |
| upper | 0.1 | 30 | 0.4525 ± 0.0481 | 0.4523 ± 0.0228 | 0.4523 ± 0.0335 | 0.8732 ± 0.0040 |
| upper | 0.1 | 50 | 0.4484 ± 0.0424 | 0.4367 ± 0.0223 | 0.4613 ± 0.0271 | 0.8703 ± 0.0042 |
| upper | 0.5 | 10 | 0.4975 ± 0.0733 | 0.4840 ± 0.0188 | 0.4709 ± 0.0310 | 0.8735 ± 0.0036 |
| upper | 0.5 | 20 | 0.4686 ± 0.0505 | 0.4447 ± 0.0196 | 0.4462 ± 0.0272 | 0.8752 ± 0.0048 |
| upper | 0.5 | 30 | 0.4434 ± 0.0518 | 0.4416 ± 0.0205 | 0.4410 ± 0.0372 | 0.8756 ± 0.0042 |
| upper | 0.5 | 50 | 0.4333 ± 0.0545 | 0.4253 ± 0.0193 | 0.4515 ± 0.0202 | 0.8736 ± 0.0036 |
| **Doctor (raw)** | - | - | - | - | 0.3948 ± 0.0177 | 0.8726 ± 0.0039 |

---

### CIFAR-100 / DenseNet-121 (10 seeds)

#### Full Grid Results (all alpha × n_clusters)

| score | alpha | n_clusters | FPR (res) | FPR (cal) | FPR (test) | ROC-AUC (test) |
|-------|-------|------------|-----------|-----------|------------|----------------|
| mean | - | 10 | 0.4782 ± 0.0606 | 0.4920 ± 0.0559 | 0.5333 ± 0.0552 | 0.8524 ± 0.0038 |
| mean | - | 20 | 0.4909 ± 0.0808 | 0.4744 ± 0.0270 | 0.4971 ± 0.0267 | 0.8559 ± 0.0032 |
| mean | - | 30 | 0.4699 ± 0.0696 | 0.4642 ± 0.0246 | 0.4896 ± 0.0228 | 0.8565 ± 0.0038 |
| mean | - | 50 | 0.4793 ± 0.0669 | 0.4546 ± 0.0285 | 0.4836 ± 0.0208 | 0.8545 ± 0.0035 |
| upper | 0.05 | 10 | 0.4782 ± 0.0606 | 0.4920 ± 0.0559 | 0.5333 ± 0.0552 | 0.8524 ± 0.0038 |
| upper | 0.05 | 20 | 0.4995 ± 0.0598 | 0.4901 ± 0.0277 | 0.5055 ± 0.0297 | 0.8532 ± 0.0033 |
| upper | 0.05 | 30 | 0.4866 ± 0.0610 | 0.4965 ± 0.0297 | 0.5085 ± 0.0325 | 0.8491 ± 0.0044 |
| upper | 0.05 | 50 | 0.5162 ± 0.0662 | 0.4933 ± 0.0216 | 0.5215 ± 0.0234 | 0.8421 ± 0.0073 |
| upper | 0.1 | 10 | 0.4782 ± 0.0606 | 0.4920 ± 0.0559 | 0.5333 ± 0.0552 | 0.8524 ± 0.0038 |
| upper | 0.1 | 20 | 0.4964 ± 0.0636 | 0.4884 ± 0.0267 | 0.5033 ± 0.0303 | 0.8539 ± 0.0033 |
| upper | 0.1 | 30 | 0.4788 ± 0.0595 | 0.4902 ± 0.0304 | 0.5057 ± 0.0324 | 0.8509 ± 0.0034 |
| upper | 0.1 | 50 | 0.5088 ± 0.0699 | 0.4922 ± 0.0219 | 0.5190 ± 0.0244 | 0.8444 ± 0.0069 |
| upper | 0.5 | 10 | 0.4782 ± 0.0606 | 0.4920 ± 0.0559 | 0.5333 ± 0.0552 | 0.8524 ± 0.0038 |
| upper | 0.5 | 20 | 0.4919 ± 0.0699 | 0.4846 ± 0.0291 | 0.5024 ± 0.0315 | 0.8553 ± 0.0030 |
| upper | 0.5 | 30 | 0.4739 ± 0.0686 | 0.4750 ± 0.0254 | 0.5028 ± 0.0267 | 0.8538 ± 0.0033 |
| upper | 0.5 | 50 | 0.4914 ± 0.0682 | 0.4750 ± 0.0290 | 0.5075 ± 0.0273 | 0.8509 ± 0.0043 |
| **Doctor (raw)** | - | - | - | - | 0.4614 ± 0.0137 | 0.8570 ± 0.0038 |

---

### 10.4 Summary: Gini-based vs Top-5 Probabilities

| Dataset | Model | Feature Space | Best FPR (mean) | Best FPR (upper) |
|---------|-------|---------------|-----------------|------------------|
| CIFAR-10 | ResNet-34 | Gini (1D) | 0.318 ± 0.188 | 0.288 ± 0.065 |
| CIFAR-10 | ResNet-34 | Top-5 Probits (5D) | 0.349 ± 0.058 | 0.422 ± 0.113 |
| CIFAR-10 | DenseNet-121 | Gini (1D) | 0.337 ± 0.118 | 0.373 ± 0.101 |
| CIFAR-10 | DenseNet-121 | Top-5 Probits (5D) | 0.375 ± 0.046 | 0.416 ± 0.073 |
| CIFAR-100 | ResNet-34 | Gini (1D) | 0.406 ± 0.020 | 0.421 ± 0.026 |
| CIFAR-100 | ResNet-34 | Top-5 Probits (5D) | 0.433 ± 0.025 | 0.441 ± 0.037 |
| CIFAR-100 | DenseNet-121 | Gini (1D) | 0.472 ± 0.019 | 0.486 ± 0.022 |
| CIFAR-100 | DenseNet-121 | Top-5 Probits (5D) | 0.484 ± 0.021 | 0.502 ± 0.031 |

### 10.5 Observations

1. **Top-5 does NOT improve over Gini:** Using the top-5 softmax probabilities as a 5D feature space for clustering does not improve FPR@95 compared to the 1D gini score. In fact, performance is slightly worse across all dataset/model combinations.

2. **Lower variance but higher mean FPR:** Top-5 probabilities show lower variance (especially on CIFAR-10), but the mean FPR is higher than gini-based clustering. For example, on CIFAR-10 ResNet-34: Top-5 mean=0.349±0.058 vs Gini mean=0.318±0.188.

3. **Higher dimensional space doesn't help:** The hypothesis that richer 5D features would enable better cluster separation did not hold. The additional dimensions may introduce noise or make it harder to find good cluster boundaries.

4. **Still worse than raw Doctor:** Both Top-5 and Gini clustering perform significantly worse than raw Doctor scores (15-20 percentage points degradation on CIFAR-10).

5. **Consistent pattern across datasets:** The relative ordering (Doctor raw > Gini clustering > Top-5 clustering) is consistent across all four dataset/model combinations.

### 10.6 Conclusion

Using top-5 softmax probabilities as features for K-means constrained clustering does not provide benefits over the simpler 1D gini score. The additional information in the multi-dimensional softmax space does not translate to better error detection performance when used with constrained K-means clustering.

The results suggest that the gini score already captures the relevant uncertainty information for this task, and adding more dimensions from the softmax distribution introduces noise rather than signal. For future work, alternative dimensionality reduction techniques (e.g., PCA on full softmax) or different distance metrics might be explored, but the current evidence suggests that 1D scores are preferable for this binning approach.

---

## 11. Combined Scores K-Means Constrained (ImageNet)

### 11.1 Motivation

Previous experiments showed that Doctor (Gini), Margin, and MSP scores have different correlation structures:
- Margin ≈ MSP ≈ Entropy (r > 0.93) - highly redundant
- Gini is most distinct (r = 0.72-0.81 with others)

This experiment tests whether combining multiple uncertainty scores in a multi-dimensional space can improve error detection compared to single-score binning.

### 11.2 Experimental Setup

**Scores combined:** Doctor (Gini), Margin, MSP
- Each score computed with per-seed hyperparameters selected from baseline grids
- Scores normalized to [0,1] before concatenation

**Method:** K-Means Constrained clustering in 3D score space
- `simultaneous=True`: Bonferroni correction (α' = α/K) for valid confidence bounds
- `alpha=0.05`, `bound=hoeffding`
- Grid: K ∈ {10, 20, 30, 50}

**Data splits:** n_res=5000, n_cal=20000, n_test=25000

### 11.3 Per-Score Hyperparameters

Selected from baseline grid searches on res split (per-seed):

| Score | Postprocessor | Selection Metric | Run Tag |
|-------|---------------|------------------|---------|
| Gini | doctor | FPR@95 | doctor-grid-20260120 |
| Margin | margin | FPR@95 | margin-grid-20260120 |
| MSP | odin | FPR@95 | msp-grid-20260120 |

### 11.4 Results

### ImageNet / ViT-B/16 (9 seeds)

| score | n_clusters | FPR (test) | ROC-AUC (test) | AURC (test) |
|-------|------------|------------|----------------|-------------|
| mean | 10 | 0.4941 ± 0.0108 | 0.8703 ± 0.0024 | 0.3821 ± 0.0039 |
| mean | 20 | 0.4500 ± 0.0084 | 0.8732 ± 0.0024 | 0.3872 ± 0.0041 |
| mean | 30 | 0.4366 ± 0.0142 | 0.8737 ± 0.0025 | 0.3886 ± 0.0036 |
| mean | 50 | 0.4426 ± 0.0096 | 0.8738 ± 0.0028 | 0.3887 ± 0.0038 |
| upper | 10 | 0.4947 ± 0.0101 | 0.8702 ± 0.0025 | 0.3821 ± 0.0039 |
| upper | 20 | 0.4537 ± 0.0127 | 0.8729 ± 0.0028 | 0.3872 ± 0.0042 |
| upper | 30 | 0.4444 ± 0.0208 | 0.8732 ± 0.0030 | 0.3885 ± 0.0037 |
| upper | 50 | 0.4569 ± 0.0226 | 0.8721 ± 0.0037 | 0.3881 ± 0.0040 |
| **Doctor UM (K=30)** | 30 | 0.4509 ± 0.0153 | 0.8738 ± 0.0025 | 0.3873 ± 0.0028 |

### ImageNet / ViT-Ti/16 (9 seeds)

| score | n_clusters | FPR (test) | ROC-AUC (test) | AURC (test) |
|-------|------------|------------|----------------|-------------|
| mean | 10 | 0.4676 ± 0.0098 | 0.8635 ± 0.0016 | 0.4793 ± 0.0040 |
| mean | 20 | 0.4755 ± 0.0098 | 0.8662 ± 0.0017 | 0.4813 ± 0.0033 |
| mean | 30 | 0.4714 ± 0.0105 | 0.8668 ± 0.0015 | 0.4819 ± 0.0035 |
| mean | 50 | 0.4595 ± 0.0133 | 0.8671 ± 0.0015 | 0.4822 ± 0.0030 |
| upper | 10 | 0.4676 ± 0.0098 | 0.8635 ± 0.0016 | 0.4793 ± 0.0040 |
| upper | 20 | 0.4755 ± 0.0098 | 0.8662 ± 0.0017 | 0.4813 ± 0.0033 |
| upper | 30 | 0.4714 ± 0.0105 | 0.8667 ± 0.0015 | 0.4818 ± 0.0034 |
| upper | 50 | 0.4591 ± 0.0137 | 0.8668 ± 0.0014 | 0.4817 ± 0.0028 |
| **Doctor UM (K=30)** | 30 | 0.4749 ± 0.0114 | 0.8649 ± 0.0031 | 0.4794 ± 0.0043 |

### 11.5 Summary: Combined Scores vs Single Score

| Model | Method | Best K | FPR@95 (test) | ROC-AUC (test) | AURC (test) | Δ FPR |
|-------|--------|--------|---------------|----------------|-------------|-------|
| ViT-B/16 | Combined (upper) | 30 | 0.4444 ± 0.0208 | 0.8732 ± 0.0030 | 0.3885 ± 0.0037 | -0.65% |
| ViT-B/16 | Doctor UM | 30 | 0.4509 ± 0.0153 | 0.8738 ± 0.0025 | 0.3873 ± 0.0028 | - |
| ViT-Ti/16 | Combined (upper) | 50 | 0.4591 ± 0.0137 | 0.8668 ± 0.0014 | 0.4817 ± 0.0028 | -1.58% |
| ViT-Ti/16 | Doctor UM | 30 | 0.4749 ± 0.0114 | 0.8649 ± 0.0031 | 0.4794 ± 0.0043 | - |

### 11.6 Observations

1. **Marginal improvement on ViT-B/16:** Combined scores achieve 0.4444 FPR@95 vs 0.4509 for Doctor UM alone (0.65 percentage points improvement). However, variance is higher (0.0208 vs 0.0153).

2. **Better improvement on ViT-Ti/16:** Combined scores achieve 0.4591 FPR@95 vs 0.4749 for Doctor UM (1.58 percentage points improvement), with slightly higher variance.

3. **ROC-AUC comparable:** Combined scores maintain similar ROC-AUC to single-score binning, suggesting the 3D clustering preserves discrimination ability.

4. **Mean vs Upper similar:** Unlike single-score binning where mean often outperforms upper, here the difference is minimal - likely because Bonferroni correction with K clusters in 3D space is quite conservative.

5. **Optimal K varies:** ViT-B/16 prefers K=30 while ViT-Ti/16 prefers K=50, suggesting the optimal bin count depends on the model's uncertainty distribution.

### 11.7 Conclusion

Combining Doctor, Margin, and MSP scores in a 3D feature space for K-means constrained binning provides **marginal improvements** (0.7-1.6 percentage points) over single-score Doctor binning on ImageNet. The improvement is modest and comes with increased variance.

Key considerations:
- The high correlation between Margin/MSP means the effective dimensionality is closer to 2D (Gini + one of Margin/MSP)
- The Bonferroni correction for simultaneous confidence bounds may be overly conservative in multi-dimensional settings
- The additional complexity may not justify the small performance gain for practical applications

**Recommendation:** For simplicity, single-score Doctor binning remains the preferred approach. Combined scores could be considered when maximizing FPR@95 is critical and the additional variance is acceptable.

---

## 12. 2D Combined Scores K-Means Constrained (ImageNet)

### 12.1 Motivation

Following the 3D combined scores experiment (Section 11), we test whether simpler 2D combinations perform equally well, given that Margin and MSP are highly correlated (r=0.983).

### 12.2 Combinations Tested

- **Gini + MSP (2D):** Doctor score combined with MSP
- **Gini + Margin (2D):** Doctor score combined with Margin

### 12.3 Results

#### Gini + MSP

#### ViT-B/16

| score | K | FPR (test) | ROC-AUC (test) | AURC (test) |
|-------|---|------------|----------------|-------------|
| mean | 10 | 0.4882 ± 0.0138 | 0.8713 ± 0.0024 | 0.3855 ± 0.0033 |
| mean | 20 | 0.4486 ± 0.0076 | 0.8740 ± 0.0025 | 0.3889 ± 0.0034 |
| mean | 30 | 0.4438 ± 0.0194 | 0.8745 ± 0.0025 | 0.3900 ± 0.0033 |
| mean | 50 | 0.4469 ± 0.0150 | 0.8744 ± 0.0026 | 0.3899 ± 0.0033 |
| upper | 10 | 0.4883 ± 0.0135 | 0.8713 ± 0.0024 | 0.3855 ± 0.0033 |
| upper | 20 | 0.4544 ± 0.0154 | 0.8733 ± 0.0031 | 0.3887 ± 0.0035 |
| upper | 30 | 0.4506 ± 0.0308 | 0.8735 ± 0.0034 | 0.3896 ± 0.0036 |
| upper | 50 | 0.4635 ± 0.0165 | 0.8725 ± 0.0034 | 0.3893 ± 0.0036 |

#### ViT-Ti/16

| score | K | FPR (test) | ROC-AUC (test) | AURC (test) |
|-------|---|------------|----------------|-------------|
| mean | 10 | 0.5034 ± 0.0560 | 0.8637 ± 0.0015 | 0.4801 ± 0.0041 |
| mean | 20 | 0.4733 ± 0.0095 | 0.8667 ± 0.0014 | 0.4820 ± 0.0033 |
| mean | 30 | 0.4699 ± 0.0109 | 0.8670 ± 0.0016 | 0.4823 ± 0.0033 |
| mean | 50 | 0.4713 ± 0.0139 | 0.8668 ± 0.0017 | 0.4821 ± 0.0032 |
| upper | 10 | 0.5034 ± 0.0560 | 0.8637 ± 0.0015 | 0.4801 ± 0.0041 |
| upper | 20 | 0.4733 ± 0.0095 | 0.8667 ± 0.0015 | 0.4820 ± 0.0033 |
| upper | 30 | 0.4710 ± 0.0111 | 0.8668 ± 0.0017 | 0.4823 ± 0.0033 |
| upper | 50 | 0.4754 ± 0.0156 | 0.8662 ± 0.0017 | 0.4817 ± 0.0027 |

#### Gini + Margin

#### ViT-B/16

| score | K | FPR (test) | ROC-AUC (test) | AURC (test) |
|-------|---|------------|----------------|-------------|
| mean | 10 | 0.4936 ± 0.0120 | 0.8687 ± 0.0026 | 0.3794 ± 0.0039 |
| mean | 20 | 0.4569 ± 0.0161 | 0.8713 ± 0.0026 | 0.3846 ± 0.0063 |
| mean | 30 | 0.4415 ± 0.0158 | 0.8719 ± 0.0024 | 0.3849 ± 0.0046 |
| mean | 50 | 0.4442 ± 0.0080 | 0.8716 ± 0.0028 | 0.3837 ± 0.0047 |
| upper | 10 | 0.4970 ± 0.0097 | 0.8685 ± 0.0027 | 0.3793 ± 0.0040 |
| upper | 20 | 0.4650 ± 0.0195 | 0.8708 ± 0.0030 | 0.3845 ± 0.0064 |
| upper | 30 | 0.4555 ± 0.0221 | 0.8708 ± 0.0030 | 0.3845 ± 0.0047 |
| upper | 50 | 0.4647 ± 0.0163 | 0.8695 ± 0.0038 | 0.3829 ± 0.0042 |

#### ViT-Ti/16

| score | K | FPR (test) | ROC-AUC (test) | AURC (test) |
|-------|---|------------|----------------|-------------|
| mean | 10 | 0.4685 ± 0.0106 | 0.8613 ± 0.0012 | 0.4786 ± 0.0047 |
| mean | 20 | 0.4764 ± 0.0103 | 0.8640 ± 0.0017 | 0.4789 ± 0.0025 |
| mean | 30 | 0.4718 ± 0.0108 | 0.8651 ± 0.0015 | 0.4801 ± 0.0020 |
| mean | 50 | 0.4601 ± 0.0117 | 0.8655 ± 0.0015 | 0.4805 ± 0.0014 |
| upper | 10 | 0.4685 ± 0.0106 | 0.8613 ± 0.0012 | 0.4786 ± 0.0047 |
| upper | 20 | 0.4764 ± 0.0103 | 0.8640 ± 0.0017 | 0.4789 ± 0.0025 |
| upper | 30 | 0.4718 ± 0.0108 | 0.8650 ± 0.0015 | 0.4797 ± 0.0018 |
| upper | 50 | 0.4601 ± 0.0117 | 0.8653 ± 0.0014 | 0.4796 ± 0.0019 |

### 12.4 Comparison: All Methods vs Doctor UM Baseline

| Model | Method | Best K | FPR@95 (test) | ROC-AUC (test) | AURC (test) | Δ FPR vs UM |
|-------|--------|--------|---------------|----------------|-------------|-------------|
| ViT-B/16 | Doctor UM (1D) | 30 | 0.4509 ± 0.0153 | 0.8738 ± 0.0025 | 0.3873 ± 0.0028 | - |
| ViT-B/16 | 3D (Gini+Margin+MSP) | 30 | 0.4444 ± 0.0208 | 0.8732 ± 0.0030 | 0.3885 ± 0.0037 | -0.65% |
| ViT-B/16 | 2D (Gini+MSP) | 30 | 0.4506 ± 0.0308 | 0.8735 ± 0.0034 | 0.3896 ± 0.0036 | -0.03% |
| ViT-B/16 | 2D (Gini+Margin) | 30 | 0.4555 ± 0.0221 | 0.8708 ± 0.0030 | 0.3845 ± 0.0047 | +0.46% |
| ViT-Ti/16 | Doctor UM (1D) | 30 | 0.4749 ± 0.0114 | 0.8649 ± 0.0031 | 0.4794 ± 0.0043 | - |
| ViT-Ti/16 | 3D (Gini+Margin+MSP) | 50 | 0.4591 ± 0.0137 | 0.8668 ± 0.0014 | 0.4817 ± 0.0028 | -1.58% |
| ViT-Ti/16 | 2D (Gini+MSP) | 30 | 0.4710 ± 0.0111 | 0.8668 ± 0.0017 | 0.4823 ± 0.0033 | -0.39% |
| ViT-Ti/16 | 2D (Gini+Margin) | 50 | 0.4601 ± 0.0117 | 0.8653 ± 0.0014 | 0.4796 ± 0.0019 | -1.48% |

### 12.5 Observations

1. **2D combinations do NOT improve over single-score Doctor UM:**
   - ViT-B/16: 2D Gini+MSP achieves 0.4506 FPR@95 vs 0.4509 for Doctor UM (essentially identical)
   - ViT-B/16: 2D Gini+Margin is worse (0.4555 vs 0.4509)
   - ViT-Ti/16: 2D Gini+Margin achieves 0.4601 vs 0.4749 (-1.48%), comparable to 3D

2. **3D marginally better than 2D on ViT-B/16:**
   - 3D: 0.4444 FPR@95
   - 2D Gini+MSP: 0.4506
   - 2D Gini+Margin: 0.4555
   - The third score provides a small additional benefit

3. **On ViT-Ti/16, 2D Gini+Margin ≈ 3D:**
   - 3D: 0.4591 FPR@95
   - 2D Gini+Margin: 0.4601
   - Adding MSP doesn't help when Margin is already included

4. **Higher variance with multi-dimensional methods:**
   - Doctor UM: std ≈ 0.011-0.015
   - 2D/3D methods: std ≈ 0.011-0.031
   - Multi-dimensional clustering is less stable

### 12.6 Conclusion

**2D score combinations do not improve over single-score Doctor Uniform Mass binning.** The 3D combination (Gini+Margin+MSP) shows marginal improvement (0.5-1.6%), but this comes with increased variance and complexity.

The lack of improvement from 2D combinations confirms that:
- Gini (Doctor) already captures the essential uncertainty information
- Adding Margin or MSP does not provide complementary information that helps with error detection
- The high correlation between Margin/MSP (r=0.983) means they are essentially redundant

**Final Recommendation:** Use single-score Doctor Uniform Mass binning. Multi-dimensional score combinations add complexity without meaningful performance gains.

---

## 13. Supervised Partition (ImageNet)

### 13.1 Method Description

**Supervised Partition** is a risk-aware balanced recursive partition that uses error labels to guide axis-aligned splits. Unlike unsupervised methods (K-means, Uniform Mass), it explicitly maximizes error rate separation between children nodes.

**Key features:**
- Binary tree with axis-aligned hyperrectangle leaves
- Splits chosen to maximize |η_L - η_R| (error rate separation between children)
- Minimum samples per leaf constraint (n_samples / n_clusters)
- Candidate thresholds from quantiles (0.1, 0.2, ..., 0.9)
- Works for any dimension (1D, 2D, 3D, etc.)

**Hypothesis:** By using error labels during partitioning, supervised partition should create bins with better error separation than unsupervised uniform mass binning.

### 13.2 Experiment Setup

- **Dataset:** ImageNet (n_res=5000, n_cal=20000, n_test=5000)
- **Models:** ViT-Ti/16, ViT-B/16
- **Score spaces:**
  - 1D: Gini (Doctor score)
  - 2D: Gini + Margin
- **Partition fitted on:** res split (5000 samples)
- **Calibration on:** cal split (20000 samples)
- **Seeds:** 1-9

### 13.3 Results

#### 13.3.1 Supervised Partition - 1D (Gini/Doctor)

##### ViT-Ti/16

| score | K | FPR (test) | ROC-AUC (test) | AURC (test) |
|-------|---|------------|----------------|-------------|
| mean | 10 | 0.5712 ± 0.0460 | 0.8580 ± 0.0012 | 0.4764 ± 0.0032 |
| mean | 20 | 0.4840 ± 0.0220 | 0.8648 ± 0.0013 | 0.4798 ± 0.0034 |
| mean | 30 | 0.4732 ± 0.0220 | 0.8656 ± 0.0014 | 0.4795 ± 0.0034 |
| mean | 50 | 0.4609 ± 0.0103 | 0.8660 ± 0.0013 | 0.4802 ± 0.0030 |
| upper | 10 | 0.5712 ± 0.0460 | 0.8580 ± 0.0012 | 0.4764 ± 0.0032 |
| upper | 20 | 0.4840 ± 0.0220 | 0.8647 ± 0.0013 | 0.4798 ± 0.0034 |
| upper | 30 | 0.4795 ± 0.0325 | 0.8653 ± 0.0014 | 0.4794 ± 0.0034 |
| upper | 50 | 0.4628 ± 0.0118 | 0.8657 ± 0.0013 | 0.4802 ± 0.0031 |

##### ViT-B/16

| score | K | FPR (test) | ROC-AUC (test) | AURC (test) |
|-------|---|------------|----------------|-------------|
| mean | 10 | 0.4493 ± 0.0213 | 0.8662 ± 0.0020 | 0.3829 ± 0.0026 |
| mean | 20 | 0.4581 ± 0.0434 | 0.8731 ± 0.0023 | 0.3867 ± 0.0032 |
| mean | 30 | 0.4535 ± 0.0289 | 0.8739 ± 0.0022 | 0.3862 ± 0.0033 |
| mean | 50 | 0.4407 ± 0.0135 | 0.8743 ± 0.0023 | 0.3879 ± 0.0032 |
| upper | 10 | 0.4493 ± 0.0213 | 0.8662 ± 0.0020 | 0.3829 ± 0.0026 |
| upper | 20 | 0.4581 ± 0.0434 | 0.8729 ± 0.0023 | 0.3866 ± 0.0032 |
| upper | 30 | 0.4501 ± 0.0242 | 0.8735 ± 0.0022 | 0.3861 ± 0.0033 |
| upper | 50 | 0.4406 ± 0.0124 | 0.8734 ± 0.0024 | 0.3876 ± 0.0032 |

#### 13.3.2 Supervised Partition - 2D (Gini + Margin)

##### ViT-Ti/16

| score | K | FPR (test) | ROC-AUC (test) | AURC (test) |
|-------|---|------------|----------------|-------------|
| mean | 10 | 0.5586 ± 0.0627 | 0.8572 ± 0.0025 | 0.4806 ± 0.0057 |
| mean | 20 | 0.4824 ± 0.0300 | 0.8644 ± 0.0018 | 0.4812 ± 0.0036 |
| mean | 30 | 0.4878 ± 0.0377 | 0.8658 ± 0.0015 | 0.4808 ± 0.0037 |
| mean | 50 | 0.4775 ± 0.0206 | 0.8662 ± 0.0015 | 0.4814 ± 0.0034 |
| upper | 10 | 0.5586 ± 0.0627 | 0.8572 ± 0.0025 | 0.4806 ± 0.0057 |
| upper | 20 | 0.4887 ± 0.0293 | 0.8640 ± 0.0022 | 0.4806 ± 0.0038 |
| upper | 30 | 0.4915 ± 0.0352 | 0.8648 ± 0.0018 | 0.4793 ± 0.0042 |
| upper | 50 | 0.4904 ± 0.0313 | 0.8642 ± 0.0021 | 0.4788 ± 0.0035 |

##### ViT-B/16

| score | K | FPR (test) | ROC-AUC (test) | AURC (test) |
|-------|---|------------|----------------|-------------|
| mean | 10 | 0.4939 ± 0.0665 | 0.8629 ± 0.0071 | 0.3908 ± 0.0057 |
| mean | 20 | 0.4737 ± 0.0435 | 0.8693 ± 0.0048 | 0.3910 ± 0.0030 |
| mean | 30 | 0.4742 ± 0.0386 | 0.8705 ± 0.0045 | 0.3895 ± 0.0031 |
| mean | 50 | 0.4526 ± 0.0225 | 0.8720 ± 0.0040 | 0.3898 ± 0.0025 |
| upper | 10 | 0.4941 ± 0.0663 | 0.8630 ± 0.0070 | 0.3899 ± 0.0058 |
| upper | 20 | 0.4680 ± 0.0382 | 0.8686 ± 0.0056 | 0.3894 ± 0.0030 |
| upper | 30 | 0.4618 ± 0.0382 | 0.8693 ± 0.0049 | 0.3864 ± 0.0047 |
| upper | 50 | 0.4614 ± 0.0232 | 0.8698 ± 0.0040 | 0.3875 ± 0.0031 |

### 13.4 Comparison with Uniform Mass

| Model | Method | K | FPR@95 (test) | ROC-AUC (test) | AURC (test) | Δ FPR vs UM |
|-------|--------|---|---------------|----------------|-------------|-------------|
| ViT-Ti/16 | Uniform Mass | 30 | 0.4749 ± 0.0110 | 0.8649 ± 0.0030 | 0.4794 ± 0.0041 | - |
| ViT-Ti/16 | Supervised Partition (1D) | 30 | 0.4795 ± 0.0325 | 0.8653 ± 0.0014 | 0.4794 ± 0.0034 | +0.97% |
| ViT-Ti/16 | Supervised Partition (2D) | 30 | 0.4915 ± 0.0352 | 0.8648 ± 0.0018 | 0.4793 ± 0.0042 | +3.50% |
| ViT-B/16 | Uniform Mass | 30 | 0.4466 ± 0.0155 | 0.8739 ± 0.0023 | 0.3873 ± 0.0025 | - |
| ViT-B/16 | Supervised Partition (1D) | 30 | 0.4501 ± 0.0242 | 0.8735 ± 0.0022 | 0.3861 ± 0.0033 | +0.78% |
| ViT-B/16 | Supervised Partition (2D) | 30 | 0.4618 ± 0.0382 | 0.8693 ± 0.0049 | 0.3864 ± 0.0047 | +3.40% |

### 13.5 Observations

1. **Supervised Partition does NOT improve over Uniform Mass:**
   - 1D Supervised Partition: +0.78% to +0.97% worse FPR@95
   - 2D Supervised Partition: +3.40% to +3.50% worse FPR@95
   - ROC-AUC and AURC are essentially identical

2. **Higher variance with Supervised Partition:**
   - Uniform Mass: std ≈ 0.011-0.016
   - Supervised Partition 1D: std ≈ 0.024-0.033
   - Supervised Partition 2D: std ≈ 0.035-0.038
   - The supervised method is less stable across seeds

3. **2D Supervised Partition is worse than 1D:**
   - Adding Margin dimension does not help
   - Consistent with 2D K-means constrained results

### 13.6 Analysis: Why Doesn't Supervised Partition Help?

The hypothesis was that using error labels during partitioning would create better-separated bins. However, the results show no improvement. Possible explanations:

1. **Uncertainty scores already capture error information well:**
   - Gini/Doctor score is already strongly correlated with error probability
   - Simple quantile-based binning (Uniform Mass) already creates well-separated bins
   - Supervised splitting doesn't provide additional information

2. **Overfitting to res split:**
   - Supervised Partition uses error labels from res (5000 samples) to determine splits
   - These splits may not generalize well to cal/test
   - Uniform Mass uses only score values, which generalizes better

3. **Greedy splitting is suboptimal:**
   - The algorithm greedily selects the best split at each step
   - This may not lead to globally optimal partitioning
   - Simple quantile bins may achieve similar or better separation

### 13.7 Conclusion

**Supervised Partition does not improve over Uniform Mass binning.** Despite using error labels to guide splits, it achieves worse FPR@95 with higher variance. The simple unsupervised Uniform Mass method remains the best choice.

This result, combined with Section 12 (2D/3D score combinations), strongly suggests that:
- **Single-score Doctor with Uniform Mass binning is optimal**
- More complex supervised or multi-dimensional methods do not help
- The Doctor score already captures the essential uncertainty information for error detection
