# Uniform-Width vs Uniform-Mass Binning for 1D Scores

This note compares two simple partitions of a 1D score `s(x)` (e.g., doctor, relu, msp) used to build resolution-based confidence intervals.

## Setup
Let `s(x)` be a scalar score and let the resolution function `r(x)` map each sample to a bin `z in {1,...,K}`. For each bin, we estimate the mean error rate and use a Hoeffding-style CI. For a bin with `N_z` calibration points, the half-width scales like:

```
half_width(z) ~ sqrt(log(2/alpha) / (2 * N_z))
```

So, bin sizes control variance (CI width), while bin widths in score space control bias (how much `eta(s)` can vary inside a bin).

## Uniform-width bins (fixed score intervals)
**Definition.** Split the score range into `K` equal-length intervals.

**Pros**
- **Bias control in score space.** If `eta(s)` is smooth/Lipschitz in `s`, the within-bin bias is bounded by a constant times the bin width. Uniform-width directly controls the maximum bin diameter in score space.
- **Interpretability.** Each bin corresponds to a fixed score interval, which is easy to interpret and compare across runs.
- **Stable boundaries.** Bin edges are deterministic once the score range is set; no quantile estimation noise.

**Cons**
- **Unequal sample sizes.** `N_z` can be very small in low-density regions, leading to wide CIs (or even `[0,1]` when empty).
- **Worst-case variance.** The largest CI width is driven by the smallest `N_z`; uniform-width can be inefficient if the score distribution is highly skewed.
- **Tail instability.** Extreme-score bins often have few samples, so uncertainty is largest where it is most needed.

## Uniform-mass bins (quantile / equal-frequency)
**Definition.** Split so that each bin has approximately the same number of samples in the resolution (res) split.

**Pros**
- **Balanced variance.** `N_z approx n/K` makes CI widths similar across bins and minimizes the worst-case half-width for fixed `K`.
- **Data-adaptive resolution.** Bins are finer where the score density is high and coarser where data are sparse, which can improve statistical efficiency.
- **Avoids empty bins.** By construction, bins are rarely empty on the res split.

**Cons**
- **Variable bin width.** Bins can be very wide in low-density regions, increasing within-bin heterogeneity and bias if `eta(s)` changes with `s`.
- **Boundary noise.** Quantile estimates depend on the res sample; with small res, bin edges can be unstable.
- **Shift sensitivity.** If the score distribution changes, the equal-mass property no longer holds, and CI widths can become unbalanced.
- **Less interpretable.** Bin score ranges differ run-to-run, making comparisons harder.

## Practical takeaways
- If you want **tight and balanced CIs**, uniform-mass is usually better.
- If you want **score-space fidelity and interpretability**, uniform-width is safer.
- A good compromise is to choose `K` so that `N_z` stays above a minimum threshold (e.g., `N_z >= log(2/alpha)/(2*eps^2)` for a target half-width `eps`) and to **learn bin edges on the res split**, then compute CIs on the independent cal split to preserve guarantees.
