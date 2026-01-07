# 1D Score Resolution Options (for partition-based guarantees)

This note lists practical resolution functions for a 1D score `s(x)` (doctor, relu, msp) that preserve the guarantee when `r` is learned on a res split and confidence intervals are built on an independent cal split.

## Notation (shared with the binning note)
- Score: $s:\mathcal{X}\to\mathbb{R}$.
- Error indicator: $E=\mathbb{1}\{Y\\neq f(X)\}$.
- Resolution function: $r(x)\in\\{1,\\dots,K\\}$.
- Bin edges: $b_0<\\cdots<b_K$, bin width $\\Delta s_z=b_z-b_{z-1}$.
- Bin count: $N_z$ and Hoeffding half-width $h_z=\\sqrt{\\ln(2/\\alpha)/(2N_z)}$.

## Options

1) Quantile binning + minimum count merge
- Start from equal-mass (quantile) bins on the res split.
- Merge adjacent bins until each bin has at least `N_min` points.
- Goal: balanced CI widths with protection against very small bins.

2) 1D optimal segmentation (DP / 1D k-means)
- Sort by score and choose cut points to minimize within-bin variance of the error rate.
- Directly targets homogeneity of `eta(s)` rather than density.

3) Isotonic regression + binning on fitted curve
- Fit monotone `eta_hat(s)` on the res split.
- Create bins so that `eta_hat(s)` varies by at most `Delta` per bin.
- Controls within-bin bias if `eta(s)` is monotone in `s`.

4) Change-point detection (Bayesian blocks)
- Detect change points in the binary error sequence ordered by score.
- Produces piecewise-constant `eta_hat(s)` bins.

5) Monotone transform + uniform-width
- Apply a transform like logit to spread dense regions and compress tails.
- Then use uniform-width bins in transformed space.

6) K via target CI width
- Choose `N_min = log(2/alpha) / (2 * eps^2)` for a desired half-width `eps`.
- Enforce `N_z >= N_min` in every bin (by merging).

## Notes
- Any option above must learn `r` on the res split only, then compute CIs on the independent cal split to keep the finite-sample guarantee.
