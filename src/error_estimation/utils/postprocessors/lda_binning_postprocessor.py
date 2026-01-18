"""
LDA Binning Postprocessor.

Combines multiple uncertainty scores using LDA (Linear Discriminant Analysis),
then applies uniform mass binning on the combined score.

Inherits from RawLDAPostprocessor for LDA fitting and score computation.

Strategy:
1. Compute multiple base scores (e.g., gini, margin, msp) with per-score perturbation
2. Use LDA to project to 1D (supervised projection that maximizes class separation)
3. Apply uniform mass binning on the 1D combined score
4. Compute mean and upper bound error rates per bin
5. Return either mean or upper bound as the final score
"""

import torch
import numpy as np

from .raw_lda_postprocessor import RawLDAPostprocessor


class LDABinningPostprocessor(RawLDAPostprocessor):
    """
    LDA-based score combination with uniform mass binning.

    Extends RawLDAPostprocessor with binning capabilities.
    """

    def __init__(self, model, cfg, result_folder, device=torch.device('cpu')):
        super().__init__(model, cfg, result_folder, device)

        # Binning configuration
        self.n_bins = int(cfg.get("n_bins", 10))

        # Confidence level for upper bound
        self.alpha = float(cfg.get("alpha", 0.05))

        # Which score to return: "mean" or "upper"
        self.score_type = cfg.get("score_type", "mean")

        # Bound type: "hoeffding" or "bernstein"
        self.bound = cfg.get("bound", "hoeffding")

        # After fitting: store bin boundaries and error rates
        self.bin_edges = []
        self.bin_error_means = []
        self.bin_error_uppers = []
        self.bin_counts = []
        self.binning_fitted = False

    def _compute_upper_bound(self, mean, n, variance=None):
        """Compute upper confidence bound for error rate."""
        if n == 0:
            return 1.0

        if self.bound.lower() == "hoeffding":
            # Hoeffding bound: half = sqrt(log(2/alpha) / (2n))
            log_term = np.log(2.0 / self.alpha)
            half = np.sqrt(log_term / (2.0 * n))
        elif self.bound.lower() == "bernstein":
            # Bernstein bound (requires variance)
            if variance is None:
                variance = mean * (1 - mean)  # Bernoulli variance
            log_term = np.log(3.0 / self.alpha)
            half = np.sqrt((2.0 * variance * log_term) / n) + (3.0 * log_term) / n
        else:
            raise ValueError(f"Unknown bound: {self.bound}")

        return min(1.0, mean + half)

    def fit_binning(self, combined_scores, labels):
        """
        Fit uniform mass binning on combined LDA scores.

        Args:
            combined_scores: 1D array of LDA scores
            labels: Binary labels (1 = error, 0 = correct)
        """
        labels = np.asarray(labels, dtype=np.float64)

        # Create bin edges using quantiles (uniform mass)
        quantiles = np.linspace(0, 100, self.n_bins + 1)
        self.bin_edges = np.percentile(combined_scores, quantiles).tolist()

        # Ensure first edge is -inf and last is +inf
        self.bin_edges[0] = -np.inf
        self.bin_edges[-1] = np.inf

        # Compute error rate per bin
        self.bin_error_means = []
        self.bin_error_uppers = []
        self.bin_counts = []

        for i in range(self.n_bins):
            lower = self.bin_edges[i]
            upper = self.bin_edges[i + 1]

            # Find samples in this bin
            if i == self.n_bins - 1:
                mask = (combined_scores >= lower) & (combined_scores <= upper)
            else:
                mask = (combined_scores >= lower) & (combined_scores < upper)

            bin_labels = labels[mask]
            n = len(bin_labels)
            self.bin_counts.append(n)

            if n > 0:
                mean = float(bin_labels.mean())
                variance = float(bin_labels.var()) if n > 1 else mean * (1 - mean)
                upper_bound = self._compute_upper_bound(mean, n, variance)
            else:
                mean = 0.0
                upper_bound = 1.0

            self.bin_error_means.append(mean)
            self.bin_error_uppers.append(upper_bound)

        self.binning_fitted = True

    def fit_binning_with_perturbation(self, evaluator, score_configs=None):
        """
        Fit binning using proper per-score perturbation.

        LDA must be fitted first via fit_lda_with_perturbation().

        Args:
            evaluator: AblationDetector instance for the calibration split
            score_configs: Optional dict of per-score configs
        """
        lda_scores, labels = self.evaluate_with_perturbation(evaluator, score_configs)
        self.fit_binning(lda_scores, labels)
        return lda_scores, labels

    @torch.no_grad()
    def fit(self, logits, detector_labels, dataloader=None, **kwargs):
        """
        Fit uniform mass binning on calibration data.

        LDA should already be fitted (via fit_lda or fit_lda_with_perturbation).
        This method creates bins and computes error rates.

        For proper perturbation support, use fit_binning_with_perturbation() instead.

        Args:
            logits: Tensor of shape [n_samples, n_classes]
            detector_labels: Tensor of shape [n_samples] with binary error labels
        """
        if self.lda is None:
            raise ValueError("LDA must be fitted first. Call fit_lda() or fit_lda_with_perturbation().")

        # Apply LDA to get combined score (no perturbation)
        combined_scores = self._apply_lda(logits)
        labels = detector_labels.detach().cpu().numpy() if isinstance(
            detector_labels, torch.Tensor) else detector_labels.astype(np.float64)

        self.fit_binning(combined_scores, labels)

    def _get_bin_index(self, score):
        """Find which bin a score belongs to."""
        for i in range(self.n_bins):
            lower = self.bin_edges[i]
            upper = self.bin_edges[i + 1]

            if i == self.n_bins - 1:
                if score >= lower:
                    return i
            else:
                if lower <= score < upper:
                    return i

        return self.n_bins - 1

    def _apply_binning(self, combined_scores):
        """Apply binning to get calibrated probabilities."""
        calibrated = np.zeros(len(combined_scores), dtype=np.float64)

        for i, score in enumerate(combined_scores):
            bin_idx = self._get_bin_index(float(score))
            if self.score_type == "mean":
                calibrated[i] = self.bin_error_means[bin_idx]
            else:  # upper
                calibrated[i] = self.bin_error_uppers[bin_idx]

        return calibrated

    @torch.no_grad()
    def __call__(self, inputs=None, logits=None):
        """
        Apply LDA + binning to get calibrated error probabilities.

        Note: This method does NOT apply perturbation.

        Returns:
            Tensor of calibrated error probabilities [n_samples]
        """
        if logits is None:
            if inputs is None:
                raise ValueError("Either logits or inputs must be provided")
            logits = self.model(inputs)

        if not self.binning_fitted or self.lda is None:
            # Return raw gini score if not fitted
            from .doctor_postprocessor import gini
            return gini(logits, temperature=self.default_temperature,
                        normalize=self.default_normalize).squeeze()

        # Apply LDA
        combined_scores = self._apply_lda(logits)

        # Apply binning
        calibrated = self._apply_binning(combined_scores)

        return torch.tensor(calibrated, dtype=torch.float32, device=self.device)

    def evaluate_with_binning_and_perturbation(self, evaluator, score_configs=None):
        """
        Evaluate on a split using perturbation and binning.

        Args:
            evaluator: AblationDetector instance
            score_configs: Optional per-score configs

        Returns:
            Tuple of (calibrated_scores, detector_labels)
        """
        if not self.binning_fitted:
            raise ValueError("Binning must be fitted first via fit_binning() or fit_binning_with_perturbation()")

        lda_scores, labels = self.evaluate_with_perturbation(evaluator, score_configs)
        calibrated = self._apply_binning(lda_scores)
        return calibrated, labels

    def get_diagnostics(self):
        """Return diagnostic information about the fitted model."""
        base_diag = super().get_diagnostics()
        base_diag.update({
            "n_bins": self.n_bins,
            "alpha": self.alpha,
            "score_type": self.score_type,
            "bound": self.bound,
            "binning_fitted": self.binning_fitted,
            "bin_edges": self.bin_edges,
            "bin_error_means": self.bin_error_means,
            "bin_error_uppers": self.bin_error_uppers,
            "bin_counts": self.bin_counts,
        })
        return base_diag
