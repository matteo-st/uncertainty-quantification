"""
LDA Binning Postprocessor.

Combines multiple uncertainty scores using LDA (Linear Discriminant Analysis),
then applies uniform mass binning on the combined score.

Strategy:
1. Compute multiple base scores (e.g., gini, margin, msp) on calibration data
2. Use LDA to project to 1D (supervised projection that maximizes class separation)
3. Apply uniform mass binning on the 1D combined score
4. Compute mean and upper bound error rates per bin
5. Return either mean or upper bound as the final score
"""

import torch
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from .base_postprocessor import BasePostprocessor
from .doctor_postprocessor import gini


class LDABinningPostprocessor(BasePostprocessor):
    """
    LDA-based score combination with uniform mass binning.

    Combines multiple uncertainty scores using LDA projection,
    then bins using quantiles (uniform mass) and computes error rates.
    """

    def __init__(self, model, cfg, result_folder, device=torch.device('cpu')):
        super().__init__(model, cfg, result_folder, device)

        # Score configuration
        self.temperature = float(cfg.get("temperature", 1.0))
        self.normalize = cfg.get("normalize", False)
        self.magnitude = float(cfg.get("magnitude", 0.0))

        # Which scores to combine
        self.base_scores = cfg.get("base_scores", ["gini", "margin"])

        # Binning configuration
        self.n_bins = int(cfg.get("n_bins", 10))

        # Confidence level for upper bound
        self.alpha = float(cfg.get("alpha", 0.05))

        # Which score to return: "mean" or "upper"
        self.score_type = cfg.get("score_type", "mean")

        # Bound type: "hoeffding" or "bernstein"
        self.bound = cfg.get("bound", "hoeffding")

        # LDA model
        self.lda = None

        # After fitting: store bin boundaries and error rates
        self.bin_edges = []
        self.bin_error_means = []
        self.bin_error_uppers = []
        self.bin_counts = []
        self.is_fitted = False

    def _compute_single_score(self, logits, score_name):
        """Compute a single uncertainty score from logits."""
        if score_name == "gini":
            return gini(logits, temperature=self.temperature, normalize=self.normalize).squeeze()
        elif score_name == "msp":
            probs = torch.softmax(logits / self.temperature, dim=1)
            return -probs.max(dim=1)[0]
        elif score_name == "margin":
            probs = torch.softmax(logits / self.temperature, dim=1)
            top2 = probs.topk(2, dim=1)[0]
            return -(top2[:, 0] - top2[:, 1])
        elif score_name == "entropy":
            probs = torch.softmax(logits / self.temperature, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
            return entropy
        elif score_name == "max_logit":
            return -logits.max(dim=1)[0]
        else:
            raise ValueError(f"Unknown score: {score_name}")

    def _compute_all_scores(self, logits):
        """Compute all base scores and stack them."""
        scores_list = []
        for score_name in self.base_scores:
            score = self._compute_single_score(logits, score_name)
            scores_list.append(score.detach().cpu().numpy())
        return np.column_stack(scores_list)

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

    def fit_lda(self, logits, detector_labels):
        """
        Fit LDA on the given data.

        Should be called on res split to learn the projection,
        before calling fit() on cal split.
        """
        # Compute all scores
        X = self._compute_all_scores(logits)
        y = detector_labels.detach().cpu().numpy().astype(int)

        # Fit LDA
        self.lda = LinearDiscriminantAnalysis(n_components=1)
        self.lda.fit(X, y)

    def _apply_lda(self, logits):
        """Apply LDA projection to get 1D combined score."""
        X = self._compute_all_scores(logits)
        return self.lda.transform(X).ravel()

    @torch.no_grad()
    def fit(self, logits, detector_labels, dataloader=None, **kwargs):
        """
        Fit uniform mass binning on calibration data.

        LDA should already be fitted (via fit_lda on res split).
        This method creates bins and computes error rates on cal.

        Args:
            logits: Tensor of shape [n_samples, n_classes]
            detector_labels: Tensor of shape [n_samples] with binary error labels
        """
        if self.lda is None:
            raise ValueError("LDA must be fitted first. Call fit_lda() on res split.")

        # Apply LDA to get combined score
        combined_scores = self._apply_lda(logits)
        labels = detector_labels.detach().cpu().numpy().astype(np.float64)

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

        self.is_fitted = True

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

    @torch.no_grad()
    def __call__(self, inputs=None, logits=None):
        """
        Apply LDA + binning to get calibrated error probabilities.

        Returns:
            Tensor of calibrated error probabilities [n_samples]
        """
        if logits is None:
            if inputs is None:
                raise ValueError("Either logits or inputs must be provided")
            logits = self.model(inputs)

        if not self.is_fitted or self.lda is None:
            # Return raw gini score if not fitted
            return gini(logits, temperature=self.temperature, normalize=self.normalize).squeeze()

        # Apply LDA
        combined_scores = self._apply_lda(logits)

        # Apply binning
        calibrated = np.zeros(len(combined_scores), dtype=np.float64)

        for i, score in enumerate(combined_scores):
            bin_idx = self._get_bin_index(float(score))
            if self.score_type == "mean":
                calibrated[i] = self.bin_error_means[bin_idx]
            else:  # upper
                calibrated[i] = self.bin_error_uppers[bin_idx]

        return torch.tensor(calibrated, dtype=torch.float32, device=self.device)

    def get_diagnostics(self):
        """Return diagnostic information about the fitted model."""
        if not self.is_fitted:
            return {"is_fitted": False}

        return {
            "is_fitted": True,
            "n_bins": self.n_bins,
            "alpha": self.alpha,
            "score_type": self.score_type,
            "base_scores": self.base_scores,
            "bin_edges": self.bin_edges,
            "bin_error_means": self.bin_error_means,
            "bin_error_uppers": self.bin_error_uppers,
            "bin_counts": self.bin_counts,
            "lda_coef": self.lda.coef_.tolist() if self.lda else None,
        }
