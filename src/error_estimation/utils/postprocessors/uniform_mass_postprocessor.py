"""
Uniform Mass Binning Postprocessor.

Creates bins with equal number of samples (uniform mass) based on a base uncertainty score.
The partition and error rate counting are both done on the calibration set.

Strategy:
1. Compute base scores (e.g., gini) on calibration data
2. Create n_bins bins such that each bin has approximately the same number of samples
3. Compute error rate per bin on the same calibration data
4. For new samples, assign to bins and return the bin's error rate
"""

import torch
import numpy as np

from .base_postprocessor import BasePostprocessor
from .doctor_postprocessor import gini


class UniformMassPostprocessor(BasePostprocessor):
    """
    Uniform mass binning postprocessor.

    Creates bins with equal number of samples based on quantiles of the base score.
    This ensures each bin has approximately n_samples / n_bins samples.

    Key property: partition and counting can be done on the same set (cal),
    unlike other binning methods that may need separate sets.
    """

    def __init__(self, model, cfg, result_folder, device=torch.device('cpu')):
        super().__init__(model, cfg, result_folder, device)

        # Base score configuration
        self.base_score = cfg.get("base_score", "gini")
        self.temperature = float(cfg.get("temperature", 1.0))
        self.normalize = cfg.get("normalize", False)
        self.magnitude = float(cfg.get("magnitude", 0.0))

        # Binning configuration
        self.n_bins = int(cfg.get("n_bins", 10))

        # After fitting: store bin boundaries and error rates
        self.bin_edges = []  # List of bin edge values (n_bins + 1 edges)
        self.bin_error_rates = []  # Error rate for each bin
        self.is_fitted = False

    def _compute_base_score(self, logits):
        """Compute the base uncertainty score from logits."""
        if self.base_score == "gini":
            return gini(logits, temperature=self.temperature, normalize=self.normalize).squeeze()
        elif self.base_score == "msp":
            probs = torch.softmax(logits / self.temperature, dim=1)
            return -probs.max(dim=1)[0]
        elif self.base_score == "margin":
            probs = torch.softmax(logits / self.temperature, dim=1)
            top2 = probs.topk(2, dim=1)[0]
            return -(top2[:, 0] - top2[:, 1])
        else:
            raise ValueError(f"Unknown base_score: {self.base_score}")

    @torch.no_grad()
    def fit(self, logits, detector_labels, dataloader=None, **kwargs):
        """
        Fit uniform mass binning on calibration data.

        Creates bins with equal number of samples using quantiles,
        then computes error rate per bin.

        Args:
            logits: Tensor of shape [n_samples, n_classes]
            detector_labels: Tensor of shape [n_samples] with binary error labels
            dataloader: Optional, not used
        """
        # Compute base scores
        base_scores = self._compute_base_score(logits)

        # Convert to numpy
        scores = base_scores.detach().cpu().numpy().astype(np.float64)
        labels = detector_labels.detach().cpu().numpy().astype(np.float64)

        # Create bin edges using quantiles (uniform mass)
        # We want n_bins bins, so we need n_bins + 1 edges
        quantiles = np.linspace(0, 100, self.n_bins + 1)
        self.bin_edges = np.percentile(scores, quantiles).tolist()

        # Ensure first edge is -inf and last is +inf for proper binning
        self.bin_edges[0] = -np.inf
        self.bin_edges[-1] = np.inf

        # Compute error rate per bin
        self.bin_error_rates = []
        for i in range(self.n_bins):
            lower = self.bin_edges[i]
            upper = self.bin_edges[i + 1]

            # Find samples in this bin
            if i == self.n_bins - 1:
                # Last bin includes upper edge
                mask = (scores >= lower) & (scores <= upper)
            else:
                mask = (scores >= lower) & (scores < upper)

            bin_labels = labels[mask]

            if len(bin_labels) > 0:
                error_rate = float(bin_labels.mean())
            else:
                error_rate = 0.0

            self.bin_error_rates.append(error_rate)

        self.is_fitted = True

        # Store diagnostics
        self.n_samples_per_bin = []
        for i in range(self.n_bins):
            lower = self.bin_edges[i]
            upper = self.bin_edges[i + 1]
            if i == self.n_bins - 1:
                mask = (scores >= lower) & (scores <= upper)
            else:
                mask = (scores >= lower) & (scores < upper)
            self.n_samples_per_bin.append(int(mask.sum()))

    def _get_bin_index(self, score):
        """Find which bin a score belongs to."""
        for i in range(self.n_bins):
            lower = self.bin_edges[i]
            upper = self.bin_edges[i + 1]

            if i == self.n_bins - 1:
                # Last bin includes upper edge
                if score >= lower:
                    return i
            else:
                if lower <= score < upper:
                    return i

        # Fallback (shouldn't happen with -inf/+inf edges)
        return self.n_bins - 1

    @torch.no_grad()
    def __call__(self, inputs=None, logits=None):
        """
        Apply uniform mass binning to get calibrated error probabilities.

        Args:
            inputs: Optional input images
            logits: Optional pre-computed logits

        Returns:
            Tensor of calibrated error probabilities [n_samples]
        """
        if logits is None:
            if inputs is None:
                raise ValueError("Either logits or inputs must be provided")
            logits = self.model(inputs)

        # Compute base scores
        base_scores = self._compute_base_score(logits)

        if not self.is_fitted or len(self.bin_edges) == 0:
            return base_scores

        # Apply bin-based prediction
        scores = base_scores.detach().cpu().numpy().astype(np.float64)
        calibrated = np.zeros(len(scores), dtype=np.float64)

        for i, score in enumerate(scores):
            bin_idx = self._get_bin_index(float(score))
            calibrated[i] = float(self.bin_error_rates[bin_idx])

        return torch.tensor(calibrated, dtype=torch.float32, device=self.device)

    def get_diagnostics(self):
        """Return diagnostic information about the fitted model."""
        if not self.is_fitted:
            return {"is_fitted": False}

        return {
            "is_fitted": True,
            "n_bins": self.n_bins,
            "bin_edges": self.bin_edges,
            "bin_error_rates": self.bin_error_rates,
            "n_samples_per_bin": self.n_samples_per_bin,
        }
