"""
Isotonic Regression with Bin Splitting Postprocessor.

First applies isotonic regression to create adaptive bins, then splits large bins
to improve resolution (reduce ties) while maintaining a minimum sample count per bin.

Strategy:
1. Fit isotonic regression on calibration data -> creates adaptive bins via PAV
2. Identify bins with more than n_max samples
3. Split large bins recursively until each bin has between n_min and n_max samples
4. Recompute error rates per sub-bin (no monotonicity enforcement)
"""

import torch
import numpy as np
from sklearn.isotonic import IsotonicRegression
from collections import defaultdict

from .base_postprocessor import BasePostprocessor
from .doctor_postprocessor import gini


class IsotonicSplittingPostprocessor(BasePostprocessor):
    """
    Isotonic regression followed by bin splitting for improved resolution.

    The algorithm:
    1. Fit isotonic regression to get initial adaptive bins
    2. For bins larger than n_max, split them at the median score
    3. Continue splitting until all bins have n_min <= size <= n_max
    4. Compute error rate per final sub-bin (without monotonicity constraint)

    This addresses the tie problem in isotonic regression where many samples
    get mapped to the same probability, hurting threshold-based metrics like FPR@95.
    """

    def __init__(self, model, cfg, result_folder, device=torch.device('cpu')):
        super().__init__(model, cfg, result_folder, device)

        # Base score configuration
        self.base_score = cfg.get("base_score", "gini")
        self.temperature = cfg.get("temperature", 1.0)
        self.normalize = cfg.get("normalize", False)
        self.magnitude = cfg.get("magnitude", 0.0)

        # Splitting configuration
        self.n_min = cfg.get("n_min", 50)  # Minimum samples per bin
        self.n_max = cfg.get("n_max", 200)  # Maximum samples per bin (triggers split)

        # Initial isotonic regression
        self.isotonic = IsotonicRegression(
            increasing=True,
            out_of_bounds="clip",
        )

        # After splitting: store bin boundaries and error rates
        self.bin_edges = None  # List of (lower, upper) score boundaries
        self.bin_error_rates = None  # Error rate for each bin
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

    def _split_bins(self, scores, labels, initial_probs):
        """
        Split isotonic bins that are too large.

        Args:
            scores: numpy array of base scores
            labels: numpy array of binary error labels
            initial_probs: numpy array of isotonic probabilities

        Returns:
            bin_edges: list of (lower_score, upper_score) tuples
            bin_error_rates: list of error rates for each bin
        """
        # Identify initial isotonic bins by unique probability values
        unique_probs = np.unique(initial_probs)

        # For each unique probability, get the score range and samples
        bins_to_process = []
        for prob in unique_probs:
            mask = initial_probs == prob
            bin_scores = scores[mask]
            bin_labels = labels[mask]
            bins_to_process.append({
                'score_min': bin_scores.min(),
                'score_max': bin_scores.max(),
                'scores': bin_scores,
                'labels': bin_labels,
            })

        # Process bins: split those that are too large
        final_bins = []

        while bins_to_process:
            bin_info = bins_to_process.pop(0)
            n_samples = len(bin_info['scores'])

            if n_samples <= self.n_max:
                # Bin is small enough, keep it
                final_bins.append(bin_info)
            else:
                # Split at median score
                median_score = np.median(bin_info['scores'])

                # Handle edge case where median equals min or max
                if median_score == bin_info['score_min'] or median_score == bin_info['score_max']:
                    # Can't split further, keep as is
                    final_bins.append(bin_info)
                    continue

                # Create two sub-bins
                mask_low = bin_info['scores'] <= median_score
                mask_high = bin_info['scores'] > median_score

                # Only split if both sub-bins have enough samples
                n_low = mask_low.sum()
                n_high = mask_high.sum()

                if n_low >= self.n_min and n_high >= self.n_min:
                    # Split into two bins
                    low_bin = {
                        'score_min': bin_info['scores'][mask_low].min(),
                        'score_max': bin_info['scores'][mask_low].max(),
                        'scores': bin_info['scores'][mask_low],
                        'labels': bin_info['labels'][mask_low],
                    }
                    high_bin = {
                        'score_min': bin_info['scores'][mask_high].min(),
                        'score_max': bin_info['scores'][mask_high].max(),
                        'scores': bin_info['scores'][mask_high],
                        'labels': bin_info['labels'][mask_high],
                    }
                    bins_to_process.append(low_bin)
                    bins_to_process.append(high_bin)
                else:
                    # Can't split without violating n_min, keep as is
                    final_bins.append(bin_info)

        # Sort bins by score range
        final_bins.sort(key=lambda b: b['score_min'])

        # Extract edges and compute error rates
        bin_edges = []
        bin_error_rates = []

        for i, bin_info in enumerate(final_bins):
            # Use score boundaries
            lower = bin_info['score_min']
            # For upper bound, use next bin's lower or infinity
            if i < len(final_bins) - 1:
                upper = final_bins[i + 1]['score_min']
            else:
                upper = np.inf

            bin_edges.append((lower, upper))

            # Compute error rate for this bin
            error_rate = bin_info['labels'].mean()
            bin_error_rates.append(error_rate)

        return bin_edges, bin_error_rates

    @torch.no_grad()
    def fit(self, logits, detector_labels, dataloader=None, **kwargs):
        """
        Fit isotonic regression then split large bins.

        Args:
            logits: Tensor of shape [n_samples, n_classes]
            detector_labels: Tensor of shape [n_samples] with binary error labels
            dataloader: Optional, not used
        """
        # Compute base scores
        base_scores = self._compute_base_score(logits)

        # Convert to numpy
        scores = base_scores.detach().cpu().numpy()
        labels = detector_labels.detach().cpu().numpy()

        # Step 1: Fit initial isotonic regression
        self.isotonic.fit(scores, labels)
        initial_probs = self.isotonic.predict(scores)

        # Step 2: Split large bins
        self.bin_edges, self.bin_error_rates = self._split_bins(
            scores, labels, initial_probs
        )

        self.is_fitted = True

        # Store diagnostics
        self.n_initial_bins = len(np.unique(initial_probs))
        self.n_final_bins = len(self.bin_edges)
        self.train_score_min = scores.min()
        self.train_score_max = scores.max()

    def _get_bin_index(self, score):
        """Find which bin a score belongs to."""
        for i, (lower, upper) in enumerate(self.bin_edges):
            if lower <= score < upper:
                return i
        # Handle edge case: score equals max
        return len(self.bin_edges) - 1

    @torch.no_grad()
    def __call__(self, inputs=None, logits=None):
        """
        Apply isotonic + splitting to get calibrated error probabilities.

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

        if not self.is_fitted:
            return base_scores

        # Apply bin-based prediction
        scores = base_scores.detach().cpu().numpy()
        calibrated = np.zeros_like(scores)

        for i, score in enumerate(scores):
            bin_idx = self._get_bin_index(score)
            calibrated[i] = self.bin_error_rates[bin_idx]

        return torch.tensor(calibrated, dtype=torch.float32, device=self.device)

    def get_diagnostics(self):
        """Return diagnostic information about the fitted model."""
        if not self.is_fitted:
            return {"is_fitted": False}

        return {
            "is_fitted": True,
            "n_initial_isotonic_bins": self.n_initial_bins,
            "n_final_bins_after_splitting": self.n_final_bins,
            "n_min": self.n_min,
            "n_max": self.n_max,
            "bin_edges": self.bin_edges,
            "bin_error_rates": self.bin_error_rates,
        }
