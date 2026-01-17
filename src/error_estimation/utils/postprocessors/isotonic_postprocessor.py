"""
Isotonic Regression Postprocessor.

Applies isotonic regression to transform a base uncertainty score (e.g., doctor/gini)
into calibrated error probabilities. The isotonic regression is fitted on calibration
data to learn a monotonic mapping from the base score to error probability.
"""

import torch
import numpy as np
from sklearn.isotonic import IsotonicRegression

from .base_postprocessor import BasePostprocessor
from .doctor_postprocessor import gini


class IsotonicPostprocessor(BasePostprocessor):
    """
    Isotonic regression postprocessor that transforms a base score into
    calibrated error probabilities.

    The base score can be:
    - 'gini': Doctor's gini impurity score
    - 'msp': Maximum softmax probability (negated)
    - 'margin': Margin between top-2 probabilities (negated)

    The isotonic regression learns a monotonic mapping:
        base_score -> P(error)

    Since higher base scores should indicate higher error probability,
    we fit with increasing=True.
    """

    def __init__(self, model, cfg, result_folder, device=torch.device('cpu')):
        super().__init__(model, cfg, result_folder, device)

        # Base score configuration
        self.base_score = cfg.get("base_score", "gini")
        self.temperature = cfg.get("temperature", 1.0)
        self.normalize = cfg.get("normalize", False)
        self.magnitude = cfg.get("magnitude", 0.0)

        # Isotonic regression model
        self.isotonic = IsotonicRegression(
            increasing=True,  # Higher base score -> higher error prob
            out_of_bounds="clip",
        )
        self.is_fitted = False

    def _compute_base_score(self, logits):
        """Compute the base uncertainty score from logits."""
        if self.base_score == "gini":
            return gini(logits, temperature=self.temperature, normalize=self.normalize).squeeze()
        elif self.base_score == "msp":
            # Negative max softmax probability (higher = more uncertain)
            probs = torch.softmax(logits / self.temperature, dim=1)
            return -probs.max(dim=1)[0]
        elif self.base_score == "margin":
            # Negative margin between top-2 (higher = more uncertain)
            probs = torch.softmax(logits / self.temperature, dim=1)
            top2 = probs.topk(2, dim=1)[0]
            return -(top2[:, 0] - top2[:, 1])
        else:
            raise ValueError(f"Unknown base_score: {self.base_score}")

    @torch.no_grad()
    def fit(self, logits, detector_labels, dataloader=None, **kwargs):
        """
        Fit isotonic regression on calibration data.

        Args:
            logits: Tensor of shape [n_samples, n_classes]
            detector_labels: Tensor of shape [n_samples] with binary error labels
                (1 = error, 0 = correct)
            dataloader: Optional, not used (for API compatibility)
        """
        # Compute base scores
        base_scores = self._compute_base_score(logits)

        # Convert to numpy
        X = base_scores.detach().cpu().numpy()
        y = detector_labels.detach().cpu().numpy()

        # Fit isotonic regression
        self.isotonic.fit(X, y)
        self.is_fitted = True

        # Store min/max for potential diagnostics
        self.train_score_min = X.min()
        self.train_score_max = X.max()

    @torch.no_grad()
    def __call__(self, inputs=None, logits=None):
        """
        Apply isotonic regression to get calibrated error probabilities.

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
            # If not fitted, just return base scores
            return base_scores

        # Apply isotonic transformation
        X = base_scores.detach().cpu().numpy()
        calibrated = self.isotonic.predict(X)

        return torch.tensor(calibrated, dtype=torch.float32, device=self.device)
