"""
Raw LDA Postprocessor.

Combines multiple uncertainty scores using LDA (Linear Discriminant Analysis)
WITHOUT binning. Returns the raw LDA score directly.

This postprocessor supports per-score ODIN perturbation through the
AblationDetector.get_multi_score_with_perturbation() method.

Workflow:
1. fit_lda_with_perturbation(evaluator, score_configs) - Fit LDA on a split with per-score perturbation
2. __call__(logits) - Apply LDA projection (quick evaluation, no perturbation)
3. evaluate_with_perturbation(evaluator, score_configs) - Full evaluation with perturbation
"""

import torch
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from .base_postprocessor import BasePostprocessor
from .doctor_postprocessor import gini


class RawLDAPostprocessor(BasePostprocessor):
    """
    LDA-based score combination WITHOUT binning.

    Combines multiple uncertainty scores using LDA projection to return
    a 1D combined score. Supports per-score hyperparameters via score_configs.
    """

    def __init__(self, model, cfg, result_folder, device=torch.device('cpu')):
        super().__init__(model, cfg, result_folder, device)

        # Default score configuration (used if per-score config not provided)
        self.default_temperature = float(cfg.get("temperature", 1.0))
        self.default_normalize = cfg.get("normalize", False)
        self.default_magnitude = float(cfg.get("magnitude", 0.0))

        # Per-score configurations: {score_name: {temperature, normalize, magnitude}}
        self.score_configs = cfg.get("score_configs", {})

        # Which scores to combine
        self.base_scores = cfg.get("base_scores", ["gini", "margin"])

        # LDA model
        self.lda = None
        self.is_fitted = False

    def set_score_configs(self, score_configs):
        """
        Set per-score hyperparameter configurations.

        Args:
            score_configs: dict mapping score_name to {temperature, normalize, magnitude}
                Example: {"gini": {"temperature": 1.2, "normalize": True, "magnitude": 0.002},
                          "margin": {"temperature": 0.8, "normalize": False, "magnitude": 0.0}}
        """
        self.score_configs = score_configs

    def _get_score_config(self, score_name):
        """Get configuration for a specific score, falling back to defaults."""
        if score_name in self.score_configs:
            cfg = self.score_configs[score_name]
            return {
                "temperature": float(cfg.get("temperature", self.default_temperature)),
                "normalize": cfg.get("normalize", self.default_normalize),
                "magnitude": float(cfg.get("magnitude", self.default_magnitude)),
            }
        return {
            "temperature": self.default_temperature,
            "normalize": self.default_normalize,
            "magnitude": self.default_magnitude,
        }

    def _compute_single_score(self, logits, score_name):
        """Compute a single uncertainty score from logits (no perturbation)."""
        cfg = self._get_score_config(score_name)
        temperature = cfg["temperature"]
        normalize = cfg["normalize"]

        if score_name == "gini":
            return gini(logits, temperature=temperature, normalize=normalize).squeeze()
        elif score_name == "msp":
            probs = torch.softmax(logits / temperature, dim=1)
            return -probs.max(dim=1)[0]
        elif score_name == "margin":
            probs = torch.softmax(logits / temperature, dim=1)
            top2 = probs.topk(2, dim=1)[0]
            return -(top2[:, 0] - top2[:, 1])
        elif score_name == "entropy":
            probs = torch.softmax(logits / temperature, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
            return entropy
        elif score_name == "max_logit":
            return -logits.max(dim=1)[0]
        else:
            raise ValueError(f"Unknown score: {score_name}")

    def _compute_all_scores(self, logits):
        """Compute all base scores and stack them (no perturbation)."""
        scores_list = []
        for score_name in self.base_scores:
            score = self._compute_single_score(logits, score_name)
            scores_list.append(score.detach().cpu().numpy())
        return np.column_stack(scores_list)

    def fit_lda(self, logits, detector_labels):
        """
        Fit LDA on logits without perturbation.

        Use fit_lda_with_perturbation() for proper per-score perturbation.
        """
        X = self._compute_all_scores(logits)
        y = detector_labels.detach().cpu().numpy().astype(int) if isinstance(
            detector_labels, torch.Tensor) else detector_labels.astype(int)

        self.lda = LinearDiscriminantAnalysis(n_components=1)
        self.lda.fit(X, y)
        self.is_fitted = True

    def fit_lda_with_perturbation(self, evaluator, score_configs=None):
        """
        Fit LDA using AblationDetector's perturbation support.

        This is the recommended method for fitting when scores have different
        perturbation settings (magnitude > 0).

        Args:
            evaluator: AblationDetector instance for the fitting split (e.g., res split)
            score_configs: Optional dict of per-score configs. If not provided,
                uses self.score_configs.
        """
        if score_configs is not None:
            self.score_configs = score_configs

        # Build score_configs dict for get_multi_score_with_perturbation
        configs = {}
        for score_name in self.base_scores:
            configs[score_name] = self._get_score_config(score_name)

        # Get scores with proper perturbation
        scores_dict, labels = evaluator.get_multi_score_with_perturbation(configs)

        # Stack scores in order
        X = np.column_stack([scores_dict[name] for name in self.base_scores])

        # Fit LDA
        self.lda = LinearDiscriminantAnalysis(n_components=1)
        self.lda.fit(X, labels)
        self.is_fitted = True

        return scores_dict, labels

    def _apply_lda(self, logits):
        """Apply LDA projection to get 1D combined score (no perturbation)."""
        X = self._compute_all_scores(logits)
        return self.lda.transform(X).ravel()

    def evaluate_with_perturbation(self, evaluator, score_configs=None):
        """
        Evaluate on a split using proper per-score perturbation.

        Args:
            evaluator: AblationDetector instance for the evaluation split
            score_configs: Optional dict of per-score configs.

        Returns:
            Tuple of (lda_scores, detector_labels)
        """
        if not self.is_fitted or self.lda is None:
            raise ValueError("LDA must be fitted first via fit_lda_with_perturbation()")

        if score_configs is not None:
            self.score_configs = score_configs

        # Build score_configs dict
        configs = {}
        for score_name in self.base_scores:
            configs[score_name] = self._get_score_config(score_name)

        # Get scores with perturbation
        scores_dict, labels = evaluator.get_multi_score_with_perturbation(configs)

        # Stack and apply LDA
        X = np.column_stack([scores_dict[name] for name in self.base_scores])
        lda_scores = self.lda.transform(X).ravel()

        return lda_scores, labels

    @torch.no_grad()
    def fit(self, logits, detector_labels, **kwargs):
        """
        Standard fit method for compatibility.

        For proper perturbation support, use fit_lda_with_perturbation() instead.
        """
        self.fit_lda(logits, detector_labels)

    @torch.no_grad()
    def __call__(self, inputs=None, logits=None):
        """
        Apply LDA projection to get combined uncertainty score.

        Note: This method does NOT apply perturbation. For evaluation with
        perturbation, use evaluate_with_perturbation().

        Returns:
            Tensor of LDA scores [n_samples]
        """
        if logits is None:
            if inputs is None:
                raise ValueError("Either logits or inputs must be provided")
            logits = self.model(inputs)

        if not self.is_fitted or self.lda is None:
            # Return gini score if not fitted
            return gini(logits, temperature=self.default_temperature,
                        normalize=self.default_normalize).squeeze()

        lda_scores = self._apply_lda(logits)
        return torch.tensor(lda_scores, dtype=torch.float32, device=self.device)

    def get_diagnostics(self):
        """Return diagnostic information about the fitted model."""
        return {
            "is_fitted": self.is_fitted,
            "base_scores": self.base_scores,
            "score_configs": self.score_configs,
            "lda_coef": self.lda.coef_.tolist() if self.lda else None,
            "lda_intercept": self.lda.intercept_.tolist() if self.lda else None,
        }
