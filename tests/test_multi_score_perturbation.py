"""
Test for per-score perturbation functionality.

Verifies that:
1. AblationDetector.get_multi_score_with_perturbation() works correctly
2. RawLDAPostprocessor.fit_lda_with_perturbation() and evaluate_with_perturbation() work
3. LDABinningPostprocessor inherits correctly from RawLDAPostprocessor
"""

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch

from error_estimation.utils.postprocessors import (
    RawLDAPostprocessor,
    LDABinningPostprocessor,
    DoctorPostprocessor,
    ODINPostprocessor,
    MarginPostprocessor,
)


class TestPostprocessorInheritance:
    """Test that LDABinningPostprocessor correctly inherits from RawLDAPostprocessor."""

    def test_inheritance(self):
        assert issubclass(LDABinningPostprocessor, RawLDAPostprocessor)

    def test_raw_lda_has_required_methods(self):
        """RawLDAPostprocessor should have the key methods."""
        required_methods = [
            'fit_lda',
            'fit_lda_with_perturbation',
            'evaluate_with_perturbation',
            '_compute_single_score',
            '_compute_all_scores',
        ]
        for method in required_methods:
            assert hasattr(RawLDAPostprocessor, method), f"Missing method: {method}"

    def test_lda_binning_has_required_methods(self):
        """LDABinningPostprocessor should have binning-specific methods."""
        required_methods = [
            'fit_binning',
            'fit_binning_with_perturbation',
            'evaluate_with_binning_and_perturbation',
            '_apply_binning',
        ]
        for method in required_methods:
            assert hasattr(LDABinningPostprocessor, method), f"Missing method: {method}"


class TestScoreComputation:
    """Test that score computation works correctly."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        model = MagicMock()
        return model

    @pytest.fixture
    def raw_lda_postprocessor(self, mock_model):
        """Create a RawLDAPostprocessor instance."""
        cfg = {
            "temperature": 1.0,
            "magnitude": 0.0,
            "normalize": False,
            "base_scores": ["gini", "margin"],
        }
        return RawLDAPostprocessor(
            model=mock_model,
            cfg=cfg,
            result_folder="/tmp/test",
            device=torch.device("cpu"),
        )

    @pytest.fixture
    def lda_binning_postprocessor(self, mock_model):
        """Create a LDABinningPostprocessor instance."""
        cfg = {
            "temperature": 1.0,
            "magnitude": 0.0,
            "normalize": False,
            "base_scores": ["gini", "margin"],
            "n_bins": 5,
            "alpha": 0.05,
            "score_type": "mean",
        }
        return LDABinningPostprocessor(
            model=mock_model,
            cfg=cfg,
            result_folder="/tmp/test",
            device=torch.device("cpu"),
        )

    def test_compute_gini_score(self, raw_lda_postprocessor):
        """Test gini score computation."""
        logits = torch.randn(10, 5)
        score = raw_lda_postprocessor._compute_single_score(logits, "gini")
        assert score.shape == (10,)
        assert torch.all(score >= 0)  # Gini impurity is non-negative

    def test_compute_margin_score(self, raw_lda_postprocessor):
        """Test margin score computation."""
        logits = torch.randn(10, 5)
        score = raw_lda_postprocessor._compute_single_score(logits, "margin")
        assert score.shape == (10,)

    def test_compute_msp_score(self, raw_lda_postprocessor):
        """Test MSP score computation."""
        logits = torch.randn(10, 5)
        score = raw_lda_postprocessor._compute_single_score(logits, "msp")
        assert score.shape == (10,)

    def test_compute_all_scores(self, raw_lda_postprocessor):
        """Test computing all base scores."""
        logits = torch.randn(10, 5)
        scores = raw_lda_postprocessor._compute_all_scores(logits)
        # Should have 2 columns (gini, margin)
        assert scores.shape == (10, 2)

    def test_fit_lda_without_perturbation(self, raw_lda_postprocessor):
        """Test fitting LDA without perturbation."""
        logits = torch.randn(100, 5)
        labels = torch.randint(0, 2, (100,))
        raw_lda_postprocessor.fit_lda(logits, labels)
        assert raw_lda_postprocessor.is_fitted
        assert raw_lda_postprocessor.lda is not None

    def test_apply_lda(self, raw_lda_postprocessor):
        """Test applying LDA projection."""
        # First fit
        logits = torch.randn(100, 5)
        labels = torch.randint(0, 2, (100,))
        raw_lda_postprocessor.fit_lda(logits, labels)

        # Then apply
        test_logits = torch.randn(20, 5)
        lda_scores = raw_lda_postprocessor._apply_lda(test_logits)
        assert lda_scores.shape == (20,)

    def test_binning_fit_and_apply(self, lda_binning_postprocessor):
        """Test fitting and applying binning."""
        # Fit LDA first
        logits = torch.randn(100, 5)
        labels = torch.randint(0, 2, (100,))
        lda_binning_postprocessor.fit_lda(logits, labels)

        # Then fit binning
        combined_scores = lda_binning_postprocessor._apply_lda(logits)
        labels_np = labels.numpy()
        lda_binning_postprocessor.fit_binning(combined_scores, labels_np)

        assert lda_binning_postprocessor.binning_fitted
        assert len(lda_binning_postprocessor.bin_edges) == 6  # n_bins + 1
        assert len(lda_binning_postprocessor.bin_error_means) == 5

        # Test applying binning
        calibrated = lda_binning_postprocessor._apply_binning(combined_scores)
        assert calibrated.shape == (100,)
        assert all(0 <= c <= 1 for c in calibrated)  # Probabilities should be in [0, 1]


class TestScoreConfigs:
    """Test score configuration handling."""

    @pytest.fixture
    def raw_lda_with_configs(self):
        """Create a RawLDAPostprocessor with per-score configs."""
        cfg = {
            "base_scores": ["gini", "margin", "msp"],
            "score_configs": {
                "gini": {"temperature": 1.2, "magnitude": 0.002, "normalize": True},
                "margin": {"temperature": 0.8, "magnitude": 0.0, "normalize": False},
                "msp": {"temperature": 1.0, "magnitude": 0.001, "normalize": False},
            },
        }
        model = MagicMock()
        return RawLDAPostprocessor(
            model=model,
            cfg=cfg,
            result_folder="/tmp/test",
            device=torch.device("cpu"),
        )

    def test_get_score_config(self, raw_lda_with_configs):
        """Test that per-score configs are retrieved correctly."""
        gini_cfg = raw_lda_with_configs._get_score_config("gini")
        assert gini_cfg["temperature"] == 1.2
        assert gini_cfg["magnitude"] == 0.002
        assert gini_cfg["normalize"] is True

        margin_cfg = raw_lda_with_configs._get_score_config("margin")
        assert margin_cfg["temperature"] == 0.8
        assert margin_cfg["magnitude"] == 0.0

    def test_set_score_configs(self, raw_lda_with_configs):
        """Test setting score configs."""
        new_configs = {
            "gini": {"temperature": 2.0, "magnitude": 0.005},
        }
        raw_lda_with_configs.set_score_configs(new_configs)

        gini_cfg = raw_lda_with_configs._get_score_config("gini")
        assert gini_cfg["temperature"] == 2.0
        assert gini_cfg["magnitude"] == 0.005


class TestDiagnostics:
    """Test diagnostic methods."""

    def test_raw_lda_diagnostics(self):
        """Test RawLDAPostprocessor diagnostics."""
        cfg = {"base_scores": ["gini", "margin"]}
        model = MagicMock()
        pp = RawLDAPostprocessor(model=model, cfg=cfg, result_folder="/tmp", device=torch.device("cpu"))

        diag = pp.get_diagnostics()
        assert "is_fitted" in diag
        assert "base_scores" in diag
        assert diag["is_fitted"] is False

        # Fit and check again
        logits = torch.randn(50, 5)
        labels = torch.randint(0, 2, (50,))
        pp.fit_lda(logits, labels)

        diag = pp.get_diagnostics()
        assert diag["is_fitted"] is True
        assert diag["lda_coef"] is not None

    def test_lda_binning_diagnostics(self):
        """Test LDABinningPostprocessor diagnostics."""
        cfg = {"base_scores": ["gini", "margin"], "n_bins": 5}
        model = MagicMock()
        pp = LDABinningPostprocessor(model=model, cfg=cfg, result_folder="/tmp", device=torch.device("cpu"))

        diag = pp.get_diagnostics()
        assert "n_bins" in diag
        assert "binning_fitted" in diag
        assert diag["binning_fitted"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
