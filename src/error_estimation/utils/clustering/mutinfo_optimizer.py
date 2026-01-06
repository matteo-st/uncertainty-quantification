from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
from torch import Tensor

from error_estimation.utils.clustering.kmeans import KMeans


class MutInfoOptimizer:
    """Fallback mutinfo optimizer using k-means clustering.

    This keeps the interface expected by the partition postprocessor while
    avoiding import errors when the original optimizer is unavailable.
    """

    def __init__(
        self,
        n_clusters: int,
        cov_type: str | None = None,
        sigmas: Tensor | None = None,
        max_iter: int | None = None,
        lr: float | None = None,
        init_scheme: str | None = None,
        n_init: int | None = None,
        seed: int | None = None,
        device: torch.device | None = None,
        with_logdet: bool | None = None,
        cov_proj_type: str | None = None,
        sig_init: float | None = None,
        alpha: float | None = None,
        mutual_computation: str | None = None,
        **kwargs: Any,
    ) -> None:
        del cov_type, sigmas, lr, with_logdet, cov_proj_type, sig_init, alpha, mutual_computation, kwargs
        self.n_clusters = n_clusters
        self.device = device or torch.device("cpu")
        self.best_init: int | None = None

        init_method = "k-means++"
        if init_scheme:
            normalized = init_scheme.lower().replace("_", "-")
            if normalized in {"kmeans", "k-means", "kmeans++", "k-means++"}:
                init_method = "k-means++"
            elif normalized in {"random", "rand", "rnd"}:
                init_method = "random"
            elif normalized == "supervised":
                init_method = "supervised"

        self._kmeans = KMeans(
            init_method=init_method,
            num_init=n_init or 1,
            max_iter=max_iter or 300,
            n_clusters=n_clusters,
            verbose=False,
            seed=seed or 0,
            device=self.device,
            collect_info=False,
        )
        self.results = SimpleNamespace(best_iter=0)

    def fit_predict(self, probits: Tensor, errors: Tensor | None = None) -> Tensor:
        del errors
        clusters = self._kmeans.fit_predict(probits, k=self.n_clusters)
        best_iter = 0
        if isinstance(self._kmeans.n_iter, set) and self._kmeans.n_iter:
            best_iter = min(self._kmeans.n_iter)
        self.results = SimpleNamespace(best_iter=best_iter)
        self.best_init = getattr(self._kmeans, "best_init", None)
        return clusters

    def predict(self, probits: Tensor) -> Tensor:
        return self._kmeans.predict(probits)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._kmeans, name)
