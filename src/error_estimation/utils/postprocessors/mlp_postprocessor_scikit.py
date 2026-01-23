import copy
import math
from typing import Dict, Iterable, List, Optional, Sequence

import torch
from sklearn import metrics as sklearn_metrics
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .base_scikit_postprocessor import BaseScikitPostprocessor, gini


def _ensure_list(value) -> List[int]:
    if value is None:
        return []
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        return list(value)
    return [int(value)]


def _make_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01)
    if name == "gelu":
        return nn.GELU()
    if name == "elu":
        return nn.ELU()
    raise ValueError(f"Unsupported activation '{name}'")


class _FeedForwardBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, activation: str, dropout: float, use_batchnorm: bool):
        super().__init__()
        layers: List[nn.Module] = [nn.Linear(in_features, out_features)]
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(out_features))
        layers.append(_make_activation(activation))
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _MLPHead(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_dims: Sequence[int],
        activation: str,
        dropout: float,
        use_batchnorm: bool,
    ):
        super().__init__()
        dims = [in_features] + list(hidden_dims)
        blocks: List[nn.Module] = []
        for idx in range(len(dims) - 1):
            blocks.append(
                _FeedForwardBlock(
                    in_features=dims[idx],
                    out_features=dims[idx + 1],
                    activation=activation,
                    dropout=dropout,
                    use_batchnorm=use_batchnorm,
                )
            )
        self.backbone = nn.Sequential(*blocks)
        self.classifier = nn.Linear(dims[-1], 1)

        self._reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.backbone:
            x = self.backbone(x)
        return self.classifier(x)

    def _reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(m.bias, -bound, bound)


class MLPPostprocessor(BaseScikitPostprocessor):
    """
    Predict misclassification probabilities with a lightweight PyTorch MLP.

    The class follows the same feature-extraction interface as the scikit postprocessors
    (logits, softmax probabilities, or Gini features) but trains a neural head instead of
    handing control to sklearn.
    """

    def __init__(self, model, cfg, result_folder, device=torch.device("cpu")):
        super().__init__(model, cfg, result_folder, device)

        self.normalize_gini = cfg.get("normalize_gini", False)
        self.hidden_dims = _ensure_list(cfg.get("hidden_dims", [1024, 512]))
        self.activation = cfg.get("activation", "relu")
        self.dropout = float(cfg.get("dropout", 0.2))
        self.use_batchnorm = bool(cfg.get("use_batchnorm", False))

        self.num_epochs = int(cfg.get("num_epochs", 10))
        self.batch_size = int(cfg.get("batch_size", 256))
        self.learning_rate = float(cfg.get("lr", 1e-3))
        self.weight_decay = float(cfg.get("weight_decay", 1e-4))
        self.grad_clip_norm = float(cfg.get("grad_clip_norm", 0.0))
        self.positive_class_weight = cfg.get("positive_class_weight", None)
        self.scheduler_cfg = cfg.get("scheduler", None)

        # Combined scores support (for space="combined")
        self.base_scores: List[str] = cfg.get("base_scores", ["gini", "margin", "msp", "entropy"])
        self.score_configs: Dict[str, dict] = cfg.get("score_configs", {})
        self.normalize_combined: bool = cfg.get("normalize_combined", True)
        self._combined_score_stats: Dict[str, Dict[str, float]] = {}

        # Early stopping / validation split
        self.val_split: float = float(cfg.get("val_split", 0.2))
        self.patience: int = int(cfg.get("patience", 10))
        self.early_stopping_metric: str = cfg.get("early_stopping_metric", "fpr")  # "fpr" or "roc_auc"

        self.model_head: Optional[_MLPHead] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
        self.loss_history: List[float] = []

    def _build_head(self, input_dim: int) -> None:
        self.model_head = _MLPHead(
            in_features=input_dim,
            hidden_dims=self.hidden_dims,
            activation=self.activation,
            dropout=self.dropout,
            use_batchnorm=self.use_batchnorm,
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model_head.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        if self.scheduler_cfg:
            if self.scheduler_cfg.get("name", "").lower() == "cosine":
                t_max = int(self.scheduler_cfg.get("t_max", self.num_epochs))
                eta_min = float(self.scheduler_cfg.get("eta_min", 0.0))
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=t_max,
                    eta_min=eta_min,
                )
            else:
                raise ValueError(f"Unsupported scheduler '{self.scheduler_cfg.get('name')}'")

    def _make_dataloader(self, embs: torch.Tensor, labels: torch.Tensor) -> DataLoader:
        dataset = TensorDataset(embs.float(), labels.float().unsqueeze(1))
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

    # ---------- Combined Score Methods ----------
    def _get_score_config(self, score_name: str) -> dict:
        """Get configuration for a specific score."""
        if score_name in self.score_configs:
            return self.score_configs[score_name]
        return {"temperature": self.temperature, "normalize": False}

    def _compute_single_score(self, logits: torch.Tensor, score_name: str) -> torch.Tensor:
        """
        Compute a single uncertainty score from logits.

        Args:
            logits: Model logits (N, C)
            score_name: One of 'gini', 'msp', 'margin', 'entropy', 'doctor'

        Returns:
            Score tensor (N,) - higher = more uncertain
        """
        cfg = self._get_score_config(score_name)
        temperature = cfg.get("temperature", 1.0)
        normalize = cfg.get("normalize", False)

        if score_name in ["gini", "doctor"]:
            score = gini(logits, temperature=temperature, normalize=normalize).squeeze(-1)
        elif score_name == "msp":
            probs = torch.softmax(logits / temperature, dim=1)
            score = -probs.max(dim=1)[0]
        elif score_name == "margin":
            probs = torch.softmax(logits / temperature, dim=1)
            top2 = probs.topk(2, dim=1)[0]
            score = 1.0 - (top2[:, 0] - top2[:, 1])
        elif score_name == "entropy":
            probs = torch.softmax(logits / temperature, dim=1)
            score = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
        elif score_name == "max_logit":
            score = -logits.max(dim=1)[0]
        else:
            raise ValueError(f"Unknown score type: {score_name}")

        return score

    def _compute_combined_scores(self, logits: torch.Tensor, fit: bool = False) -> torch.Tensor:
        """
        Compute multiple scores and concatenate them.

        Args:
            logits: Model logits (N, C)
            fit: If True, compute and store normalization stats

        Returns:
            Combined scores tensor (N, n_scores)
        """
        scores_list = []

        for score_name in self.base_scores:
            score = self._compute_single_score(logits, score_name)  # (N,)

            if self.normalize_combined:
                if fit:
                    # Store stats for inference
                    self._combined_score_stats[score_name] = {
                        "min": score.min().item(),
                        "max": score.max().item(),
                    }

                # Normalize to [0, 1]
                if score_name in self._combined_score_stats:
                    stats = self._combined_score_stats[score_name]
                    min_val = stats["min"]
                    max_val = stats["max"]
                else:
                    min_val = score.min().item()
                    max_val = score.max().item()

                if max_val - min_val > 1e-8:
                    score = (score - min_val) / (max_val - min_val)
                else:
                    score = torch.zeros_like(score)

            scores_list.append(score)

        # Stack into (N, n_scores)
        return torch.stack(scores_list, dim=1)

    @torch.no_grad()
    def _extract_embeddings(self, x=None, logits=None, fit: bool = False):
        """
        Extract embeddings from the model.

        Supports 'combined' space in addition to parent class spaces (gini, probits, logits).

        Args:
            x: Input images (optional, if logits not provided)
            logits: Pre-computed logits (optional)
            fit: If True, store normalization stats (for combined space)
        """
        if logits is None:
            if x is None:
                raise ValueError("Either logits or inputs must be provided")
            self.model.to(self.device)
            logits = self.model(x)
            self.model.to(torch.device("cpu"))

        if self.quantiz_space == "combined":
            return self._compute_combined_scores(logits, fit=fit)
        else:
            # Delegate to parent for gini/probits/logits
            return super()._extract_embeddings(x=x, logits=logits)

    # ---------- Validation Metric Computation ----------
    def _compute_validation_metric(self, predictions: torch.Tensor, labels: torch.Tensor) -> float:
        """
        Compute validation metric for early stopping.

        Args:
            predictions: Predicted probabilities (N,)
            labels: Ground truth labels (N,)

        Returns:
            Metric value (lower is better for FPR, higher is better for ROC-AUC)
        """
        preds_np = predictions.cpu().numpy()
        labels_np = labels.cpu().numpy()

        if self.early_stopping_metric == "fpr":
            # Compute FPR at 95% TPR (lower is better)
            fprs, tprs, _ = sklearn_metrics.roc_curve(labels_np, preds_np)
            # Find FPR at TPR >= 0.95
            idxs = [i for i, t in enumerate(tprs) if t >= 0.95]
            if len(idxs) > 0:
                return fprs[min(idxs)]
            return 1.0
        elif self.early_stopping_metric == "roc_auc":
            # Compute ROC-AUC (higher is better)
            try:
                return sklearn_metrics.roc_auc_score(labels_np, preds_np)
            except ValueError:
                return 0.0
        else:
            raise ValueError(f"Unknown early stopping metric: {self.early_stopping_metric}")

    def fit(self, logits=None, detector_labels=None, val_logits=None, val_detector_labels=None, verbose: bool = False):
        if logits is None or detector_labels is None:
            raise ValueError("Both logits and detector_labels must be provided to fit the MLP postprocessor.")

        detector_labels = detector_labels.detach().to(torch.device("cpu"))

        # Check if external validation data is provided (e.g., res split)
        if val_logits is not None and val_detector_labels is not None:
            # Use external validation data for early stopping
            train_embs = self._extract_embeddings(logits=logits, fit=True).to(torch.device("cpu"))
            train_labels = detector_labels
            val_embs = self._extract_embeddings(logits=val_logits, fit=False).to(torch.device("cpu"))
            val_labels = val_detector_labels.detach().to(torch.device("cpu"))
            use_early_stopping = self.patience > 0
        elif self.val_split > 0 and self.patience > 0:
            # Split train data into train/val for early stopping
            n = len(logits)
            n_val = int(n * self.val_split)
            perm = torch.randperm(n)
            val_idx, train_idx = perm[:n_val], perm[n_val:]

            train_logits = logits[train_idx]
            train_labels = detector_labels[train_idx]
            val_logits_split = logits[val_idx]
            val_labels = detector_labels[val_idx]

            # Extract train embeddings with fit=True to store normalization stats
            train_embs = self._extract_embeddings(logits=train_logits, fit=True).to(torch.device("cpu"))
            # Extract val embeddings using the stored stats
            val_embs = self._extract_embeddings(logits=val_logits_split, fit=False).to(torch.device("cpu"))

            use_early_stopping = True
        else:
            # No validation split - use all data for training
            train_embs = self._extract_embeddings(logits=logits, fit=True).to(torch.device("cpu"))
            train_labels = detector_labels
            val_embs = None
            val_labels = None
            use_early_stopping = False

        # Build head
        if self.model_head is None:
            self._build_head(input_dim=train_embs.shape[1])

        if self.model_head is None or self.optimizer is None:
            raise RuntimeError("MLP head failed to initialise.")

        # Create dataloader for training
        dataloader = self._make_dataloader(train_embs, train_labels)

        # Loss function with optional class weighting
        pos_weight_tensor = None
        if self.positive_class_weight is not None:
            pos_weight_tensor = torch.tensor(float(self.positive_class_weight), device=self.device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

        # Early stopping state
        best_metric = float("inf") if self.early_stopping_metric == "fpr" else float("-inf")
        best_state = None
        patience_counter = 0

        self.loss_history.clear()
        for epoch in range(self.num_epochs):
            # ---------- Training ----------
            self.model_head.train()
            epoch_loss = 0.0
            for batch_embs, batch_labels in dataloader:
                batch_embs = batch_embs.to(self.device)
                batch_labels = batch_labels.to(self.device)

                logits_pred = self.model_head(batch_embs)
                loss = loss_fn(logits_pred, batch_labels)

                self.optimizer.zero_grad()
                loss.backward()
                if self.grad_clip_norm > 0.0:
                    nn.utils.clip_grad_norm_(self.model_head.parameters(), max_norm=self.grad_clip_norm)
                self.optimizer.step()

                epoch_loss += loss.item() * batch_embs.size(0)

            epoch_loss /= len(dataloader.dataset)
            self.loss_history.append(epoch_loss)

            if self.scheduler is not None:
                self.scheduler.step()

            # ---------- Validation & Early Stopping ----------
            if use_early_stopping:
                self.model_head.eval()
                with torch.no_grad():
                    val_pred = torch.sigmoid(self.model_head(val_embs.to(self.device))).squeeze(-1)

                val_metric = self._compute_validation_metric(val_pred, val_labels)

                # Check if improved (lower is better for FPR, higher for ROC-AUC)
                if self.early_stopping_metric == "fpr":
                    improved = val_metric < best_metric
                else:
                    improved = val_metric > best_metric

                if improved:
                    best_metric = val_metric
                    best_state = copy.deepcopy(self.model_head.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1

                if verbose:
                    print(
                        f"[MLPPostprocessor] epoch {epoch + 1}/{self.num_epochs} "
                        f"loss={epoch_loss:.6f} val_{self.early_stopping_metric}={val_metric:.4f} "
                        f"best={best_metric:.4f} patience={patience_counter}/{self.patience}"
                    )

                if patience_counter >= self.patience:
                    if verbose:
                        print(f"[MLPPostprocessor] Early stopping at epoch {epoch + 1}")
                    break
            else:
                if verbose:
                    print(f"[MLPPostprocessor] epoch {epoch + 1}/{self.num_epochs} loss={epoch_loss:.6f}")

        # Restore best model if early stopping was used
        if use_early_stopping and best_state is not None:
            self.model_head.load_state_dict(best_state)

        self.model_head.eval()
        return self

    @torch.no_grad()
    def __call__(self, x=None, logits=None):
        if self.model_head is None:
            raise RuntimeError("MLPPostprocessor must be fitted before calling.")

        if logits is None:
            if x is None:
                raise ValueError("Either logits or inputs must be provided")
            logits = self.model(x)

        embs = self._extract_embeddings(x=x, logits=logits).to(self.device)
        self.model_head.eval()
        logits_out = self.model_head(embs.float())
        return torch.sigmoid(logits_out).squeeze(-1)
