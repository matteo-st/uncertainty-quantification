"""
Supervised Partition Clustering

Risk-aware balanced recursive partition into axis-aligned hyperrectangles.
Splits are chosen to maximize error rate separation between children,
while enforcing minimum samples per leaf for valid confidence bounds.

Works for any number of dimensions (2D, 3D, etc.).
"""

import numpy as np
from typing import Optional, List, Tuple


class Node:
    """
    Tree node representing a region in the partition.

    Uses unique IDs for equality comparison to avoid issues with numpy arrays.
    """
    _id_counter = 0

    def __init__(
        self,
        indices: np.ndarray,
        bounds: np.ndarray,
        error_rate: float,
        is_leaf: bool = True,
        split_dim: Optional[int] = None,
        split_threshold: Optional[float] = None,
        left: Optional['Node'] = None,
        right: Optional['Node'] = None,
        leaf_id: Optional[int] = None,
    ):
        self.indices = indices  # Indices of samples in this node
        self.bounds = bounds  # Shape (D, 2): [[low_0, high_0], [low_1, high_1], ...]
        self.error_rate = error_rate  # Empirical error rate in this node

        # Tree structure
        self.is_leaf = is_leaf
        self.split_dim = split_dim  # Dimension to split on
        self.split_threshold = split_threshold  # Threshold value
        self.left = left  # Left child (values <= threshold)
        self.right = right  # Right child (values > threshold)

        # Leaf ID (assigned after tree is built)
        self.leaf_id = leaf_id

        # Unique ID for this node instance
        self._node_id = Node._id_counter
        Node._id_counter += 1

    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return self._node_id == other._node_id

    def __hash__(self):
        return hash(self._node_id)


class SupervisedPartition:
    """
    Risk-aware balanced recursive partition into axis-aligned hyperrectangles.

    Parameters
    ----------
    n_clusters : int
        Target number of leaf regions (B).
    min_samples_leaf : int or float, optional
        Minimum number of samples per leaf. If float in (0, 1), interpreted as
        fraction of total samples. Default: n_samples / n_clusters.
    max_samples_leaf : int or float, optional
        Maximum samples per leaf (for tighter uniformity). Default: None (no limit).
    quantiles : array-like, optional
        Quantile levels for candidate thresholds. Default: [0.1, 0.2, ..., 0.9].
    balance_penalty : bool, optional
        If True, penalize unbalanced splits using sqrt(n_L * n_R / n^2).
        Default: False.
    random_state : int, optional
        Random seed (for tie-breaking). Default: None.

    Attributes
    ----------
    root_ : Node
        Root node of the partition tree.
    leaves_ : List[Node]
        List of leaf nodes after fitting.
    n_leaves_ : int
        Actual number of leaves (may be < n_clusters if no valid splits).
    """

    def __init__(
        self,
        n_clusters: int = 30,
        min_samples_leaf: Optional[int] = None,
        max_samples_leaf: Optional[int] = None,
        quantiles: Optional[np.ndarray] = None,
        balance_penalty: bool = False,
        random_state: Optional[int] = None,
    ):
        self.n_clusters = n_clusters
        self.min_samples_leaf = min_samples_leaf
        self.max_samples_leaf = max_samples_leaf
        self.quantiles = quantiles if quantiles is not None else np.arange(0.1, 1.0, 0.1)
        self.balance_penalty = balance_penalty
        self.random_state = random_state

        self.root_ = None
        self.leaves_ = []
        self.n_leaves_ = 0
        self._n_dims = None

    def fit(self, X: np.ndarray, errors: np.ndarray) -> 'SupervisedPartition':
        """
        Build the partition tree.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_dims)
            Score vectors (e.g., [gini, margin] or [gini, margin, msp]).
        errors : ndarray of shape (n_samples,)
            Binary error labels (0 = correct, 1 = error).

        Returns
        -------
        self
        """
        X = np.asarray(X)
        errors = np.asarray(errors).astype(float)

        n_samples, n_dims = X.shape
        self._n_dims = n_dims
        self._X = X  # Store for predict
        self._errors = errors

        # Set min_samples_leaf
        if self.min_samples_leaf is None:
            min_samples = max(1, n_samples // self.n_clusters)
        elif isinstance(self.min_samples_leaf, float) and 0 < self.min_samples_leaf < 1:
            min_samples = max(1, int(n_samples * self.min_samples_leaf))
        else:
            min_samples = int(self.min_samples_leaf)
        self._min_samples = min_samples

        # Set max_samples_leaf
        if self.max_samples_leaf is None:
            self._max_samples = None
        elif isinstance(self.max_samples_leaf, float) and 0 < self.max_samples_leaf < 1:
            self._max_samples = int(n_samples * self.max_samples_leaf)
        else:
            self._max_samples = int(self.max_samples_leaf)

        # Initialize root node
        initial_bounds = np.column_stack([X.min(axis=0), X.max(axis=0)])  # (D, 2)
        self.root_ = Node(
            indices=np.arange(n_samples),
            bounds=initial_bounds,
            error_rate=errors.mean(),
        )

        # Build tree
        self.leaves_ = [self.root_]
        self._grow_tree(X, errors)

        # Assign leaf IDs
        for i, leaf in enumerate(self.leaves_):
            leaf.leaf_id = i
        self.n_leaves_ = len(self.leaves_)

        return self

    def _grow_tree(self, X: np.ndarray, errors: np.ndarray):
        """Grow the tree until we have n_clusters leaves or no valid splits."""
        while len(self.leaves_) < self.n_clusters:
            best_leaf = None
            best_split = None
            best_score = -np.inf

            for leaf in self.leaves_:
                split_info = self._find_best_split(leaf, X, errors)
                if split_info is not None:
                    dim, threshold, score = split_info
                    if score > best_score:
                        best_score = score
                        best_leaf = leaf
                        best_split = (dim, threshold)

            if best_leaf is None:
                # No valid splits found
                break

            # Perform the split
            self._split_node(best_leaf, best_split[0], best_split[1], X, errors)

    def _find_best_split(
        self, node: Node, X: np.ndarray, errors: np.ndarray
    ) -> Optional[Tuple[int, float, float]]:
        """
        Find the best split for a node.

        Returns
        -------
        (dim, threshold, score) or None if no valid split exists.
        """
        indices = node.indices
        n_node = len(indices)

        if n_node < 2 * self._min_samples:
            return None  # Can't split

        X_node = X[indices]
        errors_node = errors[indices]

        best_dim = None
        best_threshold = None
        best_score = -np.inf

        for dim in range(self._n_dims):
            values = X_node[:, dim]

            # Get candidate thresholds from quantiles
            thresholds = np.unique(np.percentile(values, self.quantiles * 100))

            for threshold in thresholds:
                left_mask = values <= threshold
                right_mask = ~left_mask

                n_left = left_mask.sum()
                n_right = right_mask.sum()

                # Check admissibility
                if n_left < self._min_samples or n_right < self._min_samples:
                    continue
                if self._max_samples is not None:
                    if n_left > self._max_samples or n_right > self._max_samples:
                        continue

                # Compute error rates
                eta_left = errors_node[left_mask].mean() if n_left > 0 else 0
                eta_right = errors_node[right_mask].mean() if n_right > 0 else 0

                # Compute split score (error separation)
                score = abs(eta_left - eta_right)

                if self.balance_penalty:
                    # Penalize unbalanced splits
                    balance = np.sqrt(n_left * n_right) / n_node
                    score *= balance

                if score > best_score:
                    best_score = score
                    best_dim = dim
                    best_threshold = threshold

        if best_dim is None:
            return None

        return (best_dim, best_threshold, best_score)

    def _split_node(
        self, node: Node, dim: int, threshold: float, X: np.ndarray, errors: np.ndarray
    ):
        """Split a node into two children."""
        indices = node.indices
        values = X[indices, dim]

        left_mask = values <= threshold
        left_indices = indices[left_mask]
        right_indices = indices[~left_mask]

        # Update bounds
        left_bounds = node.bounds.copy()
        left_bounds[dim, 1] = threshold  # Upper bound for left

        right_bounds = node.bounds.copy()
        right_bounds[dim, 0] = threshold  # Lower bound for right

        # Create child nodes
        left_child = Node(
            indices=left_indices,
            bounds=left_bounds,
            error_rate=errors[left_indices].mean(),
        )
        right_child = Node(
            indices=right_indices,
            bounds=right_bounds,
            error_rate=errors[right_indices].mean(),
        )

        # Update parent
        node.is_leaf = False
        node.split_dim = dim
        node.split_threshold = threshold
        node.left = left_child
        node.right = right_child

        # Update leaves list
        self.leaves_.remove(node)
        self.leaves_.append(left_child)
        self.leaves_.append(right_child)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Assign samples to leaf regions.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_dims)
            Score vectors.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Leaf IDs for each sample.
        """
        X = np.asarray(X)
        n_samples = X.shape[0]
        labels = np.zeros(n_samples, dtype=int)

        for i in range(n_samples):
            labels[i] = self._predict_single(X[i])

        return labels

    def _predict_single(self, x: np.ndarray) -> int:
        """Traverse tree to find leaf for a single sample."""
        node = self.root_

        while not node.is_leaf:
            if x[node.split_dim] <= node.split_threshold:
                node = node.left
            else:
                node = node.right

        return node.leaf_id

    def fit_predict(self, X: np.ndarray, errors: np.ndarray) -> np.ndarray:
        """Fit and return cluster labels."""
        self.fit(X, errors)
        return self.predict(X)

    def get_leaf_info(self) -> List[dict]:
        """
        Get information about each leaf.

        Returns
        -------
        List of dicts with keys: 'leaf_id', 'n_samples', 'error_rate', 'bounds'
        """
        info = []
        for leaf in self.leaves_:
            info.append({
                'leaf_id': leaf.leaf_id,
                'n_samples': len(leaf.indices),
                'error_rate': leaf.error_rate,
                'bounds': leaf.bounds.copy(),
            })
        return info

    @property
    def cluster_centers_(self) -> np.ndarray:
        """
        Return cluster centers (centroids of each leaf region).
        For compatibility with k-means interface.
        """
        centers = []
        for leaf in self.leaves_:
            # Centroid of samples in leaf
            center = self._X[leaf.indices].mean(axis=0)
            centers.append(center)
        return np.array(centers)

    @property
    def labels_(self) -> np.ndarray:
        """Return labels for training data (for compatibility)."""
        return self.predict(self._X)
