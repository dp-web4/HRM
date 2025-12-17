#!/usr/bin/env python3
"""
Context Classifier for SAGE Contextual Trust

Classifies input embeddings into semantic contexts to enable context-dependent
expert reputation tracking and selection.

Web4 Pattern: MRH (Minimal Resonance Hypothesis)
- Different contexts create different resonance patterns
- Expert performance varies by input context (code vs prose vs reasoning)
- Contextual trust more accurate than global trust

Design Philosophy:
- Unsupervised learning (no labeled data required)
- Online adaptation (contexts evolve over time)
- Lightweight (fast classification during inference)
- Interpretable (context IDs map to semantic clusters)

Implementation Approaches:
1. Clustering-based (k-means, DBSCAN) - implemented first
2. Heuristic-based (token distribution, attention patterns) - future
3. Learned classifier (small network) - future if needed

Created: Session 57 (2025-12-16)
Part of: Web4 → SAGE pattern transfer (contextual trust)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import json
import pickle
import time

try:
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.preprocessing import normalize
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    MiniBatchKMeans = None
    normalize = None


@dataclass
class ContextInfo:
    """Information about a classified context."""
    context_id: str             # Unique context identifier (e.g., "cluster_0")
    cluster_index: int          # Numeric cluster index
    centroid_distance: float    # Distance from embedding to centroid
    confidence: float           # Classification confidence (0-1)
    description: Optional[str] = None  # Human-readable description (optional)
    sample_count: int = 0       # Number of samples in this cluster
    last_updated: float = 0.0   # Timestamp of last cluster update


@dataclass
class ContextStats:
    """Statistics about context classification."""
    total_classifications: int
    num_contexts: int
    context_distribution: Dict[str, int]  # {context_id: count}
    avg_confidence: float
    cluster_quality: float  # Silhouette score or similar (if available)
    last_trained: float     # Timestamp of last training


class ContextClassifier:
    """
    Classifies input embeddings into semantic contexts using clustering.

    Uses MiniBatchKMeans for online learning with partial_fit:
    - Start with initial clustering on small dataset
    - Adapt online as new embeddings seen
    - Periodic retraining to adjust cluster boundaries

    Context classification enables:
    - Context-dependent expert reputation
    - Trust scores that vary by input type
    - Smarter expert selection and substitution

    Web4 Pattern: MRH applied to expert selection
    - Same expert, different effectiveness in different contexts
    - Contextual trust captures this variation
    """

    def __init__(
        self,
        num_contexts: int = 20,
        embedding_dim: Optional[int] = None,
        normalize_embeddings: bool = True,
        confidence_threshold: float = 0.5,
        retrain_interval: int = 1000,  # Retrain after N classifications
        cache_dir: Optional[Path] = None
    ):
        """
        Initialize context classifier.

        Args:
            num_contexts: Number of semantic contexts (clusters)
            embedding_dim: Expected embedding dimension (validated on first use)
            normalize_embeddings: L2-normalize embeddings before clustering
            confidence_threshold: Min confidence for classification (< threshold → unknown)
            retrain_interval: Retrain cluster model every N classifications
            cache_dir: Directory to save/load cluster model
        """
        if not HAS_SKLEARN:
            raise ImportError(
                "scikit-learn required for ContextClassifier. "
                "Install with: pip install scikit-learn"
            )

        self.num_contexts = num_contexts
        self.embedding_dim = embedding_dim
        self.normalize_embeddings = normalize_embeddings
        self.confidence_threshold = confidence_threshold
        self.retrain_interval = retrain_interval
        self.cache_dir = Path(cache_dir) if cache_dir else None

        # Clustering model
        self.clusterer: Optional[MiniBatchKMeans] = None
        self.is_fitted = False

        # Statistics
        self.total_classifications = 0
        self.context_counts: Dict[str, int] = {}
        self.confidence_sum = 0.0
        self.last_retrain = 0

        # Buffered embeddings for retraining
        self.embedding_buffer: List[np.ndarray] = []
        self.buffer_max_size = retrain_interval * 2

        # Context metadata
        self.context_info: Dict[str, ContextInfo] = {}

    def fit(self, embeddings: np.ndarray, force: bool = False) -> None:
        """
        Fit clustering model on embeddings.

        Args:
            embeddings: [N, embedding_dim] array
            force: Force retraining even if already fitted
        """
        if self.is_fitted and not force:
            # Already fitted, use partial_fit instead
            self.partial_fit(embeddings)
            return

        # Validate embedding dimension
        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2D embeddings, got shape {embeddings.shape}")

        if self.embedding_dim is None:
            self.embedding_dim = embeddings.shape[1]
        elif embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.embedding_dim}, "
                f"got {embeddings.shape[1]}"
            )

        # Normalize if requested
        if self.normalize_embeddings:
            embeddings = normalize(embeddings, axis=1, norm='l2')

        # Initialize clusterer
        self.clusterer = MiniBatchKMeans(
            n_clusters=self.num_contexts,
            random_state=42,
            batch_size=min(256, len(embeddings)),
            max_iter=100
        )

        # Fit on data
        self.clusterer.fit(embeddings)
        self.is_fitted = True
        self.last_retrain = self.total_classifications

        # Update context info
        self._update_context_info(embeddings)

        # Save model if cache dir specified
        if self.cache_dir:
            self.save()

    def partial_fit(self, embeddings: np.ndarray) -> None:
        """
        Update clustering model with new embeddings (online learning).

        Args:
            embeddings: [N, embedding_dim] array
        """
        if not self.is_fitted:
            # First time, do full fit
            self.fit(embeddings)
            return

        # Normalize if requested
        if self.normalize_embeddings:
            embeddings = normalize(embeddings, axis=1, norm='l2')

        # Update clustering model
        self.clusterer.partial_fit(embeddings)

        # Buffer embeddings for periodic retraining
        self.embedding_buffer.extend(embeddings.tolist())
        if len(self.embedding_buffer) > self.buffer_max_size:
            # Keep most recent embeddings
            self.embedding_buffer = self.embedding_buffer[-self.buffer_max_size:]

        # Periodic full retraining
        if self.total_classifications - self.last_retrain >= self.retrain_interval:
            if len(self.embedding_buffer) >= self.num_contexts * 2:
                # Enough data for meaningful retraining
                buffer_array = np.array(self.embedding_buffer)
                self.fit(buffer_array, force=True)

    def classify(self, embedding: np.ndarray) -> ContextInfo:
        """
        Classify a single embedding into a context.

        Args:
            embedding: [embedding_dim] or [1, embedding_dim] array

        Returns:
            ContextInfo with classification results
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Classifier not fitted. Call fit() with training embeddings first."
            )

        # Reshape if needed
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        # Normalize if requested
        if self.normalize_embeddings:
            embedding = normalize(embedding, axis=1, norm='l2')

        # Predict cluster
        cluster_idx = self.clusterer.predict(embedding)[0]
        context_id = f"context_{cluster_idx}"

        # Compute distance to centroid (for confidence)
        centroid = self.clusterer.cluster_centers_[cluster_idx]
        distance = np.linalg.norm(embedding[0] - centroid)

        # Confidence: inverse of distance (normalized)
        # Heuristic: confidence = exp(-distance²)
        confidence = float(np.exp(-distance ** 2))

        # Update statistics
        self.total_classifications += 1
        self.context_counts[context_id] = self.context_counts.get(context_id, 0) + 1
        self.confidence_sum += confidence

        # Get or create context info
        if context_id not in self.context_info:
            self.context_info[context_id] = ContextInfo(
                context_id=context_id,
                cluster_index=cluster_idx,
                centroid_distance=distance,
                confidence=confidence,
                sample_count=1,
                last_updated=time.time()
            )
        else:
            # Update existing info
            info = self.context_info[context_id]
            info.centroid_distance = distance
            info.confidence = confidence
            info.sample_count += 1
            info.last_updated = time.time()

        return self.context_info[context_id]

    def classify_batch(self, embeddings: np.ndarray) -> List[ContextInfo]:
        """
        Classify multiple embeddings.

        Args:
            embeddings: [N, embedding_dim] array

        Returns:
            List of ContextInfo, one per embedding
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Classifier not fitted. Call fit() with training embeddings first."
            )

        # Normalize if requested
        if self.normalize_embeddings:
            embeddings = normalize(embeddings, axis=1, norm='l2')

        # Predict clusters
        cluster_indices = self.clusterer.predict(embeddings)

        # Compute distances and confidences
        results = []
        for i, cluster_idx in enumerate(cluster_indices):
            context_id = f"context_{cluster_idx}"
            centroid = self.clusterer.cluster_centers_[cluster_idx]
            distance = np.linalg.norm(embeddings[i] - centroid)
            confidence = float(np.exp(-distance ** 2))

            # Update statistics
            self.total_classifications += 1
            self.context_counts[context_id] = self.context_counts.get(context_id, 0) + 1
            self.confidence_sum += confidence

            # Create context info
            info = ContextInfo(
                context_id=context_id,
                cluster_index=cluster_idx,
                centroid_distance=distance,
                confidence=confidence,
                sample_count=self.context_counts[context_id],
                last_updated=time.time()
            )
            results.append(info)

        return results

    def get_statistics(self) -> ContextStats:
        """
        Get classifier statistics.

        Returns:
            ContextStats with current statistics
        """
        avg_confidence = (
            self.confidence_sum / max(1, self.total_classifications)
        )

        # TODO: Compute cluster quality (silhouette score) if enough data
        cluster_quality = 0.0

        return ContextStats(
            total_classifications=self.total_classifications,
            num_contexts=len(self.context_counts),
            context_distribution=self.context_counts.copy(),
            avg_confidence=avg_confidence,
            cluster_quality=cluster_quality,
            last_trained=self.last_retrain
        )

    def _update_context_info(self, embeddings: np.ndarray) -> None:
        """
        Update context info after training.

        Args:
            embeddings: Training embeddings used to fit model
        """
        # Predict clusters for training data
        cluster_indices = self.clusterer.predict(embeddings)

        # Count samples per cluster
        unique, counts = np.unique(cluster_indices, return_counts=True)
        cluster_counts = dict(zip(unique, counts))

        # Update context info
        for cluster_idx in range(self.num_contexts):
            context_id = f"context_{cluster_idx}"
            sample_count = cluster_counts.get(cluster_idx, 0)

            self.context_info[context_id] = ContextInfo(
                context_id=context_id,
                cluster_index=cluster_idx,
                centroid_distance=0.0,
                confidence=1.0,
                sample_count=sample_count,
                last_updated=time.time()
            )

    def save(self, path: Optional[Path] = None) -> None:
        """
        Save classifier state to disk.

        Args:
            path: Path to save (defaults to cache_dir/context_classifier)
        """
        if path is None:
            if self.cache_dir is None:
                raise ValueError("No cache_dir specified and no explicit path provided")
            path = self.cache_dir / "context_classifier"

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save configuration as JSON
        config = {
            'num_contexts': self.num_contexts,
            'embedding_dim': self.embedding_dim,
            'normalize_embeddings': self.normalize_embeddings,
            'confidence_threshold': self.confidence_threshold,
            'retrain_interval': self.retrain_interval,
            'is_fitted': self.is_fitted,
            'total_classifications': self.total_classifications,
            'context_counts': self.context_counts,
            'confidence_sum': self.confidence_sum,
            'last_retrain': self.last_retrain,
        }

        with open(f"{path}.json", 'w') as f:
            json.dump(config, f, indent=2)

        # Save sklearn model with pickle (if fitted)
        if self.is_fitted and self.clusterer is not None:
            with open(f"{path}.pkl", 'wb') as f:
                pickle.dump(self.clusterer, f)

    def load(self, path: Optional[Path] = None) -> None:
        """
        Load classifier state from disk.

        Args:
            path: Path to load (defaults to cache_dir/context_classifier)
        """
        if path is None:
            if self.cache_dir is None:
                raise ValueError("No cache_dir specified and no explicit path provided")
            path = self.cache_dir / "context_classifier"

        path = Path(path)
        json_path = Path(f"{path}.json")
        pkl_path = Path(f"{path}.pkl")

        if not json_path.exists():
            raise FileNotFoundError(f"Classifier config not found at {json_path}")

        # Load configuration
        with open(json_path, 'r') as f:
            config = json.load(f)

        # Restore configuration
        self.num_contexts = config['num_contexts']
        self.embedding_dim = config['embedding_dim']
        self.normalize_embeddings = config['normalize_embeddings']
        self.confidence_threshold = config['confidence_threshold']
        self.retrain_interval = config['retrain_interval']

        # Restore statistics
        self.is_fitted = config['is_fitted']
        self.total_classifications = config['total_classifications']
        self.context_counts = config['context_counts']
        self.confidence_sum = config['confidence_sum']
        self.last_retrain = config['last_retrain']

        # Restore cluster model (if fitted)
        if self.is_fitted and pkl_path.exists():
            with open(pkl_path, 'rb') as f:
                self.clusterer = pickle.load(f)


# Convenience functions

def create_context_classifier(
    num_contexts: int = 20,
    embedding_dim: Optional[int] = None,
    cache_dir: Optional[Path] = None
) -> ContextClassifier:
    """
    Create a context classifier with default settings.

    Args:
        num_contexts: Number of semantic contexts
        embedding_dim: Expected embedding dimension
        cache_dir: Directory for model persistence

    Returns:
        ContextClassifier instance
    """
    return ContextClassifier(
        num_contexts=num_contexts,
        embedding_dim=embedding_dim,
        cache_dir=cache_dir
    )


def classify_context(
    classifier: ContextClassifier,
    embedding: np.ndarray
) -> str:
    """
    Classify embedding and return context ID.

    Args:
        classifier: Fitted ContextClassifier
        embedding: Input embedding

    Returns:
        Context ID string (e.g., "context_5")
    """
    info = classifier.classify(embedding)
    return info.context_id if info.confidence >= classifier.confidence_threshold else "context_unknown"
