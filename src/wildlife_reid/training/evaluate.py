from __future__ import annotations

import numpy as np


def top_k_accuracy(
    similarities: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
    k: int,
) -> float:
    """Calculate retrieval Top-k accuracy from a query-by-gallery score matrix."""
    if similarities.shape != (len(query_labels), len(gallery_labels)):
        raise ValueError("Similarity matrix dimensions do not match the labels")
    k = min(k, len(gallery_labels))
    nearest = np.argpartition(-similarities, kth=k - 1, axis=1)[:, :k]
    predictions = gallery_labels[nearest]
    correct = (predictions == query_labels[:, None]).any(axis=1)
    return float(correct.mean())
