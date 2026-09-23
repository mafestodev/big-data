from __future__ import annotations

import numpy as np


def top_k_accuracy(
    similarities: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
    k: int,
) -> float:
    """Calculate retrieval accuracy within the top k results."""

    similarities = np.asarray(similarities)
    query_labels = np.asarray(query_labels)
    gallery_labels = np.asarray(gallery_labels)

    if similarities.ndim != 2:
        raise ValueError("Similarities must be a two-dimensional array")

    if len(query_labels) != similarities.shape[0]:
        raise ValueError("Number of query labels must match number of queries")

    if len(gallery_labels) != similarities.shape[1]:
        raise ValueError("Number of gallery labels must match gallery size")

    if k <= 0:
        raise ValueError("k must be greater than zero")

    if len(gallery_labels) == 0:
        raise ValueError("Gallery cannot be empty")

    k = min(k, similarities.shape[1])

    correct = 0

    for query_index in range(similarities.shape[0]):
        ranked_indices = np.argsort(
            -similarities[query_index]
        )[:k]

        retrieved_labels = gallery_labels[ranked_indices]

        if query_labels[query_index] in retrieved_labels:
            correct += 1

    return correct / len(query_labels)


def top_1_accuracy(
    similarities: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
) -> float:
    """Calculate Top-1 retrieval accuracy."""

    return top_k_accuracy(
        similarities,
        query_labels,
        gallery_labels,
        k=1,
    )


def top_5_accuracy(
    similarities: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
) -> float:
    """Calculate Top-5 retrieval accuracy."""

    return top_k_accuracy(
        similarities,
        query_labels,
        gallery_labels,
        k=5,
    )


def mean_average_precision(
    similarities: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
) -> float:
    """Calculate mean average precision for retrieval."""

    similarities = np.asarray(similarities)
    query_labels = np.asarray(query_labels)
    gallery_labels = np.asarray(gallery_labels)

    if similarities.ndim != 2:
        raise ValueError("Similarities must be a two-dimensional array")

    if len(query_labels) != similarities.shape[0]:
        raise ValueError("Number of query labels must match number of queries")

    if len(gallery_labels) != similarities.shape[1]:
        raise ValueError("Number of gallery labels must match gallery size")

    if len(gallery_labels) == 0:
        raise ValueError("Gallery cannot be empty")

    average_precisions = []

    for query_index in range(similarities.shape[0]):
        ranked_indices = np.argsort(
            -similarities[query_index]
        )

        ranked_labels = gallery_labels[ranked_indices]

        relevant = ranked_labels == query_labels[query_index]

        total_relevant = int(np.sum(relevant))

        if total_relevant == 0:
            continue

        hits = 0
        precision_sum = 0.0

        for rank, is_relevant in enumerate(relevant, start=1):
            if is_relevant:
                hits += 1
                precision_sum += hits / rank

        average_precision = precision_sum / total_relevant
        average_precisions.append(average_precision)

    if not average_precisions:
        return 0.0

    return float(np.mean(average_precisions))


def cosine_similarity_matrix(
    query_embeddings: np.ndarray,
    gallery_embeddings: np.ndarray,
) -> np.ndarray:
    """Calculate cosine similarity between query and gallery embeddings."""

    query_embeddings = np.asarray(
        query_embeddings,
        dtype=np.float32,
    )

    gallery_embeddings = np.asarray(
        gallery_embeddings,
        dtype=np.float32,
    )

    if query_embeddings.ndim != 2:
        raise ValueError(
            "Query embeddings must be a two-dimensional array"
        )

    if gallery_embeddings.ndim != 2:
        raise ValueError(
            "Gallery embeddings must be a two-dimensional array"
        )

    if query_embeddings.shape[1] != gallery_embeddings.shape[1]:
        raise ValueError(
            "Query and gallery embeddings must have the same dimension"
        )

    query_norms = np.linalg.norm(
        query_embeddings,
        axis=1,
        keepdims=True,
    )

    gallery_norms = np.linalg.norm(
        gallery_embeddings,
        axis=1,
        keepdims=True,
    )

    if np.any(query_norms == 0):
        raise ValueError("Query embeddings cannot contain zero vectors")

    if np.any(gallery_norms == 0):
        raise ValueError("Gallery embeddings cannot contain zero vectors")

    normalized_queries = query_embeddings / query_norms
    normalized_gallery = gallery_embeddings / gallery_norms

    return normalized_queries @ normalized_gallery.T


def evaluate_retrieval(
    query_embeddings: np.ndarray,
    gallery_embeddings: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
) -> dict[str, float]:
    """Run the complete retrieval evaluation."""

    similarities = cosine_similarity_matrix(
        query_embeddings,
        gallery_embeddings,
    )

    return {
        "top_1": top_1_accuracy(
            similarities,
            query_labels,
            gallery_labels,
        ),
        "top_5": top_5_accuracy(
            similarities,
            query_labels,
            gallery_labels,
        ),
        "mAP": mean_average_precision(
            similarities,
            query_labels,
            gallery_labels,
        ),
    }