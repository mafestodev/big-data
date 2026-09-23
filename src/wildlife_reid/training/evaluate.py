from __future__ import annotations

import numpy as np


def top_k_accuracy(
    similarities: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
    k: int,
) -> float:
    """Calculate retrieval Top-k accuracy from a query-by-gallery score matrix."""

    if similarities.shape != (
        len(query_labels),
        len(gallery_labels),
    ):
        raise ValueError(
            "Similarity matrix dimensions do not match the labels"
        )

    if k <= 0:
        raise ValueError("k must be greater than zero")

    if len(gallery_labels) == 0:
        raise ValueError("Gallery labels cannot be empty")

    k = min(k, len(gallery_labels))

    nearest = np.argpartition(
        -similarities,
        kth=k - 1,
        axis=1,
    )[:, :k]

    predictions = gallery_labels[nearest]

    correct = (
        predictions == query_labels[:, None]
    ).any(axis=1)

    return float(correct.mean())


def top_1_accuracy(
    similarities: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
) -> float:
    """Calculate retrieval Top-1 accuracy."""

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
    """Calculate retrieval Top-5 accuracy."""

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
    """Calculate mean Average Precision for retrieval."""

    if similarities.shape != (
        len(query_labels),
        len(gallery_labels),
    ):
        raise ValueError(
            "Similarity matrix dimensions do not match the labels"
        )

    if len(gallery_labels) == 0:
        raise ValueError("Gallery labels cannot be empty")

    average_precisions = []

    for query_index, query_label in enumerate(query_labels):

        scores = similarities[query_index]

        ranking = np.argsort(
            -scores
        )

        ranked_labels = gallery_labels[ranking]

        relevant = (
            ranked_labels == query_label
        )

        number_of_relevant = int(
            relevant.sum()
        )

        if number_of_relevant == 0:
            continue

        relevant_positions = np.flatnonzero(
            relevant
        )

        precisions = []

        for position in relevant_positions:
            retrieved = position + 1
            relevant_retrieved = relevant[:retrieved].sum()

            precisions.append(
                relevant_retrieved / retrieved
            )

        average_precisions.append(
            float(np.mean(precisions))
        )

    if not average_precisions:
        return 0.0

    return float(
        np.mean(average_precisions)
    )