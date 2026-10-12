
from __future__ import annotations

import numpy as np


def cosine_similarity_matrix(
    query_embeddings: np.ndarray,
    gallery_embeddings: np.ndarray,
) -> np.ndarray:
    """Calculate cosine similarities between queries and gallery images."""

    queries = np.asarray(query_embeddings, dtype=np.float32)
    gallery = np.asarray(gallery_embeddings, dtype=np.float32)

    if queries.ndim != 2 or gallery.ndim != 2:
        raise ValueError("Embeddings must be two-dimensional")

    if queries.shape[1] != gallery.shape[1]:
        raise ValueError(
            "Query and gallery embeddings must have the same dimension"
        )

    query_norms = np.linalg.norm(queries, axis=1, keepdims=True)
    gallery_norms = np.linalg.norm(gallery, axis=1, keepdims=True)

    if np.any(query_norms == 0) or np.any(gallery_norms == 0):
        raise ValueError("Embeddings cannot contain zero vectors")

    queries = queries / query_norms
    gallery = gallery / gallery_norms

    return queries @ gallery.T


def top_k_accuracy(
    similarities: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
    k: int,
) -> float:
    """Calculate the fraction of queries correctly identified in the top k."""

    similarities = np.asarray(similarities)
    query_labels = np.asarray(query_labels)
    gallery_labels = np.asarray(gallery_labels)

    if similarities.ndim != 2:
        raise ValueError("Similarities must be two-dimensional")

    if len(query_labels) != similarities.shape[0]:
        raise ValueError("Query label count does not match similarities")

    if len(gallery_labels) != similarities.shape[1]:
        raise ValueError("Gallery label count does not match similarities")

    if k <= 0:
        raise ValueError("k must be greater than zero")

    if len(query_labels) == 0 or len(gallery_labels) == 0:
        raise ValueError("Query and gallery labels cannot be empty")

    k = min(k, similarities.shape[1])
    correct = 0

    for i in range(len(query_labels)):
        indices = np.argsort(-similarities[i])[:k]
        if np.any(gallery_labels[indices] == query_labels[i]):
            correct += 1

    return float(correct / len(query_labels))


def top_1_accuracy(
    similarities: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
) -> float:
    """Calculate Top-1 retrieval accuracy."""

    return top_k_accuracy(
        similarities, query_labels, gallery_labels, k=1
    )


def top_5_accuracy(
    similarities: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
) -> float:
    """Calculate Top-5 retrieval accuracy."""

    return top_k_accuracy(
        similarities, query_labels, gallery_labels, k=5
    )


def mean_average_precision(
    similarities: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
) -> float:
    """Calculate mAP, excluding queries with no relevant gallery images."""

    similarities = np.asarray(similarities)
    query_labels = np.asarray(query_labels)
    gallery_labels = np.asarray(gallery_labels)

    if similarities.ndim != 2:
        raise ValueError("Similarities must be two-dimensional")

    if len(query_labels) != similarities.shape[0]:
        raise ValueError("Query label count does not match similarities")

    if len(gallery_labels) != similarities.shape[1]:
        raise ValueError("Gallery label count does not match similarities")

    if len(gallery_labels) == 0:
        raise ValueError("Gallery cannot be empty")

    average_precisions = []

    for i in range(len(query_labels)):
        relevant_total = int(np.count_nonzero(
            gallery_labels == query_labels[i]
        ))

        if relevant_total == 0:
            continue

        order = np.argsort(-similarities[i])
        relevant = gallery_labels[order] == query_labels[i]

        hits = np.cumsum(relevant)
        ranks = np.arange(1, len(relevant) + 1)

        ap = np.sum((hits / ranks) * relevant) / relevant_total
        average_precisions.append(float(ap))

    return float(np.mean(average_precisions)) if average_precisions else 0.0


def evaluate_retrieval(
    query_embeddings: np.ndarray,
    gallery_embeddings: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
    batch_size: int = 16,
) -> dict[str, float]:
    """
    Evaluate closed-set retrieval in batches to limit memory use.

    Queries whose identities are absent from the gallery are reported
    separately and excluded from closed-set metrics.

    Returns metrics as fractions between 0 and 1, plus query counts.
    """

    queries = np.asarray(query_embeddings, dtype=np.float32)
    gallery = np.asarray(gallery_embeddings, dtype=np.float32)
    query_labels = np.asarray(query_labels)
    gallery_labels = np.asarray(gallery_labels)

    if queries.ndim != 2 or gallery.ndim != 2:
        raise ValueError("Embeddings must be two-dimensional")

    if queries.shape[1] != gallery.shape[1]:
        raise ValueError(
            "Query and gallery embeddings must have the same dimension"
        )

    if len(queries) != len(query_labels):
        raise ValueError("Query embedding and label counts do not match")

    if len(gallery) != len(gallery_labels):
        raise ValueError("Gallery embedding and label counts do not match")

    if len(queries) == 0 or len(gallery) == 0:
        raise ValueError("Query and gallery sets cannot be empty")

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")

    query_norms = np.linalg.norm(queries, axis=1, keepdims=True)
    gallery_norms = np.linalg.norm(gallery, axis=1, keepdims=True)

    if np.any(query_norms == 0) or np.any(gallery_norms == 0):
        raise ValueError("Embeddings cannot contain zero vectors")

    queries = queries / query_norms
    gallery = gallery / gallery_norms

    gallery_id_set = set(gallery_labels.tolist())
    covered = np.array(
        [label in gallery_id_set for label in query_labels],
        dtype=bool,
    )

    known_count = int(covered.sum())
    unseen_count = int((~covered).sum())

    if known_count == 0:
        raise ValueError(
            "No query identities occur in the gallery; "
            "closed-set metrics cannot be calculated."
        )

    top1_hits = 0
    top5_hits = 0
    precision5_sum = 0.0
    ap_sum = 0.0

    gallery_size = len(gallery_labels)
    k = min(5, gallery_size)

    for start in range(0, len(queries), batch_size):
        end = min(start + batch_size, len(queries))

        # Only this batch's similarity matrix is held in memory.
        similarities = queries[start:end] @ gallery.T

        for local_idx, query_idx in enumerate(range(start, end)):
            if not covered[query_idx]:
                continue

            order = np.argsort(-similarities[local_idx])
            ranked_labels = gallery_labels[order]
            relevant = ranked_labels == query_labels[query_idx]

            relevant_total = int(relevant.sum())
            if relevant_total == 0:
                raise RuntimeError(
                    "A known-identity query has no relevant gallery images."
                )

            top1_hits += int(relevant[0])
            top5_hits += int(relevant[:k].any())
            precision5_sum += float(relevant[:k].mean())

            hits = np.cumsum(relevant)
            ranks = np.arange(1, len(relevant) + 1)
            ap_sum += float(
                np.sum((hits / ranks) * relevant) / relevant_total
            )

        print(f"Processed {end:,}/{len(queries):,} queries")

    return {
        "top_1": float(top1_hits / known_count),
        "top_5": float(top5_hits / known_count),
        "precision_at_5": float(precision5_sum / known_count),
        "mAP": float(ap_sum / known_count),
        "known_queries": known_count,
        "unseen_queries": unseen_count,
        "identity_coverage": float(known_count / len(queries)),
    }
