import numpy as np


def cosine_similarity_matrix(query_embeddings, gallery_embeddings):
    """
    Compute cosine similarity between every query and gallery embedding.
    """
    queries = np.asarray(query_embeddings, dtype=np.float32)
    gallery = np.asarray(gallery_embeddings, dtype=np.float32)

    query_norms = np.linalg.norm(queries, axis=1, keepdims=True)
    gallery_norms = np.linalg.norm(gallery, axis=1, keepdims=True)

    # Avoid division by zero.
    queries = queries / np.clip(query_norms, 1e-12, None)
    gallery = gallery / np.clip(gallery_norms, 1e-12, None)

    return queries @ gallery.T


def rank_gallery(query_embeddings, gallery_embeddings, gallery_ids):
    """
    Rank gallery identities for every query using cosine similarity.
    """
    similarities = cosine_similarity_matrix(query_embeddings, gallery_embeddings)
    gallery_ids = np.asarray(gallery_ids)

    rankings = []
    for row in similarities:
        order = np.argsort(-row)
        rankings.append(gallery_ids[order].tolist())

    return rankings, similarities


def top_k_accuracy(rankings, query_ids, k=1):
    """
    Percentage of queries whose correct identity appears in the top k results.
    """
    query_ids = np.asarray(query_ids)

    correct = 0
    for ranking, true_id in zip(rankings, query_ids):
        if true_id in ranking[:k]:
            correct += 1

    return correct / len(query_ids) if len(query_ids) else 0.0


def average_precision(ranking, true_id):
    """
    Average Precision for one query
    """
    relevant_total = sum(gallery_id == true_id for gallery_id in ranking)

    if relevant_total == 0:
        return 0.0

    hits = 0
    precision_sum = 0.0

    for rank, gallery_id in enumerate(ranking, start=1):
        if gallery_id == true_id:
            hits += 1
            precision_sum += hits / rank

    return precision_sum / relevant_total


def mean_average_precision(rankings, query_ids):
    """
    Mean Average Precision across all queries.
    """
    aps = [
        average_precision(ranking, true_id)
        for ranking, true_id in zip(rankings, query_ids)
    ]

    return float(np.mean(aps)) if aps else 0.0


def evaluate(query_embeddings, query_ids, gallery_embeddings, gallery_ids):
    """
    Run the complete evaluation.
    """
    rankings, similarities = rank_gallery(
        query_embeddings,
        gallery_embeddings,
        gallery_ids,
    )

    results = {
        "top_1_accuracy": top_k_accuracy(rankings, query_ids, k=1),
        "top_5_accuracy": top_k_accuracy(rankings, query_ids, k=5),
        "mAP": mean_average_precision(rankings, query_ids),
        "rankings": rankings,
        "similarities": similarities,
    }

    return results
