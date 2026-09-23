import numpy as np

from wildlife_reid.training.evaluate import (
    cosine_similarity_matrix,
    evaluate_retrieval,
    mean_average_precision,
    top_1_accuracy,
    top_5_accuracy,
)


def test_top_1_accuracy():
    similarities = np.array(
        [
            [0.9, 0.2, 0.1],
            [0.1, 0.8, 0.2],
            [0.2, 0.3, 0.7],
        ]
    )

    query_labels = np.array(["a", "b", "c"])
    gallery_labels = np.array(["a", "b", "c"])

    assert top_1_accuracy(
        similarities,
        query_labels,
        gallery_labels,
    ) == 1.0


def test_top_5_accuracy():
    similarities = np.array(
        [
            [0.9, 0.2, 0.1],
            [0.1, 0.8, 0.2],
            [0.2, 0.3, 0.7],
        ]
    )

    query_labels = np.array(["a", "b", "c"])
    gallery_labels = np.array(["a", "b", "c"])

    assert top_5_accuracy(
        similarities,
        query_labels,
        gallery_labels,
    ) == 1.0


def test_mean_average_precision_perfect_retrieval():
    similarities = np.array(
        [
            [0.9, 0.2, 0.1],
            [0.1, 0.8, 0.2],
            [0.2, 0.3, 0.7],
        ]
    )

    query_labels = np.array(["a", "b", "c"])
    gallery_labels = np.array(["a", "b", "c"])

    assert mean_average_precision(
        similarities,
        query_labels,
        gallery_labels,
    ) == 1.0


def test_top_k_handles_gallery_smaller_than_k():
    similarities = np.array([[0.9, 0.1]])
    query_labels = np.array(["a"])
    gallery_labels = np.array(["a", "b"])

    assert top_5_accuracy(
        similarities,
        query_labels,
        gallery_labels,
    ) == 1.0


def test_metrics_handle_imperfect_retrieval():
    similarities = np.array(
        [
            [0.1, 0.9, 0.2],
            [0.8, 0.2, 0.1],
        ]
    )

    query_labels = np.array(["a", "b"])
    gallery_labels = np.array(["a", "b", "c"])

    assert top_1_accuracy(
        similarities,
        query_labels,
        gallery_labels,
    ) == 0.0

    assert top_5_accuracy(
        similarities,
        query_labels,
        gallery_labels,
    ) == 1.0

    assert 0.0 <= mean_average_precision(
        similarities,
        query_labels,
        gallery_labels,
    ) <= 1.0


def test_cosine_similarity_matrix():
    query_embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    gallery_embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    similarities = cosine_similarity_matrix(
        query_embeddings,
        gallery_embeddings,
    )

    assert similarities.shape == (2, 2)

    assert np.allclose(
        similarities,
        np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        ),
    )


def test_evaluate_retrieval_end_to_end():
    query_embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    gallery_embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    query_labels = np.array(
        ["animal_a", "animal_b", "animal_c"]
    )

    gallery_labels = np.array(
        ["animal_a", "animal_b", "animal_c"]
    )

    results = evaluate_retrieval(
        query_embeddings=query_embeddings,
        gallery_embeddings=gallery_embeddings,
        query_labels=query_labels,
        gallery_labels=gallery_labels,
    )

    assert results["top_1"] == 1.0
    assert results["top_5"] == 1.0
    assert results["mAP"] == 1.0