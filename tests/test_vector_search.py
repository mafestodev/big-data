import numpy as np

from wildlife_reid.gallery.vector_search import ExactCosineIndex


def test_search_returns_closest_identity():
    index = ExactCosineIndex(
        embeddings=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        identities=np.array(["animal-a", "animal-b"]),
    )

    result = index.search(np.array([0.9, 0.1]), k=1)[0]

    assert result.identity == "animal-a"
    assert result.similarity > 0.9

