from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SearchResult:
    identity: str
    similarity: float
    image_path: str


class ExactCosineIndex:
    """Small, transparent baseline index for normalized gallery embeddings."""

    def __init__(
        self,
        embeddings: np.ndarray,
        identities: np.ndarray,
        image_paths: np.ndarray,
    ) -> None:
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a two-dimensional array")
        if len(embeddings) == 0:
            raise ValueError("Gallery must contain at least one embedding")
        if (
            len(embeddings) != len(identities)
            or len(embeddings) != len(image_paths)
        ):
            raise ValueError(
                "Gallery embeddings, identities and image paths must be aligned"
            )
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("Gallery contains a zero-length embedding")
        self.embeddings = (embeddings / norms).astype(np.float32)
        self.identities = identities.astype(str)
        self.image_paths = image_paths.astype(str)

    @classmethod
    def load(cls, path: Path) -> "ExactCosineIndex":
        with np.load(path, allow_pickle=False) as data:
            required_arrays = {
                "embeddings",
                "identities",
                "image_paths",
            }
            missing = required_arrays.difference(data.files)
            if missing:
                raise ValueError(
                    "Gallery is missing required arrays: "
                    + ", ".join(sorted(missing))
                    + ". Regenerate the gallery with the current builder."
                )
            return cls(
                embeddings=data["embeddings"],
                identities=data["identities"],
                image_paths=data["image_paths"],
            )

    def search(self, query: np.ndarray, k: int = 1) -> list[SearchResult]:
        vector = np.asarray(query, dtype=np.float32).reshape(-1)
        if vector.shape[0] != self.embeddings.shape[1]:
            raise ValueError("Query embedding has an unexpected dimension")

        norm = np.linalg.norm(vector)
        if norm == 0:
            raise ValueError("Query embedding cannot be zero")

        scores = self.embeddings @ (vector / norm)
        k = min(k, len(scores))
        positions = np.argsort(-scores)[:k]

        return [
            SearchResult(
                identity=self.identities[position],
                similarity=float(scores[position]),
                image_path=self.image_paths[position],
            )
            for position in positions
        ]

