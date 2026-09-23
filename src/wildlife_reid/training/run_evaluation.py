from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from wildlife_reid.training.evaluate import evaluate_retrieval


def load_embeddings(path: Path):
    """Load query or gallery embeddings from an NPZ file."""

    if not path.exists():
        raise FileNotFoundError(
            f"Embedding file not found: {path}"
        )

    with np.load(path, allow_pickle=False) as data:
        required = {"embeddings", "identities"}

        missing = required - set(data.files)

        if missing:
            raise ValueError(
                f"Embedding file is missing: {sorted(missing)}"
            )

        embeddings = data["embeddings"]
        identities = data["identities"]

    if embeddings.ndim != 2:
        raise ValueError(
            "Embeddings must be a two-dimensional array"
        )

    if identities.ndim != 1:
        raise ValueError(
            "Identities must be a one-dimensional array"
        )

    if len(embeddings) != len(identities):
        raise ValueError(
            "Number of embeddings must match number of identities"
        )

    if len(embeddings) == 0:
        raise ValueError(
            "Embedding file cannot be empty"
        )

    return embeddings, identities


def run_evaluation(
    query_path: Path,
    gallery_path: Path,
) -> dict[str, float]:
    """Evaluate query embeddings against gallery embeddings."""

    query_embeddings, query_labels = load_embeddings(
        query_path
    )

    gallery_embeddings, gallery_labels = load_embeddings(
        gallery_path
    )

    if query_embeddings.shape[1] != gallery_embeddings.shape[1]:
        raise ValueError(
            "Query and gallery embeddings must have "
            "the same dimension"
        )

    results = evaluate_retrieval(
        query_embeddings=query_embeddings,
        gallery_embeddings=gallery_embeddings,
        query_labels=query_labels,
        gallery_labels=gallery_labels,
    )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate wildlife re-identification embeddings"
    )

    parser.add_argument(
        "--query",
        required=True,
        help="Path to query embeddings NPZ file",
    )

    parser.add_argument(
        "--gallery",
        required=True,
        help="Path to gallery embeddings NPZ file",
    )

    args = parser.parse_args()

    results = run_evaluation(
        query_path=Path(args.query),
        gallery_path=Path(args.gallery),
    )

    print()
    print("Wildlife Re-Identification Evaluation")
    print("-------------------------------------")
    print(f"Top-1 Accuracy : {results['top_1']:.4f}")
    print(f"Top-5 Accuracy : {results['top_5']:.4f}")
    print(f"mAP            : {results['mAP']:.4f}")
    print()


if __name__ == "__main__":
    main()