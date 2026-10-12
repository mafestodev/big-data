
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from wildlife_reid.training.evaluate import evaluate_retrieval


def load_embeddings(path: Path):
    """Load query or gallery embeddings from an NPZ file."""

    if not path.exists():
        raise FileNotFoundError(f"Embedding file not found: {path}")

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
        raise ValueError("Embeddings must be a two-dimensional array")

    if identities.ndim != 1:
        raise ValueError("Identities must be a one-dimensional array")

    if len(embeddings) != len(identities):
        raise ValueError(
            "Number of embeddings must match number of identities"
        )

    if len(embeddings) == 0:
        raise ValueError("Embedding file cannot be empty")

    return embeddings, identities


def run_evaluation(
    query_path: Path,
    gallery_path: Path,
    batch_size: int = 16,
) -> dict[str, float]:
    """Evaluate query embeddings against gallery embeddings."""

    query_embeddings, query_labels = load_embeddings(query_path)
    gallery_embeddings, gallery_labels = load_embeddings(gallery_path)

    if query_embeddings.shape[1] != gallery_embeddings.shape[1]:
        raise ValueError(
            "Query and gallery embeddings must have the same dimension"
        )

    return evaluate_retrieval(
        query_embeddings=query_embeddings,
        gallery_embeddings=gallery_embeddings,
        query_labels=query_labels,
        gallery_labels=gallery_labels,
        batch_size=batch_size,
    )


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

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Number of queries processed per batch (default: 16)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to save the evaluation report as text",
    )

    args = parser.parse_args()

    results = run_evaluation(
        query_path=Path(args.query),
        gallery_path=Path(args.gallery),
        batch_size=args.batch_size,
    )

    report = "\n".join([
        "WILDLIFE RE-IDENTIFICATION EVALUATION",
        "=" * 40,
        f"Top-1 accuracy: {results['top_1'] * 100:.2f}%",
        f"Top-5 accuracy: {results['top_5'] * 100:.2f}%",
        f"Precision@5: {results['precision_at_5'] * 100:.2f}%",
        f"Mean Average Precision (mAP): {results['mAP'] * 100:.2f}%",
        "",
        f"Known-identity queries: {int(results['known_queries']):,}",
        f"Unseen-identity queries: {int(results['unseen_queries']):,}",
        f"Identity coverage: {results['identity_coverage'] * 100:.2f}%",
        "",
        "Note: closed-set metrics exclude queries whose identities",
        "are not represented in the gallery.",
        "Open-set rejection performance is not measured.",
    ])

    print("\n" + report)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")
        print(f"\nReport saved to: {args.output}")


if __name__ == "__main__":
    main()
