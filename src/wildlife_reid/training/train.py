from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the Swin-ArcFace model")
    parser.add_argument("--metadata", required=True, help="Prepared training metadata file")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(
        "Training command scaffold is ready. Next implement metadata parsing, "
        f"identity-balanced sampling and the training loop for {args.metadata}."
    )


if __name__ == "__main__":
    main()

