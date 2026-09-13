from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the reference embedding gallery")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output", default="gallery/embeddings.npz")
    args = parser.parse_args()
    raise SystemExit(
        "Gallery builder scaffold is ready. Next load the selected checkpoint, "
        f"embed gallery records from {args.metadata}, and write {args.output}."
    )


if __name__ == "__main__":
    main()

