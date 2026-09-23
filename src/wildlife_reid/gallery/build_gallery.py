from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor

from wildlife_reid.common.config import Settings
from wildlife_reid.training.model import SwinEmbeddingModel


class EmbeddingDataset(Dataset):
    """Dataset used to convert metadata records into model embeddings."""

    def __init__(
        self,
        metadata,
        dataset_path,
        image_processor,
    ):
        self.metadata = metadata
        self.dataset_path = Path(dataset_path)
        self.image_processor = image_processor

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        row = self.metadata[index]

        image_path = self.dataset_path / row["path"]

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        image = Image.open(image_path).convert("RGB")

        processed = self.image_processor(
            images=image,
            return_tensors="pt",
        )

        pixel_values = processed["pixel_values"].squeeze(0)

        return pixel_values, row["identity"]


def load_metadata(metadata_path):
    """Load metadata containing image paths and identity labels."""

    metadata_path = Path(metadata_path)

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_path}"
        )

    with open(
        metadata_path,
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames or []

        required_columns = {"path", "identity"}

        if not required_columns.issubset(fieldnames):
            raise ValueError(
                "Metadata must contain 'path' and 'identity' columns."
            )

        metadata = list(reader)

    if not metadata:
        raise ValueError(
            "Metadata file contains no records."
        )

    return metadata


def generate_embeddings(
    checkpoint_path,
    metadata_path,
    dataset_path,
    output_path,
    batch_size,
):
    """Generate embeddings from a trained checkpoint."""

    settings = Settings()

    device = torch.device(settings.device)

    print("Embedding device:", device)

    metadata = load_metadata(metadata_path)

    print("Images:", len(metadata))

    image_processor = AutoImageProcessor.from_pretrained(
        settings.model_name
    )

    dataset = EmbeddingDataset(
        metadata=metadata,
        dataset_path=dataset_path,
        image_processor=image_processor,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=settings.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = SwinEmbeddingModel(
        model_name=settings.model_name,
        embedding_dimension=settings.embedding_dimension,
        pretrained=False,
    )

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    print("Loading checkpoint:", checkpoint_path)

    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
    )

    model = model.to(device)
    model.eval()

    embeddings = []
    identities = []

    print("Generating embeddings...")

    with torch.no_grad():
        for batch_index, (images, batch_identities) in enumerate(
            loader
        ):
            images = images.to(
                device,
                non_blocking=True,
            )

            batch_embeddings = model(images)

            embeddings.append(
                batch_embeddings.cpu().numpy()
            )

            identities.extend(batch_identities)

            if (batch_index + 1) % 100 == 0:
                print(
                    f"Batch [{batch_index + 1}/{len(loader)}]"
                )

    embeddings = np.concatenate(
        embeddings,
        axis=0,
    ).astype(np.float32)

    identities = np.asarray(
        identities,
        dtype=str,
    )

    if len(embeddings) != len(identities):
        raise RuntimeError(
            "Number of embeddings does not match number of identities."
        )

    if embeddings.shape[1] != settings.embedding_dimension:
        raise RuntimeError(
            f"Expected {settings.embedding_dimension}-D embeddings, "
            f"received {embeddings.shape[1]}-D embeddings."
        )

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        output_path,
        embeddings=embeddings,
        identities=identities,
    )

    print()
    print("Embedding generation complete.")
    print("Embedding shape:", embeddings.shape)
    print("Identity labels:", len(identities))
    print("Saved to:", output_path)

    return embeddings, identities


def build_gallery(
    checkpoint_path,
    metadata_path,
    dataset_path,
    output_path,
    batch_size,
):
    """Build a reference gallery of embeddings."""

    print("Building gallery...")
    
    return generate_embeddings(
        checkpoint_path=checkpoint_path,
        metadata_path=metadata_path,
        dataset_path=dataset_path,
        output_path=output_path,
        batch_size=batch_size,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate wildlife re-identification embeddings"
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to trained model checkpoint",
    )

    parser.add_argument(
        "--metadata",
        required=True,
        help="CSV containing path and identity columns",
    )

    parser.add_argument(
        "--dataset-path",
        required=True,
        help="Root directory containing the dataset images",
    )

    parser.add_argument(
        "--output",
        default="gallery/embeddings.npz",
        help="Output NPZ file",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Embedding batch size",
    )

    args = parser.parse_args()

    generate_embeddings(
        checkpoint_path=args.checkpoint,
        metadata_path=args.metadata,
        dataset_path=args.dataset_path,
        output_path=args.output,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()