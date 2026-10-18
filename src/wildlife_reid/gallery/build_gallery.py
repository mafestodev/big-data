from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from wildlife_reid.common.config import Settings
from wildlife_reid.training.preprocessing import (
    create_image_processor,
    preprocess_image,
)


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

        pixel_values = preprocess_image(
            self.image_processor,
            image,
        )

        return (
            pixel_values,
            row["identity"],
            row["species"],
            row["path"],
        )


def load_metadata(metadata_path, split="train"):
    """Load image metadata, optionally retaining one official dataset split."""

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

        required_columns = {"path", "identity", "species"}

        if split is not None:
            required_columns.add("split")

        if not required_columns.issubset(fieldnames):
            missing = sorted(required_columns.difference(fieldnames))
            raise ValueError(
                "Metadata is missing required columns: "
                + ", ".join(missing)
            )

        metadata = list(reader)

    if split is not None:
        metadata = [
            row
            for row in metadata
            if row["split"] == split
        ]

    if not metadata:
        raise ValueError(
            "Metadata contains no records"
            + (f" for split='{split}'." if split is not None else ".")
        )

    return metadata


def extract_model_state_dict(checkpoint):
    """Support both complete training checkpoints and model-only artifacts."""

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Checkpoint must be a dictionary containing model weights."
        )

    if "model_state_dict" in checkpoint:
        model_state_dict = checkpoint["model_state_dict"]
    else:
        model_state_dict = checkpoint

    if not isinstance(model_state_dict, dict) or not model_state_dict:
        raise ValueError(
            "Checkpoint does not contain a valid model_state_dict."
        )

    return model_state_dict


def generate_embeddings(checkpoint_path, metadata_path, dataset_path, output_path,
    batch_size,
    metadata_split="train",
):
    """Generate embeddings from a trained checkpoint."""

    from wildlife_reid.training.model import SwinEmbeddingModel

    settings = Settings.from_environment()

    device = torch.device(settings.device)

    print("Embedding device:", device)

    metadata = load_metadata(
        metadata_path,
        split=metadata_split,
    )

    print("Metadata split:", metadata_split or "all")
    print("Images:", len(metadata))

    image_processor = create_image_processor(
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
        pin_memory=device.type == "cuda",
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

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(
        extract_model_state_dict(checkpoint)
    )

    model = model.to(device)
    model.eval()

    embeddings = []
    identities = []
    species = []
    image_paths = []

    print("Generating embeddings...")

    with torch.inference_mode():
        for batch_index, (
            images,
            batch_identities,
            batch_species,
            batch_image_paths,
        ) in enumerate(loader):
            images = images.to(
                device,
                non_blocking=True,
            )

            batch_embeddings = model(images)

            embeddings.append(
                batch_embeddings.cpu().numpy()
            )

            identities.extend(batch_identities)
            species.extend(batch_species)
            image_paths.extend(batch_image_paths)

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

    species = np.asarray(
        species,
        dtype=str,
    )

    image_paths = np.asarray(
        image_paths,
        dtype=str,
    )

    if len(embeddings) != len(identities):
        raise RuntimeError(
            "Number of embeddings does not match number of identities."
        )

    if len(embeddings) != len(image_paths):
        raise RuntimeError(
            "Number of embeddings does not match number of image paths."
        )

    if len(embeddings) != len(species):
        raise RuntimeError(
            "Number of embeddings does not match number of species labels."
        )

    if embeddings.shape[1] != settings.embedding_dimension:
        raise RuntimeError(
            f"Expected {settings.embedding_dimension}-D embeddings, "
            f"received {embeddings.shape[1]}-D embeddings."
        )

    if not np.isfinite(embeddings).all():
        raise RuntimeError(
            "Generated embeddings contain NaN or infinite values."
        )

    embedding_norms = np.linalg.norm(
        embeddings,
        axis=1,
    )

    if not np.allclose(
        embedding_norms,
        1.0,
        atol=1e-4,
    ):
        raise RuntimeError(
            "Generated embeddings are not L2-normalized."
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
        species=species,
        image_paths=image_paths,
    )

    print()
    print("Embedding generation complete.")
    print("Embedding shape:", embeddings.shape)
    print("Identity labels:", len(identities))
    print("Species labels:", len(species))
    print("Image paths:", len(image_paths))
    print("Saved to:", output_path)

    return embeddings, identities


def build_gallery(
    checkpoint_path,
    metadata_path,
    dataset_path,
    output_path,
    batch_size,
    metadata_split="train",
):
    """Build a reference gallery of embeddings."""

    print("Building gallery...")
    
    return generate_embeddings(
        checkpoint_path=checkpoint_path,
        metadata_path=metadata_path,
        dataset_path=dataset_path,
        output_path=output_path,
        batch_size=batch_size,
        metadata_split=metadata_split,
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
        help="CSV containing path, identity and species columns",
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

    parser.add_argument(
        "--split",
        default="train",
        help=(
            "Metadata split to embed (default: train). "
            "Use 'all' only with an already prepared gallery CSV."
        ),
    )

    args = parser.parse_args()

    metadata_split = None if args.split.lower() == "all" else args.split

    generate_embeddings(
        checkpoint_path=args.checkpoint,
        metadata_path=args.metadata,
        dataset_path=args.dataset_path,
        output_path=args.output,
        batch_size=args.batch_size,
        metadata_split=metadata_split,
    )


if __name__ == "__main__":
    main()
