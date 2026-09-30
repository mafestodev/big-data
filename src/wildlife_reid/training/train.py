from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoImageProcessor

from wildlife_reid.common.config import Settings
from wildlife_reid.training.arcface import ArcFace
from wildlife_reid.training.dataset import create_train_loader
from wildlife_reid.training.model import SwinEmbeddingModel


def load_dataset_metadata(dataset_path):
    """Load the WildlifeReID metadata.csv file."""

    dataset_path = Path(dataset_path)

    metadata_path = dataset_path / "metadata.csv"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Dataset metadata not found: {metadata_path}"
        )

    with open(
        metadata_path,
        "r",
        newline="",
        encoding="utf-8",
    ) as file:

        reader = csv.DictReader(file)

        required_columns = {
            "path",
            "identity",
            "split",
        }

        if not required_columns.issubset(
            reader.fieldnames or []
        ):
            raise ValueError(
                "metadata.csv must contain "
                "'path', 'identity', and 'split' columns."
            )

        metadata = list(reader)

    if not metadata:
        raise ValueError(
            "metadata.csv contains no records."
        )

    return metadata


def create_training_metadata(metadata):
    """Select the official training split."""

    train_metadata = [
        row
        for row in metadata
        if row["split"] == "train"
    ]

    if not train_metadata:
        raise ValueError(
            "No records with split='train' were found."
        )

    return train_metadata


def create_identity_mapping(train_metadata):
    """Create a stable identity-to-label mapping."""

    identities = sorted(
        {
            row["identity"]
            for row in train_metadata
        }
    )

    return {
        identity: label
        for label, identity in enumerate(identities)
    }


def find_latest_checkpoint(checkpoint_path):
    """
    Find the most recent epoch checkpoint.

    Returns None if no epoch checkpoint exists.
    """

    checkpoint_path = Path(checkpoint_path)

    checkpoint_files = list(
        checkpoint_path.parent.glob(
            f"{checkpoint_path.stem}_epoch_*{checkpoint_path.suffix}"
        )
    )

    if not checkpoint_files:
        return None

    def get_epoch_number(path):
        try:
            return int(
                path.stem.rsplit("_epoch_", 1)[1]
            )
        except (IndexError, ValueError):
            return -1

    checkpoint_files.sort(
        key=get_epoch_number
    )

    return checkpoint_files[-1]


def save_checkpoint(
    checkpoint_path,
    epoch,
    model,
    arcface,
    optimizer,
    average_loss,
):
    """Save a complete training checkpoint."""

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "arcface_state_dict": arcface.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": average_loss,
    }

    torch.save(
        checkpoint,
        checkpoint_path,
    )


def load_checkpoint(
    checkpoint_path,
    model,
    arcface,
    optimizer,
    device,
):
    """Load a complete training checkpoint."""

    print(
        "Loading checkpoint:",
        checkpoint_path,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    arcface.load_state_dict(
        checkpoint["arcface_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    completed_epoch = checkpoint["epoch"]

    print(
        f"Checkpoint loaded. "
        f"Training completed through epoch "
        f"{completed_epoch}."
    )

    return completed_epoch


def train_model(
    train_metadata,
    dataset_path,
    identity_to_label,
    resume=False,
):
    """
    Train the Swin embedding model using ArcFace.

    Supports saving and resuming complete training
    checkpoints.
    """

    settings = Settings()

    device = torch.device(settings.device)

    print("Training device:", device)
    print(
        "Training identities:",
        len(identity_to_label),
    )

    image_processor = AutoImageProcessor.from_pretrained(
        settings.model_name
    )

    train_dataset, train_loader = create_train_loader(
        train_metadata=train_metadata,
        dataset_path=dataset_path,
        image_processor=image_processor,
        identity_to_label=identity_to_label,
        batch_size=settings.batch_size,
        num_workers=settings.num_workers,
    )

    print(
        "Training images:",
        len(train_dataset),
    )

    print(
        "Training batches:",
        len(train_loader),
    )

    model = SwinEmbeddingModel(
        model_name=settings.model_name,
        embedding_dimension=settings.embedding_dimension,
        pretrained=True,
    )

    model = model.to(device)

    sample_images, sample_labels = next(
        iter(train_loader)
    )

    sample_images = sample_images.to(device)

    model.eval()

    with torch.no_grad():
        sample_embeddings = model(sample_images)

    print(
        "Embedding shape:",
        sample_embeddings.shape,
    )

    expected_shape = (
        sample_images.shape[0],
        settings.embedding_dimension,
    )

    if sample_embeddings.shape != expected_shape:
        raise RuntimeError(
            "Unexpected embedding dimensions. "
            f"Expected {expected_shape}, "
            f"received {tuple(sample_embeddings.shape)}."
        )

    num_identities = len(identity_to_label)

    arcface = ArcFace(
        embedding_dimension=settings.embedding_dimension,
        num_classes=num_identities,
        scale=settings.arcface_scale,
        margin=settings.arcface_margin,
    )

    arcface = arcface.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        list(model.parameters())
        + list(arcface.parameters()),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )

    checkpoint_path = Path(
        settings.checkpoint_path
    )

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    start_epoch = 0

    # ---------------------------------------------------------
    # RESUME FROM LATEST CHECKPOINT
    # ---------------------------------------------------------

    if resume:

        latest_checkpoint = find_latest_checkpoint(
            checkpoint_path
        )

        if latest_checkpoint is None:

            print(
                "Resume requested, but no epoch checkpoint "
                "was found."
            )

            print(
                "Starting training from the beginning."
            )

        else:

            start_epoch = load_checkpoint(
                checkpoint_path=latest_checkpoint,
                model=model,
                arcface=arcface,
                optimizer=optimizer,
                device=device,
            )

    # ---------------------------------------------------------
    # CHECK WHETHER TRAINING IS ALREADY COMPLETE
    # ---------------------------------------------------------

    if start_epoch >= settings.num_epochs:

        print()
        print(
            "Training is already complete."
        )

        print(
            f"Completed epochs: {start_epoch}"
        )

        return model

    print()
    print(
        f"Starting training from epoch "
        f"{start_epoch + 1}."
    )

    print(
        f"Total epochs: {settings.num_epochs}"
    )

    print()

    # ---------------------------------------------------------
    # TRAINING LOOP
    # ---------------------------------------------------------

    for epoch in range(
        start_epoch,
        settings.num_epochs,
    ):

        model.train()
        arcface.train()

        running_loss = 0.0

        for batch_index, (
            images,
            labels,
        ) in enumerate(train_loader):

            images = images.to(
                device,
                non_blocking=True,
            )

            labels = labels.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad()

            embeddings = model(images)

            logits = arcface(
                embeddings,
                labels,
            )

            loss = criterion(
                logits,
                labels,
            )

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            if (
                batch_index + 1
            ) % 100 == 0:

                print(
                    f"Epoch [{epoch + 1}/"
                    f"{settings.num_epochs}] "
                    f"Batch [{batch_index + 1}/"
                    f"{len(train_loader)}] "
                    f"Loss: {loss.item():.4f}"
                )

        average_loss = (
            running_loss
            / len(train_loader)
        )

        print()
        print(
            f"Epoch {epoch + 1} complete. "
            f"Average loss: {average_loss:.4f}"
        )

        # -----------------------------------------------------
        # SAVE COMPLETE EPOCH CHECKPOINT
        # -----------------------------------------------------

        epoch_checkpoint_path = (
            checkpoint_path.with_name(
                f"{checkpoint_path.stem}"
                f"_epoch_{epoch + 1}"
                f"{checkpoint_path.suffix}"
            )
        )

        save_checkpoint(
            checkpoint_path=epoch_checkpoint_path,
            epoch=epoch + 1,
            model=model,
            arcface=arcface,
            optimizer=optimizer,
            average_loss=average_loss,
        )

        print(
            "Epoch checkpoint saved to:",
            epoch_checkpoint_path,
        )

        print()

    # ---------------------------------------------------------
    # SAVE FINAL CHECKPOINT
    # ---------------------------------------------------------

    save_checkpoint(
        checkpoint_path=checkpoint_path,
        epoch=settings.num_epochs,
        model=model,
        arcface=arcface,
        optimizer=optimizer,
        average_loss=average_loss,
    )

    print()
    print(
        "Training complete."
    )

    print(
        "Final checkpoint saved to:",
        checkpoint_path,
    )

    return model


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Train the wildlife "
            "re-identification model"
        )
    )

    parser.add_argument(
        "--dataset-path",
        required=True,
        help=(
            "Root directory of WildlifeReID-10k "
            "containing metadata.csv"
        ),
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume training from the latest "
            "epoch checkpoint."
        ),
    )

    args = parser.parse_args()

    dataset_path = Path(
        args.dataset_path
    )

    metadata = load_dataset_metadata(
        dataset_path
    )

    train_metadata = create_training_metadata(
        metadata
    )

    identity_to_label = create_identity_mapping(
        train_metadata
    )

    print(
        "Training images:",
        len(train_metadata),
    )

    print(
        "Training identities:",
        len(identity_to_label),
    )

    train_model(
        train_metadata=train_metadata,
        dataset_path=dataset_path,
        identity_to_label=identity_to_label,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()