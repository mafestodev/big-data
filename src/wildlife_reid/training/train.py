from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoImageProcessor

from wildlife_reid.common.config import Settings
from wildlife_reid.training.arcface import ArcFace
from wildlife_reid.training.dataset import create_train_loader
from wildlife_reid.training.model import SwinEmbeddingModel


PROJECT_ROOT = Path(__file__).resolve().parents[3]


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


def parse_epoch_batch(checkpoint_file):
    """
    Parse (epoch, batch) from a checkpoint filename.

    Examples:
        swin_embedding_model_epoch_3.pt
            -> (3, 0)

        swin_embedding_model_epoch_3_batch_800.pt
            -> (3, 800)
    """

    name = Path(checkpoint_file).stem

    if "_epoch_" not in name:
        return (-1, -1)

    tail = name.split("_epoch_", 1)[1]

    if "_batch_" in tail:
        epoch_str, batch_str = tail.split("_batch_", 1)

        try:
            return (int(epoch_str), int(batch_str))
        except ValueError:
            return (-1, -1)

    try:
        return (int(tail), 0)
    except ValueError:
        return (-1, -1)


def find_latest_checkpoint(checkpoint_path):
    """
    Find the most recent completed full-epoch checkpoint.

    Full-epoch checkpoints are preferred over mid-epoch
    checkpoints so that training resumes cleanly from the
    last completely finished epoch.

    Returns None if no completed epoch checkpoint exists.
    """

    checkpoint_path = Path(checkpoint_path)

    # Find only completed full-epoch checkpoints.
    # These have the format:
    # swin_embedding_model_epoch_4.pt
    #
    # Mid-epoch checkpoints such as:
    # swin_embedding_model_epoch_5_batch_1000.pt
    #
    # are intentionally ignored when a completed epoch
    # checkpoint is available.
    checkpoint_files = list(
        checkpoint_path.parent.glob(
            f"{checkpoint_path.stem}_epoch_*{checkpoint_path.suffix}"
        )
    )

    checkpoint_files = [
        checkpoint
        for checkpoint in checkpoint_files
        if parse_epoch_batch(checkpoint)[1] == 0
    ]

    if not checkpoint_files:
        return None

    checkpoint_files.sort(key=parse_epoch_batch)

    return checkpoint_files[-1]


def save_checkpoint(
    checkpoint_path,
    epoch,
    model,
    arcface,
    optimizer,
    average_loss,
    batch_index=0,
):
    """Save a complete training checkpoint."""

    checkpoint_path = Path(checkpoint_path)

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "epoch": epoch,
        "batch_index": batch_index,
        "model_state_dict": model.state_dict(),
        "arcface_state_dict": arcface.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": average_loss,
    }

    # Write to a temporary file first so a runtime interruption
    # cannot leave the real checkpoint partially written.

    temp_path = checkpoint_path.with_name(
        checkpoint_path.name + ".tmp"
    )

    with open(temp_path, "wb") as file:
        torch.save(checkpoint, file)
        file.flush()
        os.fsync(file.fileno())

    os.replace(
        temp_path,
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
    completed_batch = checkpoint.get("batch_index", 0)

    print(
        f"Checkpoint loaded. "
        f"Training completed through epoch "
        f"{completed_epoch}, batch {completed_batch}."
    )

    return completed_epoch, completed_batch


def train_model(
    train_metadata,
    dataset_path,
    identity_to_label,
    resume=False,
):
    """
    Train the Swin embedding model using ArcFace.

    Supports saving and resuming complete training
    checkpoints, including mid-epoch checkpoints.
    """

    settings = Settings.from_environment()

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

    # ── mid-epoch checkpoint configuration ──
    #
    # Save a checkpoint every N batches so that a Colab
    # disconnect never loses more than N batches of work.
    #
    # At ~1718 batches per epoch, 200 batches is roughly
    # 11-12% of an epoch, or ~25 minutes at current speed.

    SAVE_EVERY_N_BATCHES = 200

    # If True, keep every mid-epoch checkpoint forever.
    # If False, delete a completed epoch's mid-epoch
    # checkpoints after the full-epoch checkpoint is written.

    KEEP_MID_EPOCH_FILES = False

    start_epoch = 0
    start_batch = 0

    if resume:
        latest_checkpoint = find_latest_checkpoint(
            checkpoint_path
        )

        if latest_checkpoint is None:
            print(
                "Resume requested, but no completed epoch checkpoint "
                "was found."
            )

            print(
                "Starting training from the beginning."
            )

        else:
            start_epoch, start_batch = load_checkpoint(
                checkpoint_path=latest_checkpoint,
                model=model,
                arcface=arcface,
                optimizer=optimizer,
                device=device,
            )

    # Completed epoch checkpoints are 1-indexed.
    # The training loop is 0-indexed.
    #
    # Because we now resume from completed full-epoch
    # checkpoints, the next epoch is always start_epoch.

    resume_epoch_index = start_epoch

    if resume_epoch_index >= settings.num_epochs:
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
        f"{resume_epoch_index + 1}."
    )

    print(
        f"Total epochs: {settings.num_epochs}"
    )

    print()

    average_loss = 0.0

    for epoch in range(
        resume_epoch_index,
        settings.num_epochs,
    ):

        model.train()
        arcface.train()

        running_loss = 0.0
        batches_seen = 0

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
            batches_seen += 1

            if (
                batch_index + 1
            ) % 100 == 0:

                print(
                    f"Epoch [{epoch + 1}/"
                    f"{settings.num_epochs}] "
                    f"Batch [{batch_index + 1}/"
                    f"{len(train_loader)}] "
                    f"Loss: {loss.item():.4f}",
                    flush=True,
                )

            # ── mid-epoch checkpoint ──

            if (
                batch_index + 1
            ) % SAVE_EVERY_N_BATCHES == 0:

                partial_loss = (
                    running_loss
                    / max(batches_seen, 1)
                )

                mid_checkpoint_path = (
                    checkpoint_path.with_name(
                        f"{checkpoint_path.stem}"
                        f"_epoch_{epoch + 1}"
                        f"_batch_{batch_index + 1}"
                        f"{checkpoint_path.suffix}"
                    )
                )

                save_checkpoint(
                    checkpoint_path=mid_checkpoint_path,
                    epoch=epoch + 1,
                    batch_index=batch_index + 1,
                    model=model,
                    arcface=arcface,
                    optimizer=optimizer,
                    average_loss=partial_loss,
                )

                print(
                    "Mid-epoch checkpoint saved:",
                    mid_checkpoint_path.name,
                    flush=True,
                )

        average_loss = (
            running_loss
            / max(batches_seen, 1)
        )

        print()

        print(
            f"Epoch {epoch + 1} complete. "
            f"Average loss: {average_loss:.4f}",
            flush=True,
        )

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
            batch_index=0,
            model=model,
            arcface=arcface,
            optimizer=optimizer,
            average_loss=average_loss,
        )

        print(
            "Epoch checkpoint saved to:",
            epoch_checkpoint_path,
            flush=True,
        )

        # ── optional cleanup of mid-epoch files ──

        if not KEEP_MID_EPOCH_FILES:

            for old_checkpoint in (
                checkpoint_path.parent.glob(
                    f"{checkpoint_path.stem}"
                    f"_epoch_{epoch + 1}"
                    f"_batch_*"
                    f"{checkpoint_path.suffix}"
                )
            ):

                try:
                    old_checkpoint.unlink()
                except OSError:
                    pass

        print()

    save_checkpoint(
        checkpoint_path=checkpoint_path,
        epoch=settings.num_epochs,
        batch_index=0,
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
            "completed epoch checkpoint."
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