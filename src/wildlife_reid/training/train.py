from __future__ import annotations                     # Enable postponed evaluation of type hints.

import argparse                                         # Command-line argument parsing.
import csv                                              # Reading the metadata CSV file.
import os                                               # Environment variables and file operations.
from pathlib import Path                                # Object-oriented filesystem paths.

import torch                                            # PyTorch core.
import torch.nn as nn                                   # Neural network layers and losses.
from transformers import AutoImageProcessor             # HuggingFace image preprocessing.

from wildlife_reid.common.config import Settings        # Project settings loader.
from wildlife_reid.training.arcface import ArcFace      # ArcFace classification head.
from wildlife_reid.training.dataset import create_train_loader   # Training dataloader factory.
from wildlife_reid.training.model import SwinEmbeddingModel      # Swin embedding backbone.


def load_dataset_metadata(dataset_path):                # Load the WildlifeReID metadata.csv file.
    """Load the WildlifeReID metadata.csv file."""      # Docstring.

    dataset_path = Path(dataset_path)                   # Normalize to a Path object.

    metadata_path = dataset_path / "metadata.csv"       # Build the metadata file path.

    if not metadata_path.exists():                      # Fail early if metadata is missing.
        raise FileNotFoundError(                        # Raise a clear error.
            f"Dataset metadata not found: {metadata_path}"   # Error message with path.
        )

    with open(                                          # Open the metadata file.
        metadata_path,                                  # Path to open.
        "r",                                            # Read mode.
        newline="",                                     # Let csv module handle newlines.
        encoding="utf-8",                               # UTF-8 encoding.
    ) as file:                                          # Context manager ensures close.

        reader = csv.DictReader(file)                   # Parse rows as dicts.

        required_columns = {                            # Columns we require.
            "path",                                     # Image path column.
            "identity",                                 # Identity label column.
            "split",                                    # Train/val/test split column.
        }

        if not required_columns.issubset(               # Check all required columns exist.
            reader.fieldnames or []                     # Guard against None fieldnames.
        ):
            raise ValueError(                           # Raise if schema is wrong.
                "metadata.csv must contain "            # Error message part 1.
                "'path', 'identity', and 'split' columns."   # Error message part 2.
            )

        metadata = list(reader)                         # Materialize all rows.

    if not metadata:                                    # Fail if the file was empty.
        raise ValueError(                               # Raise a clear error.
            "metadata.csv contains no records."         # Error message.
        )

    return metadata                                     # Return the list of rows.


def create_training_metadata(metadata):                 # Select the official training split.
    """Select the official training split."""           # Docstring.

    train_metadata = [                                  # Build the training subset.
        row                                             # Keep each row...
        for row in metadata                             # ...from the full metadata...
        if row["split"] == "train"                      # ...whose split is "train".
    ]

    if not train_metadata:                              # Fail if no training rows found.
        raise ValueError(                               # Raise a clear error.
            "No records with split='train' were found." # Error message.
        )

    return train_metadata                               # Return the training rows.


def create_identity_mapping(train_metadata):            # Create a stable identity-to-label mapping.
    """Create a stable identity-to-label mapping."""    # Docstring.

    identities = sorted(                                # Sort identities for determinism.
        {                                               # Build a set...
            row["identity"]                             # ...of unique identity strings...
            for row in train_metadata                   # ...from training rows.
        }
    )

    return {                                            # Return the mapping dict.
        identity: label                                 # identity string -> integer label.
        for label, identity in enumerate(identities)    # Enumerate in sorted order.
    }


def parse_epoch_batch(checkpoint_file):                 # Parse (epoch, batch) from a filename.
    """
    Parse (epoch, batch) from a checkpoint filename.     # Docstring.

    Examples:                                           # Example filenames.
        swin_embedding_model_epoch_3.pt            -> (3, 0)     # Full-epoch checkpoint.
        swin_embedding_model_epoch_3_batch_800.pt  -> (3, 800)   # Mid-epoch checkpoint.
    """

    name = Path(checkpoint_file).stem                   # Strip directory and .pt suffix.

    if "_epoch_" not in name:                           # Bail out if pattern is absent.
        return (-1, -1)                                 # Sort key that always loses.

    tail = name.split("_epoch_", 1)[1]                  # Extract everything after "_epoch_".

    if "_batch_" in tail:                               # Check for mid-epoch form.
        epoch_str, batch_str = tail.split("_batch_", 1) # Split epoch and batch parts.
        try:                                            # Attempt conversion.
            return (int(epoch_str), int(batch_str))     # Return parsed tuple.
        except ValueError:                              # Handle malformed numbers.
            return (-1, -1)                             # Sort key that always loses.

    try:                                                # Full-epoch form.
        return (int(tail), 0)                           # Batch index is 0.
    except ValueError:                                  # Handle malformed numbers.
        return (-1, -1)                                 # Sort key that always loses.


def find_latest_checkpoint(checkpoint_path):            # Find the latest checkpoint file.
    """
    Find the most recent epoch or mid-epoch checkpoint. # Docstring.

    Returns None if no checkpoint exists.               # Return contract.
    """

    checkpoint_path = Path(checkpoint_path)             # Normalize to a Path object.

    checkpoint_files = list(                            # List matching checkpoint files.
        checkpoint_path.parent.glob(                    # Search the parent directory.
            f"{checkpoint_path.stem}_epoch_*{checkpoint_path.suffix}"   # Match epoch files.
        )
    )

    if not checkpoint_files:                            # No checkpoints found.
        return None                                     # Signal "nothing to resume".

    checkpoint_files.sort(key=parse_epoch_batch)        # Sort by (epoch, batch).

    return checkpoint_files[-1]                         # Return the newest one.


def save_checkpoint(                                    # Save a training checkpoint.
    checkpoint_path,                                    # Where to write it.
    epoch,                                              # Completed epoch number.
    model,                                              # Model to save.
    arcface,                                            # ArcFace head to save.
    optimizer,                                          # Optimizer state to save.
    average_loss,                                       # Loss value for logging.
    batch_index=0,                                      # Completed batch within epoch.
):
    """Save a complete training checkpoint safely."""   # Docstring.

    checkpoint_path = Path(checkpoint_path)             # Normalize to a Path object.

    checkpoint_path.parent.mkdir(                       # Ensure the parent directory...
        parents=True,                                   # ...creating intermediate dirs...
        exist_ok=True,                                  # ...without failing if it exists.
    )

    checkpoint = {                                      # Build the checkpoint dict.
        "epoch": epoch,                                 # Completed epoch.
        "batch_index": batch_index,                     # Completed batch within epoch.
        "model_state_dict": model.state_dict(),         # Model weights.
        "arcface_state_dict": arcface.state_dict(),     # ArcFace weights.
        "optimizer_state_dict": optimizer.state_dict(), # Optimizer state.
        "loss": average_loss,                           # Loss for logging.
    }

    # Write to a temporary file first so a runtime interruption
    # cannot leave the real checkpoint partially written.
    temp_path = checkpoint_path.with_name(              # Build the temp file path.
        checkpoint_path.name + ".tmp"                   # Append .tmp to the filename.
    )

    with open(temp_path, "wb") as file:                 # Open temp file for binary write.
        torch.save(checkpoint, file)                    # Serialize the checkpoint.
        file.flush()                                    # Flush Python buffers.
        os.fsync(file.fileno())                         # Force OS-level write to disk.

    os.replace(                                         # Atomically move temp -> real file.
        temp_path,                                      # Source temp path.
        checkpoint_path,                                # Destination real path.
    )


def load_checkpoint(                                    # Load a training checkpoint.
    checkpoint_path,                                    # File to load.
    model,                                              # Model to populate.
    arcface,                                            # ArcFace head to populate.
    optimizer,                                          # Optimizer to populate.
    device,                                             # Device to map tensors to.
):
    """Load a complete training checkpoint."""          # Docstring.

    print(                                              # Announce the load.
        "Loading checkpoint:",                          # Message prefix.
        checkpoint_path,                                # File being loaded.
    )

    checkpoint = torch.load(                            # Deserialize the checkpoint.
        checkpoint_path,                                # File to load.
        map_location=device,                            # Place tensors on the right device.
        weights_only=False,                             # Allow non-tensor objects.
    )

    model.load_state_dict(                              # Restore model weights.
        checkpoint["model_state_dict"]                  # Stored model state.
    )

    arcface.load_state_dict(                            # Restore ArcFace weights.
        checkpoint["arcface_state_dict"]                # Stored ArcFace state.
    )

    optimizer.load_state_dict(                          # Restore optimizer state.
        checkpoint["optimizer_state_dict"]              # Stored optimizer state.
    )

    completed_epoch = checkpoint["epoch"]               # Read the completed epoch.
    completed_batch = checkpoint.get("batch_index", 0)  # Read batch, defaulting to 0.

    print(                                              # Report what was loaded.
        f"Checkpoint loaded. "                          # Message part 1.
        f"Training completed through epoch "            # Message part 2.
        f"{completed_epoch}, batch {completed_batch}."  # Message part 3.
    )

    return completed_epoch, completed_batch             # Return both for the resume logic.


def train_model(                                        # Main training entry point.
    train_metadata,                                     # Training rows.
    dataset_path,                                       # Dataset root directory.
    identity_to_label,                                  # Identity-to-label mapping.
    resume=False,                                       # Whether to resume from checkpoint.
):
    """
    Train the Swin embedding model using ArcFace.       # Docstring.

    Supports saving and resuming complete training      # Docstring.
    checkpoints, including mid-epoch checkpoints.       # Docstring.
    """

    settings = Settings.from_environment()              # Load settings from env vars.

    device = torch.device(settings.device)              # Resolve the compute device.

    print("Training device:", device)                   # Report the device.
    print(                                              # Report identity count.
        "Training identities:",                         # Message prefix.
        len(identity_to_label),                         # Number of unique identities.
    )

    image_processor = AutoImageProcessor.from_pretrained(   # Load the image processor.
        settings.model_name                             # Backbone name from settings.
    )

    train_dataset, train_loader = create_train_loader(  # Build dataset and dataloader.
        train_metadata=train_metadata,                  # Training rows.
        dataset_path=dataset_path,                      # Dataset root.
        image_processor=image_processor,                # Image preprocessing.
        identity_to_label=identity_to_label,            # Label mapping.
        batch_size=settings.batch_size,                 # Batch size.
        num_workers=settings.num_workers,               # Dataloader worker count.
    )

    print(                                              # Report dataset size.
        "Training images:",                             # Message prefix.
        len(train_dataset),                             # Number of training images.
    )

    print(                                              # Report batch count.
        "Training batches:",                            # Message prefix.
        len(train_loader),                              # Number of batches per epoch.
    )

    model = SwinEmbeddingModel(                         # Build the Swin embedding model.
        model_name=settings.model_name,                 # Backbone name.
        embedding_dimension=settings.embedding_dimension,   # Output embedding size.
        pretrained=True,                                # Start from pretrained weights.
    )

    model = model.to(device)                            # Move model to the device.

    sample_images, sample_labels = next(                # Grab one batch for a sanity check.
        iter(train_loader)                              # Iterate the dataloader once.
    )

    sample_images = sample_images.to(device)            # Move sample images to device.

    model.eval()                                        # Set model to eval for the check.

    with torch.no_grad():                               # Disable grad for the check.
        sample_embeddings = model(sample_images)        # Compute sample embeddings.

    print(                                              # Report embedding shape.
        "Embedding shape:",                             # Message prefix.
        sample_embeddings.shape,                        # Actual shape.
    )

    expected_shape = (                                  # Compute expected shape.
        sample_images.shape[0],                         # Batch size.
        settings.embedding_dimension,                   # Embedding dimension.
    )

    if sample_embeddings.shape != expected_shape:       # Validate embedding shape.
        raise RuntimeError(                             # Raise if mismatch.
            "Unexpected embedding dimensions. "         # Message part 1.
            f"Expected {expected_shape}, "              # Message part 2.
            f"received {tuple(sample_embeddings.shape)}."   # Message part 3.
        )

    num_identities = len(identity_to_label)             # Number of output classes.

    arcface = ArcFace(                                  # Build the ArcFace head.
        embedding_dimension=settings.embedding_dimension,   # Input embedding size.
        num_classes=num_identities,                     # Number of identities.
        scale=settings.arcface_scale,                   # ArcFace scale.
        margin=settings.arcface_margin,                 # ArcFace margin.
    )

    arcface = arcface.to(device)                        # Move ArcFace to the device.

    criterion = nn.CrossEntropyLoss()                   # Loss function.

    optimizer = torch.optim.AdamW(                      # AdamW optimizer.
        list(model.parameters())                        # Model parameters.
        + list(arcface.parameters()),                   # ArcFace parameters.
        lr=settings.learning_rate,                      # Learning rate.
        weight_decay=settings.weight_decay,             # Weight decay.
    )

    checkpoint_path = Path(                             # Resolve the base checkpoint path.
        settings.checkpoint_path                         # From settings.
    )

    checkpoint_path.parent.mkdir(                       # Ensure checkpoint dir exists.
        parents=True,                                   # Create intermediate dirs.
        exist_ok=True,                                  # Do not fail if it exists.
    )

    # ── mid-epoch checkpoint configuration ──
    #
    # Save a checkpoint every N batches so that a Colab
    # disconnect never loses more than N batches of work.
    #
    # At ~1718 batches per epoch, 200 batches is roughly
    # 11-12% of an epoch, or ~25 minutes at current speed.

    SAVE_EVERY_N_BATCHES = 200                          # Mid-epoch save interval.

    # If True, keep every mid-epoch checkpoint forever.
    # If False, delete a completed epoch's mid-epoch
    # checkpoints after the full-epoch checkpoint is written.
    KEEP_MID_EPOCH_FILES = False                        # Cleanup flag.

    start_epoch = 0                                     # Epoch to start from.
    start_batch = 0                                     # Batch to start from within that epoch.

    if resume:                                          # Only attempt resume if requested.

        latest_checkpoint = find_latest_checkpoint(     # Locate the newest checkpoint.
            checkpoint_path                             # Base checkpoint path.
        )

        if latest_checkpoint is None:                   # No checkpoint found.

            print(                                      # Inform the user.
                "Resume requested, but no epoch checkpoint "   # Message part 1.
                "was found."                            # Message part 2.
            )

            print(                                      # Inform the user.
                "Starting training from the beginning." # Message.
            )

        else:                                           # Checkpoint found.

            start_epoch, start_batch = load_checkpoint( # Load it.
                checkpoint_path=latest_checkpoint,      # File to load.
                model=model,                            # Model to populate.
                arcface=arcface,                        # ArcFace to populate.
                optimizer=optimizer,                    # Optimizer to populate.
                device=device,                          # Device for tensors.
            )

    # Checkpoint epochs are 1-indexed (they store epoch + 1).
    # The training loop is 0-indexed. If we loaded a mid-epoch
    # checkpoint, the loop should re-enter the SAME epoch index
    # and skip the first `start_batch` batches. If we loaded a
    # full-epoch checkpoint (batch_index == 0), the loop moves
    # to the next epoch.
    if start_batch > 0:                                 # Mid-epoch checkpoint case.
        resume_epoch_index = start_epoch - 1            # Re-enter the same epoch.
    else:                                               # Full-epoch checkpoint case.
        resume_epoch_index = start_epoch                # Move to the next epoch.

    if resume_epoch_index >= settings.num_epochs:       # Training already finished.

        print()                                         # Blank line.
        print(                                          # Report completion.
            "Training is already complete."             # Message.
        )

        print(                                          # Report completed epochs.
            f"Completed epochs: {start_epoch}"          # Message.
        )

        return model                                    # Nothing left to do.

    print()                                             # Blank line.
    print(                                              # Report starting epoch.
        f"Starting training from epoch "                # Message part 1.
        f"{resume_epoch_index + 1}."                    # Message part 2.
    )

    print(                                              # Report total epochs.
        f"Total epochs: {settings.num_epochs}"          # Message.
    )

    if start_batch > 0:                                 # Report batch resume.
        print(                                          # Message.
            f"Resuming at batch {start_batch + 1} "     # Message part 1.
            f"of that epoch."                           # Message part 2.
        )

    print()                                             # Blank line.

    average_loss = 0.0                                  # Placeholder for final loss.

    for epoch in range(                                 # Iterate epochs from the resume point.
        resume_epoch_index,                             # Start index.
        settings.num_epochs,                            # End (exclusive).
    ):

        model.train()                                   # Set model to training mode.
        arcface.train()                                 # Set ArcFace to training mode.

        running_loss = 0.0                              # Accumulator for batch losses.
        batches_seen = 0                                # Counter of batches processed.

        # On the very first epoch after resume, skip the
        # batches that were already completed in the
        # mid-epoch checkpoint we loaded.
        skip_batches = (                                # Batches to skip this epoch.
            start_batch                                 # Skip count from checkpoint...
            if epoch == resume_epoch_index              # ...only on the resumed epoch...
            else 0                                      # ...otherwise skip nothing.
        )

        for batch_index, (                              # Iterate batches in this epoch.
            images,                                     # Image tensor.
            labels,                                     # Label tensor.
        ) in enumerate(train_loader):

            if batch_index < skip_batches:              # Skip already-completed batches.
                continue                                # Move on to the next batch.

            images = images.to(                         # Move images to device.
                device,                                 # Target device.
                non_blocking=True,                      # Async host->device copy.
            )

            labels = labels.to(                         # Move labels to device.
                device,                                 # Target device.
                non_blocking=True,                      # Async host->device copy.
            )

            optimizer.zero_grad()                       # Clear gradients.

            embeddings = model(images)                  # Forward pass through the backbone.

            logits = arcface(                           # Forward pass through ArcFace.
                embeddings,                             # Embeddings.
                labels,                                 # Labels for margin application.
            )

            loss = criterion(                           # Compute the loss.
                logits,                                 # Predicted logits.
                labels,                                 # Ground-truth labels.
            )

            loss.backward()                             # Backpropagate gradients.

            optimizer.step()                            # Update parameters.

            running_loss += loss.item()                 # Accumulate scalar loss.
            batches_seen += 1                           # Increment batch counter.

            if (                                        # Periodic log message.
                batch_index + 1                         # Number of batches processed.
            ) % 100 == 0:                               # Every 100 batches.

                print(                                  # Print progress.
                    f"Epoch [{epoch + 1}/"              # Epoch part.
                    f"{settings.num_epochs}] "          # Total epochs.
                    f"Batch [{batch_index + 1}/"        # Batch part.
                    f"{len(train_loader)}] "            # Total batches.
                    f"Loss: {loss.item():.4f}"          # Current loss.
                )

            # ── mid-epoch checkpoint ──
            if (                                        # Check save interval.
                batch_index + 1                         # Batches processed.
            ) % SAVE_EVERY_N_BATCHES == 0:              # Every N batches.

                partial_loss = (                        # Average loss so far this epoch.
                    running_loss                        # Total loss.
                    / max(batches_seen, 1)              # Divide by batches seen.
                )

                mid_checkpoint_path = (                 # Build the mid-epoch filename.
                    checkpoint_path.with_name(          # Based on the base checkpoint path.
                        f"{checkpoint_path.stem}"       # Base stem.
                        f"_epoch_{epoch + 1}"           # Epoch suffix.
                        f"_batch_{batch_index + 1}"     # Batch suffix.
                        f"{checkpoint_path.suffix}"     # Extension.
                    )
                )

                save_checkpoint(                        # Write the mid-epoch checkpoint.
                    checkpoint_path=mid_checkpoint_path,   # Destination path.
                    epoch=epoch + 1,                    # 1-indexed epoch.
                    batch_index=batch_index + 1,        # 1-indexed batch.
                    model=model,                        # Model to save.
                    arcface=arcface,                    # ArcFace to save.
                    optimizer=optimizer,                # Optimizer to save.
                    average_loss=partial_loss,          # Loss for logging.
                )

                print(                                  # Announce the save.
                    "Mid-epoch checkpoint saved:",      # Message prefix.
                    mid_checkpoint_path.name,           # Filename.
                )

        average_loss = (                                # Average loss for the epoch.
            running_loss                                # Total accumulated loss.
            / max(batches_seen, 1)                      # Divide by batches seen.
        )

        print()                                         # Blank line.
        print(                                          # Report epoch completion.
            f"Epoch {epoch + 1} complete. "             # Message part 1.
            f"Average loss: {average_loss:.4f}"         # Message part 2.
        )

        epoch_checkpoint_path = (                       # Build the full-epoch filename.
            checkpoint_path.with_name(                  # Based on the base path.
                f"{checkpoint_path.stem}"               # Base stem.
                f"_epoch_{epoch + 1}"                   # Epoch suffix.
                f"{checkpoint_path.suffix}"             # Extension.
            )
        )

        save_checkpoint(                                # Write the full-epoch checkpoint.
            checkpoint_path=epoch_checkpoint_path,      # Destination path.
            epoch=epoch + 1,                            # 1-indexed epoch.
            batch_index=0,                              # No batch offset.
            model=model,                                # Model to save.
            arcface=arcface,                            # ArcFace to save.
            optimizer=optimizer,                        # Optimizer to save.
            average_loss=average_loss,                  # Loss for logging.
        )

        print(                                          # Announce the save.
            "Epoch checkpoint saved to:",               # Message prefix.
            epoch_checkpoint_path,                      # Path.
        )

        # ── optional cleanup of mid-epoch files ──
        if not KEEP_MID_EPOCH_FILES:                    # Only if cleanup enabled.

            for old_checkpoint in (                     # Iterate matching files.
                checkpoint_path.parent.glob(            # Search the checkpoint dir.
                    f"{checkpoint_path.stem}"           # Base stem.
                    f"_epoch_{epoch + 1}"               # This epoch.
                    f"_batch_*"                         # Any batch.
                    f"{checkpoint_path.suffix}"         # Extension.
                )
            ):
                try:                                    # Attempt to delete.
                    old_checkpoint.unlink()             # Remove the file.
                except OSError:                         # Ignore missing-file races.
                    pass                                # Continue silently.

        print()                                         # Blank line.

    save_checkpoint(                                    # Save the final consolidated checkpoint.
        checkpoint_path=checkpoint_path,                # Base checkpoint path.
        epoch=settings.num_epochs,                      # Final epoch number.
        batch_index=0,                                  # No batch offset.
        model=model,                                    # Model to save.
        arcface=arcface,                                # ArcFace to save.
        optimizer=optimizer,                            # Optimizer to save.
        average_loss=average_loss,                      # Final loss.
    )

    print()                                             # Blank line.
    print(                                              # Announce completion.
        "Training complete."                            # Message.
    )

    print(                                              # Report final checkpoint.
        "Final checkpoint saved to:",                   # Message prefix.
        checkpoint_path,                                # Path.
    )

    return model                                        # Return the trained model.


def main():                                             # Command-line entry point.

    parser = argparse.ArgumentParser(                   # Create the arg parser.
        description=(                                   # Help text.
            "Train the wildlife "                       # Message part 1.
            "re-identification model"                   # Message part 2.
        )
    )

    parser.add_argument(                                # Define --dataset-path.
        "--dataset-path",                               # Flag name.
        required=True,                                  # Must be provided.
        help=(                                          # Help text.
            "Root directory of WildlifeReID-10k "       # Message part 1.
            "containing metadata.csv"                   # Message part 2.
        ),
    )

    parser.add_argument(                                # Define --resume.
        "--resume",                                     # Flag name.
        action="store_true",                            # Boolean flag.
        help=(                                          # Help text.
            "Resume training from the latest "          # Message part 1.
            "epoch or mid-epoch checkpoint."            # Message part 2.
        ),
    )

    args = parser.parse_args()                          # Parse CLI arguments.

    dataset_path = Path(                                # Normalize dataset path.
        args.dataset_path                               # From CLI.
    )

    metadata = load_dataset_metadata(                   # Load all metadata rows.
        dataset_path                                    # Dataset root.
    )

    train_metadata = create_training_metadata(          # Filter to training rows.
        metadata                                        # All rows.
    )

    identity_to_label = create_identity_mapping(        # Build identity-to-label map.
        train_metadata                                  # Training rows.
    )

    print(                                              # Report training image count.
        "Training images:",                             # Message prefix.
        len(train_metadata),                            # Count.
    )

    print(                                              # Report training identity count.
        "Training identities:",                         # Message prefix.
        len(identity_to_label),                         # Count.
    )

    train_model(                                        # Run training.
        train_metadata=train_metadata,                  # Training rows.
        dataset_path=dataset_path,                      # Dataset root.
        identity_to_label=identity_to_label,            # Label mapping.
        resume=args.resume,                             # Resume flag.
    )


if __name__ == "__main__":                              # Only run when executed directly.
    main()                                              # Invoke the entry point.