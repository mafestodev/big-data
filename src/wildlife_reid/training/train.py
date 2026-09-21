
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoImageProcessor

from wildlife_reid.common.config import Settings
from wildlife_reid.training.dataset import create_train_loader
from wildlife_reid.training.model import SwinEmbeddingModel
from wildlife_reid.training.arcface import ArcFace


def train_model(
    train_metadata,
    dataset_path,
    identity_to_label
):
    """
    Train the Swin embedding model using ArcFace.

    Returns the trained embedding model.
    """

    # ---------------------------------------------------------
    # LOAD CONFIGURATION
    # ---------------------------------------------------------

    settings = Settings()

    device = torch.device(settings.device)

    print("Training device:", device)
    print("Training identities:", len(identity_to_label))


    # ---------------------------------------------------------
    # STEP 1 + STEP 2
    # DATASET AND DATALOADER
    # ---------------------------------------------------------

    image_processor = AutoImageProcessor.from_pretrained(
        settings.model_name
    )

    train_dataset, train_loader = create_train_loader(
        train_metadata=train_metadata,
        dataset_path=dataset_path,
        image_processor=image_processor,
        identity_to_label=identity_to_label,
        batch_size=settings.batch_size,
        num_workers=settings.num_workers
    )

    print("Training images:", len(train_dataset))
    print("Training batches:", len(train_loader))


    # ---------------------------------------------------------
    # STEP 3
    # CREATE SWIN EMBEDDING MODEL
    # ---------------------------------------------------------

    model = SwinEmbeddingModel(
        model_name=settings.model_name,
        embedding_dimension=settings.embedding_dimension,
        pretrained=True
    )

    model = model.to(device)


    # ---------------------------------------------------------
    # STEP 4
    # VERIFY THAT THE MODEL PRODUCES EMBEDDINGS
    # ---------------------------------------------------------

    sample_images, sample_labels = next(iter(train_loader))

    sample_images = sample_images.to(device)

    model.eval()

    with torch.no_grad():
        sample_embeddings = model(sample_images)

    print(
        "Embedding shape:",
        sample_embeddings.shape
    )

    expected_shape = (
        sample_images.shape[0],
        settings.embedding_dimension
    )

    if sample_embeddings.shape != expected_shape:
        raise RuntimeError(
            "Unexpected embedding dimensions. "
            f"Expected {expected_shape}, "
            f"received {tuple(sample_embeddings.shape)}."
        )


    # ---------------------------------------------------------
    # STEP 5
    # CREATE ARCFACE
    # ---------------------------------------------------------

    num_identities = len(identity_to_label)

    arcface = ArcFace(
        embedding_dimension=settings.embedding_dimension,
        num_classes=num_identities,
        scale=settings.arcface_scale,
        margin=settings.arcface_margin
    )

    arcface = arcface.to(device)


    # ---------------------------------------------------------
    # STEP 6
    # LOSS FUNCTION AND OPTIMIZER
    # ---------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        list(model.parameters()) +
        list(arcface.parameters()),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay
    )


    # ---------------------------------------------------------
    # STEP 7
    # ACTUAL TRAINING
    # ---------------------------------------------------------

    print("\nStarting training...\n")

    for epoch in range(settings.num_epochs):

        model.train()
        arcface.train()

        running_loss = 0.0

        for batch_index, (images, labels) in enumerate(train_loader):

            # Transfer the current batch to the GPU/CPU.
            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )

            # Remove gradients calculated for the previous batch.
            optimizer.zero_grad()

            # Generate wildlife embeddings.
            embeddings = model(images)

            # Calculate ArcFace logits.
            logits = arcface(
                embeddings,
                labels
            )

            # Calculate training loss.
            loss = criterion(
                logits,
                labels
            )

            # Calculate gradients.
            loss.backward()

            # Update Swin, embedding layer and ArcFace parameters.
            optimizer.step()

            running_loss += loss.item()

            # Display progress every 100 batches.
            if (batch_index + 1) % 100 == 0:

                print(
                    f"Epoch [{epoch + 1}/{settings.num_epochs}] "
                    f"Batch [{batch_index + 1}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f}"
                )

        average_loss = (
            running_loss /
            len(train_loader)
        )

        print(
            f"Epoch {epoch + 1} complete. "
            f"Average loss: {average_loss:.4f}"
        )


    # ---------------------------------------------------------
    # STEP 8
    # SAVE TRAINED MODEL CHECKPOINT
    # ---------------------------------------------------------

    checkpoint_path = Path(
        settings.checkpoint_path
    )

    # Create the models directory if it does not already exist.
    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save only the embedding model.
    #
    # ArcFace is required for training but is NOT required
    # when generating embeddings during inference.
    torch.save(
        model.state_dict(),
        checkpoint_path
    )

    print(
        "\nTraining complete."
    )

    print(
        "Checkpoint saved to:",
        checkpoint_path
    )

    return model
