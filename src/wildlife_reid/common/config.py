from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class Settings:
    # Swin Transformer used as our feature extractor.
    model_name: str = "microsoft/swin-tiny-patch4-window7-224"

    # Size of the embedding generated for each wildlife image.
    embedding_dimension: int = 512

    # Swin input image size.
    image_size: int = 224

    # Number of images processed in each training batch.
    batch_size: int = 32

    # Training learning rate.
    learning_rate: float = 1e-4

    # Weight decay used by AdamW.
    weight_decay: float = 1e-4

    # Number of complete passes through the training dataset.
    num_epochs: int = 5

    # ArcFace parameters.
    arcface_scale: float = 64.0
    arcface_margin: float = 0.5

    # Number of DataLoader worker processes.
    num_workers: int = 2

    # Location where the trained embedding model will be saved.
    checkpoint_path: Path = PROJECT_ROOT / "models" / "swin_embedding_model.pt"

    # Gallery location used later by inference.
    gallery_path: Path = PROJECT_ROOT / "gallery" / "embeddings.npz"

    # Maximum size of an uploaded image accepted by the API.
    max_upload_bytes: int = 10 * 1024 * 1024

    # Use the GPU when one is available.
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def from_environment(cls) -> "Settings":
        return cls(
            model_name=os.getenv(
                "WILDLIFE_MODEL_NAME",
                cls.model_name,
            ),
            embedding_dimension=int(
                os.getenv(
                    "WILDLIFE_EMBEDDING_DIMENSION",
                    cls.embedding_dimension,
                )
            ),
            image_size=int(
                os.getenv(
                    "WILDLIFE_IMAGE_SIZE",
                    cls.image_size,
                )
            ),
            batch_size=int(
                os.getenv(
                    "WILDLIFE_BATCH_SIZE",
                    cls.batch_size,
                )
            ),
            learning_rate=float(
                os.getenv(
                    "WILDLIFE_LEARNING_RATE",
                    cls.learning_rate,
                )
            ),
            weight_decay=float(
                os.getenv(
                    "WILDLIFE_WEIGHT_DECAY",
                    cls.weight_decay,
                )
            ),
            num_epochs=int(
                os.getenv(
                    "WILDLIFE_NUM_EPOCHS",
                    cls.num_epochs,
                )
            ),
            arcface_scale=float(
                os.getenv(
                    "WILDLIFE_ARCFACE_SCALE",
                    cls.arcface_scale,
                )
            ),
            arcface_margin=float(
                os.getenv(
                    "WILDLIFE_ARCFACE_MARGIN",
                    cls.arcface_margin,
                )
            ),
            num_workers=int(
                os.getenv(
                    "WILDLIFE_NUM_WORKERS",
                    cls.num_workers,
                )
            ),
            checkpoint_path=Path(
                os.getenv(
                    "WILDLIFE_CHECKPOINT_PATH",
                    str(cls.checkpoint_path),
                )
            ),
            gallery_path=Path(
                os.getenv(
                    "WILDLIFE_GALLERY_PATH",
                    str(cls.gallery_path),
                )
            ),
            max_upload_bytes=int(
                os.getenv(
                    "WILDLIFE_MAX_UPLOAD_BYTES",
                    cls.max_upload_bytes,
                )
            ),
            device=os.getenv(
                "WILDLIFE_DEVICE",
                cls.device,
            ),
        )