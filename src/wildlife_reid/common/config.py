from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class Settings:
    model_name: str = "swin_tiny_patch4_window7_224"
    embedding_dimension: int = 512
    image_size: int = 224
    checkpoint_path: Path = PROJECT_ROOT / "models" / "swin_arcface_v1.pt"
    gallery_path: Path = PROJECT_ROOT / "gallery" / "embeddings.npz"
    max_upload_bytes: int = 10 * 1024 * 1024
    device: str = "cpu"

    @classmethod
    def from_environment(cls) -> "Settings":
        return cls(
            model_name=os.getenv("WILDLIFE_MODEL_NAME", cls.model_name),
            embedding_dimension=int(
                os.getenv("WILDLIFE_EMBEDDING_DIMENSION", cls.embedding_dimension)
            ),
            image_size=int(os.getenv("WILDLIFE_IMAGE_SIZE", cls.image_size)),
            checkpoint_path=Path(
                os.getenv("WILDLIFE_CHECKPOINT_PATH", str(cls.checkpoint_path))
            ),
            gallery_path=Path(os.getenv("WILDLIFE_GALLERY_PATH", str(cls.gallery_path))),
            max_upload_bytes=int(
                os.getenv("WILDLIFE_MAX_UPLOAD_BYTES", cls.max_upload_bytes)
            ),
            device=os.getenv("WILDLIFE_DEVICE", cls.device),
        )

