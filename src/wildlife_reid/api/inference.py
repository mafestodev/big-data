from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from wildlife_reid.common.config import Settings
from wildlife_reid.gallery.vector_search import ExactCosineIndex
from wildlife_reid.training.model import SwinEmbeddingModel
from wildlife_reid.training.preprocessing import (
    create_image_processor,
    preprocess_image,
)


@dataclass(frozen=True)
class Identification:
    identity: str
    species: str
    similarity: float
    matched_image_path: str


class InferenceService:
    def __init__(
        self,
        model: SwinEmbeddingModel,
        index: ExactCosineIndex,
        settings: Settings,
    ) -> None:
        self.model = model.to(settings.device).eval()
        self.index = index
        self.settings = settings
        self.image_processor = create_image_processor(
            settings.model_name
        )

    @classmethod
    def load(cls, settings: Settings) -> InferenceService:
        if not settings.checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {settings.checkpoint_path}")
        if not settings.gallery_path.exists():
            raise FileNotFoundError(f"Gallery not found: {settings.gallery_path}")

        model = SwinEmbeddingModel(
            model_name=settings.model_name,
            embedding_dimension=settings.embedding_dimension,
            pretrained=False,
        )
        checkpoint = torch.load(Path(settings.checkpoint_path), map_location=settings.device, weights_only=True)

        model_state_dict = checkpoint.get(
            "model_state_dict",
            checkpoint,
        )

        model.load_state_dict(model_state_dict)

        return cls(model, ExactCosineIndex.load(settings.gallery_path), settings)

    @torch.inference_mode()
    def identify(self, image: Image.Image) -> Identification:
        pixel_values = preprocess_image(
            self.image_processor,
            image,
        )
        batch = pixel_values.unsqueeze(0).to(
            self.settings.device
        )
        embedding = (
            self.model(batch)
            .squeeze(0)
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        match = self.index.search(embedding, k=1)[0]

        return Identification(
            identity=match.identity,
            species=match.species,
            similarity=match.similarity,
            matched_image_path=match.image_path,
        )

