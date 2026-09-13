from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from wildlife_reid.common.config import Settings
from wildlife_reid.gallery.vector_search import ExactCosineIndex
from wildlife_reid.training.model import SwinEmbeddingModel
from wildlife_reid.training.transforms import inference_transforms


@dataclass(frozen=True)
class Identification:
    identity: str
    similarity: float


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
        self.transform = inference_transforms(settings.image_size)

    @classmethod
    def load(cls, settings: Settings) -> "InferenceService":
        if not settings.checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {settings.checkpoint_path}")
        if not settings.gallery_path.exists():
            raise FileNotFoundError(f"Gallery not found: {settings.gallery_path}")

        model = SwinEmbeddingModel(
            model_name=settings.model_name,
            embedding_dimension=settings.embedding_dimension,
            pretrained=False,
        )
        checkpoint = torch.load(
            Path(settings.checkpoint_path), map_location=settings.device, weights_only=True
        )
        model.load_state_dict(checkpoint)
        return cls(model, ExactCosineIndex.load(settings.gallery_path), settings)

    @torch.inference_mode()
    def identify(self, image: Image.Image) -> Identification:
        tensor = self.transform(image).unsqueeze(0).to(self.settings.device)
        embedding = self.model(tensor).squeeze(0).cpu().numpy().astype(np.float32)
        match = self.index.search(embedding, k=1)[0]
        return Identification(match.identity, match.similarity)

