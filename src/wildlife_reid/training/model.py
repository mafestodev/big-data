from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class SwinEmbeddingModel(nn.Module):
    """Swin backbone followed by a normalized embedding projection."""

    def __init__(
        self,
        model_name: str = "swin_tiny_patch4_window7_224",
        embedding_dimension: int = 512,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as exc:  # pragma: no cover - dependency error path
            raise RuntimeError("Install project dependencies to create the model") from exc

        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        feature_dimension = self.backbone.num_features
        self.projection = nn.Linear(feature_dimension, embedding_dimension)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        embeddings = self.projection(features)
        return F.normalize(embeddings, p=2, dim=1)

