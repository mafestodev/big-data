from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ArcFaceHead(nn.Module):
    """Additive angular-margin classification head used only during training."""

    def __init__(
        self,
        embedding_dimension: int,
        number_of_classes: int,
        scale: float = 64.0,
        margin: float = 0.5,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.empty(number_of_classes, embedding_dimension))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        target_angles = torch.acos(cosine)
        margin_logits = torch.cos(target_angles + self.margin)
        one_hot = F.one_hot(labels, num_classes=cosine.shape[1]).to(cosine.dtype)
        logits = one_hot * margin_logits + (1.0 - one_hot) * cosine
        return logits * self.scale
