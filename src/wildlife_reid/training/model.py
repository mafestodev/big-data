
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import SwinConfig, SwinModel


class SwinEmbeddingModel(nn.Module):
    """
    Swin Transformer feature extractor followed by an embedding layer.
    """

    def __init__(
        self,
        model_name,
        embedding_dimension,
        pretrained=True
    ):
        super().__init__()

        # During training we begin with pretrained Swin weights.
        if pretrained:
            self.backbone = SwinModel.from_pretrained(
                model_name
            )

        # During inference the architecture is created first,
        # after which our trained checkpoint is loaded into it.
        else:
            config = SwinConfig.from_pretrained(
                model_name
            )

            self.backbone = SwinModel(config)

        # Determine Swin's output feature size.
        hidden_size = self.backbone.config.hidden_size

        # Project Swin features into our chosen embedding space.
        self.embedding_layer = nn.Linear(
            hidden_size,
            embedding_dimension
        )


    def forward(self, pixel_values):

        # Extract image features using Swin.
        outputs = self.backbone(
            pixel_values=pixel_values
        )

        # Obtain Swin's pooled image representation.
        features = outputs.pooler_output

        # Convert the features into our embedding representation.
        embeddings = self.embedding_layer(features)

        # L2 normalisation makes embeddings suitable for
        # angular/cosine-based comparison.
        embeddings = F.normalize(
            embeddings,
            p=2,
            dim=1
        )

        return embeddings
