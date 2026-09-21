
import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFace(nn.Module):
    """
    ArcFace classification head used to train discriminative embeddings.
    """

    def __init__(
        self,
        embedding_dimension,
        num_classes,
        scale=64.0,
        margin=0.5
    ):
        super().__init__()

        self.scale = scale
        self.margin = margin

        # Learnable class prototypes.
        self.weight = nn.Parameter(
            torch.empty(
                num_classes,
                embedding_dimension
            )
        )

        # Initialise ArcFace weights.
        nn.init.xavier_uniform_(self.weight)


    def forward(self, embeddings, labels):

        # Normalise the class weights.
        normalized_weights = F.normalize(
            self.weight,
            p=2,
            dim=1
        )

        # Cosine similarity between each embedding
        # and every training identity.
        cosine = F.linear(
            embeddings,
            normalized_weights
        )

        # Keep values inside the safe numerical range for acos.
        cosine = torch.clamp(
            cosine,
            -1.0 + 1e-7,
            1.0 - 1e-7
        )

        # Convert cosine similarity into angles.
        theta = torch.acos(cosine)

        # Apply the angular margin.
        target_cosine = torch.cos(
            theta + self.margin
        )

        # Identify the correct class for every image.
        one_hot = F.one_hot(
            labels,
            num_classes=cosine.size(1)
        ).float()

        # Apply the ArcFace margin only to the correct identity.
        logits = (
            one_hot * target_cosine
            +
            (1.0 - one_hot) * cosine
        )

        # Apply ArcFace scaling.
        logits = logits * self.scale

        return logits
