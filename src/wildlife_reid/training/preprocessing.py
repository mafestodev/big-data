from __future__ import annotations

from typing import Any

import torch
from PIL import Image


def create_image_processor(model_name: str):
    """Create the canonical image processor used by training and inference."""

    from transformers import AutoImageProcessor

    return AutoImageProcessor.from_pretrained(model_name)


def preprocess_image(image_processor: Any, image: Image.Image) -> torch.Tensor:
    """Convert one RGB image to the pixel tensor expected by Swin."""

    processed = image_processor(
        images=image,
        return_tensors="pt",
    )

    if "pixel_values" not in processed:
        raise ValueError(
            "Image processor did not return 'pixel_values'."
        )

    pixel_values = processed["pixel_values"]

    if not isinstance(pixel_values, torch.Tensor):
        raise TypeError(
            "Image processor 'pixel_values' must be a torch.Tensor."
        )

    if pixel_values.ndim != 4 or pixel_values.shape[0] != 1:
        raise ValueError(
            "Expected one processed image with shape [1, C, H, W], "
            f"received {tuple(pixel_values.shape)}."
        )

    return pixel_values.squeeze(0)

