from __future__ import annotations

from io import BytesIO

from PIL import Image, UnidentifiedImageError


class InvalidImageError(ValueError):
    pass


def decode_rgb_image(payload: bytes) -> Image.Image:
    if not payload:
        raise InvalidImageError("The uploaded image is empty")
    try:
        with Image.open(BytesIO(payload)) as source:
            source.verify()
        with Image.open(BytesIO(payload)) as source:
            return source.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidImageError("The uploaded file is not a valid image") from exc

