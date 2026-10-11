import torch
from PIL import Image

from wildlife_reid.training.preprocessing import preprocess_image


class FakeImageProcessor:
    def __call__(self, *, images, return_tensors):
        assert images.mode == "RGB"
        assert return_tensors == "pt"
        return {
            "pixel_values": torch.ones(
                (1, 3, 224, 224),
                dtype=torch.float32,
            )
        }


def test_preprocess_image_returns_single_image_tensor():
    image = Image.new("RGB", (80, 40), "green")

    result = preprocess_image(
        FakeImageProcessor(),
        image,
    )

    assert result.shape == (3, 224, 224)
    assert result.dtype == torch.float32
