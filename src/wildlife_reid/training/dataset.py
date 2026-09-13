from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    identity: str
    class_index: int


class WildlifeDataset(Dataset):
    """Dataset backed by validated image/identity records.

    A separate preparation command should convert WildlifeReID-10k metadata
    into split-specific records before constructing this class.
    """

    def __init__(self, records: list[ImageRecord], transform: Callable | None = None) -> None:
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        with Image.open(record.path) as source:
            image = source.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, record.class_index, record.identity

