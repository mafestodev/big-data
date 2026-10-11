import csv
from pathlib import Path

import torch

from wildlife_reid.gallery.build_gallery import (
    extract_model_state_dict,
    load_metadata,
)


def write_metadata(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["path", "identity", "split"],
        )
        writer.writeheader()
        writer.writerows(
            [
                {"path": "train.jpg", "identity": "animal-a", "split": "train"},
                {"path": "test.jpg", "identity": "animal-a", "split": "test"},
            ]
        )


def test_load_metadata_selects_requested_split(tmp_path: Path):
    metadata_path = tmp_path / "metadata.csv"
    write_metadata(metadata_path)

    metadata = load_metadata(metadata_path, split="train")

    assert len(metadata) == 1
    assert metadata[0]["path"] == "train.jpg"


def test_extract_model_state_dict_from_training_checkpoint():
    weights = {"embedding_layer.weight": torch.ones((2, 2))}

    extracted = extract_model_state_dict(
        {
            "epoch": 15,
            "model_state_dict": weights,
            "optimizer_state_dict": {},
        }
    )

    assert extracted is weights


def test_extract_model_state_dict_accepts_model_only_artifact():
    weights = {"embedding_layer.weight": torch.ones((2, 2))}

    assert extract_model_state_dict(weights) is weights
