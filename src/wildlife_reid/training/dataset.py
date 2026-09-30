import os

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def create_train_augmentation():
    """
    Create mild augmentation suitable for wildlife re-identification.

    The transformations introduce realistic variation without deliberately
    destroying the visual characteristics used to identify an individual.
    """

    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.1
        )
    ])


class WildlifeReIDDataset(Dataset):
    """
    PyTorch Dataset for WildlifeReID-10k.

    The dataset loads and augments images.

    Hugging Face image preprocessing is intentionally performed in the
    batch collate function rather than once per individual image.
    """

    def __init__(
        self,
        metadata,
        dataset_path,
        image_processor,
        identity_to_label,
        augmentation=None
    ):
        self.metadata = list(metadata)
        self.dataset_path = dataset_path
        self.image_processor = image_processor
        self.identity_to_label = identity_to_label
        self.augmentation = augmentation

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        row = self.metadata[index]

        image_path = os.path.join(
            self.dataset_path,
            row["path"]
        )

        image = Image.open(image_path).convert("RGB")

        if self.augmentation is not None:
            image = self.augmentation(image)

        label = self.identity_to_label[row["identity"]]

        return (
            image,
            torch.tensor(label, dtype=torch.long)
        )


def create_collate_fn(image_processor):
    """
    Create a collate function that processes an entire batch of images
    through the Hugging Face image processor at once.
    """

    def collate_fn(batch):
        images, labels = zip(*batch)

        processed = image_processor(
            images=list(images),
            return_tensors="pt"
        )

        pixel_values = processed["pixel_values"]

        labels = torch.stack(labels)

        return pixel_values, labels

    return collate_fn


def create_train_loader(
    train_metadata,
    dataset_path,
    image_processor,
    identity_to_label,
    batch_size,
    num_workers=4
):
    """
    Create the Dataset and DataLoader used during training.
    """

    train_augmentation = create_train_augmentation()

    train_dataset = WildlifeReIDDataset(
        metadata=train_metadata,
        dataset_path=dataset_path,
        image_processor=image_processor,
        identity_to_label=identity_to_label,
        augmentation=train_augmentation
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        collate_fn=create_collate_fn(image_processor)
    )

    return train_dataset, train_loader