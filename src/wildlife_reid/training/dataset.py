
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
    """

    def __init__(
        self,
        metadata,
        dataset_path,
        image_processor,
        identity_to_label,
        augmentation=None
    ):
        self.metadata = metadata.reset_index(drop=True)
        self.dataset_path = dataset_path
        self.image_processor = image_processor
        self.identity_to_label = identity_to_label
        self.augmentation = augmentation


    def __len__(self):
        return len(self.metadata)


    def __getitem__(self, index):

        # Get metadata belonging to this image.
        row = self.metadata.iloc[index]

        # Construct the complete image path.
        image_path = os.path.join(
            self.dataset_path,
            row["path"]
        )

        # Load the image as RGB.
        image = Image.open(image_path).convert("RGB")

        # Apply augmentation to training images.
        if self.augmentation is not None:
            image = self.augmentation(image)

        # Apply the preprocessing expected by Swin.
        processed = self.image_processor(
            images=image,
            return_tensors="pt"
        )

        # Remove the temporary batch dimension.
        pixel_values = processed["pixel_values"].squeeze(0)

        # Convert the wildlife identity into its numerical training label.
        label = self.identity_to_label[row["identity"]]

        return (
            pixel_values,
            torch.tensor(label, dtype=torch.long)
        )


def create_train_loader(
    train_metadata,
    dataset_path,
    image_processor,
    identity_to_label,
    batch_size,
    num_workers=2
):
    """
    Create the Dataset and DataLoader used during training.
    """

    # Create training augmentation.
    train_augmentation = create_train_augmentation()

    # STEP 1: Create the training Dataset.
    train_dataset = WildlifeReIDDataset(
        metadata=train_metadata,
        dataset_path=dataset_path,
        image_processor=image_processor,
        identity_to_label=identity_to_label,
        augmentation=train_augmentation
    )

    # STEP 2: Create batches of training images.
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_dataset, train_loader
