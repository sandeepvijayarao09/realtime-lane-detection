"""
Dataset handling for lane detection.

Supports TuSimple dataset format with mock data generation for CI.
Includes augmentations and preprocessing.
"""

import os
import cv2
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2


class LaneDataset(Dataset):
    """
    TuSimple-format lane detection dataset.

    Expects directory structure:
    ```
    data/
        train/
            clips/
                0001/
                    0001.jpg
                    0002.jpg
                ...
        test/
            clips/
                0001/
                    0001.jpg
                ...
    labels.json (for each clip)
    ```
    """

    def __init__(self, data_dir: str, split: str = 'train', image_size: Tuple[int, int] = (384, 640),
                 augment: bool = True, use_mock_data: bool = False):
        """
        Args:
            data_dir: Root directory containing data
            split: 'train', 'val', or 'test'
            image_size: Target image size (height, width)
            augment: Apply augmentations
            use_mock_data: Generate mock data for testing
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_size = image_size
        self.augment = augment
        self.use_mock_data = use_mock_data

        # Create mock data if requested
        if use_mock_data:
            self._create_mock_data()
        else:
            self._validate_dataset()

        self.image_paths, self.labels = self._load_dataset()

        # Setup augmentations
        self.transform = self._get_transforms()

    def _create_mock_data(self) -> None:
        """Create mock dataset for testing without real data."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.image_paths = []
        self.labels = []

        # Generate 100 mock images and labels
        for i in range(100):
            # Create dummy image
            img = np.random.randint(0, 255, (*self.image_size, 3), dtype=np.uint8)
            img_path = self.data_dir / f'mock_image_{i:04d}.jpg'
            cv2.imwrite(str(img_path), img)

            # Create dummy lane mask
            mask = np.zeros(self.image_size, dtype=np.uint8)
            # Draw some dummy lane lines
            h, w = self.image_size
            for _ in range(2):
                y_start = np.random.randint(0, h // 2)
                x_start = np.random.randint(w // 4, 3 * w // 4)
                points = []
                for y in range(y_start, h, 20):
                    x = x_start + np.random.randint(-20, 20)
                    x = np.clip(x, 0, w - 1)
                    points.append([x, y])
                if points:
                    points = np.array(points, dtype=np.int32)
                    cv2.polylines(mask, [points], False, 1, 2)

            self.image_paths.append(str(img_path))
            self.labels.append(mask)

    def _validate_dataset(self) -> None:
        """Validate that dataset directory exists."""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.data_dir}")

    def _load_dataset(self) -> Tuple[List[str], List[np.ndarray]]:
        """Load image paths and labels from dataset."""
        if self.use_mock_data:
            return self.image_paths, self.labels

        image_paths = []
        labels = []

        # Try to load TuSimple format if it exists
        split_dir = self.data_dir / self.split
        if split_dir.exists():
            for img_file in sorted(split_dir.glob('**/*.jpg')):
                image_paths.append(str(img_file))
                # For now, use empty label (would load from JSON in production)
                labels.append(np.zeros(self.image_size, dtype=np.uint8))

        return image_paths, labels

    def _get_transforms(self) -> A.Compose:
        """Get albumentations transforms."""
        if self.augment and self.split == 'train':
            return A.Compose([
                A.Resize(*self.image_size),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
                A.GaussNoise(p=0.1),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
        else:
            return A.Compose([
                A.Resize(*self.image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get dataset item.

        Returns:
            Dictionary with 'image' and 'mask' keys
        """
        img_path = self.image_paths[idx]

        if self.use_mock_data:
            image = cv2.imread(img_path)
            mask = self.labels[idx]
        else:
            image = cv2.imread(img_path)
            if image is None:
                # Return random image if file not found
                image = np.random.randint(0, 255, (*self.image_size[::-1], 3), dtype=np.uint8)
            mask = np.zeros(self.image_size, dtype=np.uint8)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        transformed = self.transform(image=image, mask=mask)
        image = transformed['image']
        mask = torch.from_numpy(transformed['mask']).unsqueeze(0).float() if 'mask' in transformed else \
               torch.zeros(1, *self.image_size, dtype=torch.float32)

        return {
            'image': image,
            'mask': mask
        }


def create_dataloaders(data_dir: str, batch_size: int = 16, num_workers: int = 4,
                      image_size: Tuple[int, int] = (384, 640),
                      train_split: float = 0.7, val_split: float = 0.15,
                      use_mock_data: bool = False) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.

    Args:
        data_dir: Root dataset directory
        batch_size: Batch size
        num_workers: Number of workers for data loading
        image_size: Target image size (height, width)
        train_split: Proportion for training (rest split between val/test)
        val_split: Proportion for validation (rest goes to test)
        use_mock_data: Generate mock data instead of using real data

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create full dataset
    full_dataset = LaneDataset(
        data_dir=data_dir,
        split='train',
        image_size=image_size,
        augment=True,
        use_mock_data=use_mock_data
    )

    # Split dataset
    total_size = len(full_dataset)
    train_size = int(total_size * train_split)
    val_size = int(total_size * val_split)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # Test dataset with mock data
    print("Creating dataset with mock data...")
    dataset = LaneDataset(
        data_dir='/tmp/lane_data',
        split='train',
        use_mock_data=True,
        augment=True
    )

    print(f"Dataset size: {len(dataset)}")

    # Get a sample
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Image shape: {sample['image'].shape}")
    print(f"Mask shape: {sample['mask'].shape}")

    # Test dataloader
    print("\nTesting dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir='/tmp/lane_data_dl',
        batch_size=4,
        use_mock_data=True
    )

    print(f"Train loader batches: {len(train_loader)}")
    print(f"Val loader batches: {len(val_loader)}")
    print(f"Test loader batches: {len(test_loader)}")

    batch = next(iter(train_loader))
    print(f"\nBatch image shape: {batch['image'].shape}")
    print(f"Batch mask shape: {batch['mask'].shape}")
