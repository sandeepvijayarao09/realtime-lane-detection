"""
Dataset handling for lane detection.

Reads the TuSimple lane benchmark format and rasterizes its lane polylines into
binary and per-lane instance masks. A synthetic mode (random-noise images with
random polylines) exists only so the pipeline can be smoke-tested without the
dataset; it teaches the model nothing about real roads.
"""

import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Sequence
import albumentations as A
from albumentations.pytorch import ToTensorV2


# TuSimple frames are 1280x720
TUSIMPLE_SIZE = (720, 1280)


def rasterize_tusimple_lanes(lanes: Sequence[Sequence[float]], h_samples: Sequence[float],
                             image_shape: Tuple[int, int] = TUSIMPLE_SIZE,
                             thickness: int = 5) -> np.ndarray:
    """
    Turn one TuSimple annotation into an instance mask.

    TuSimple stores each lane as x-coordinates sampled at the shared row
    positions `h_samples`; x < 0 (normally -2) means the lane is absent at
    that row.

    Args:
        lanes: List of lanes, each a list of x values aligned with h_samples
        h_samples: Row (y) positions shared by all lanes
        image_shape: (height, width) of the source image
        thickness: Line thickness in pixels at source resolution

    Returns:
        uint8 mask of shape image_shape: 0 = background, i = lane i (1-based)
    """
    mask = np.zeros(image_shape, dtype=np.uint8)
    for lane_id, xs in enumerate(lanes, start=1):
        points = [(int(round(x)), int(round(y))) for x, y in zip(xs, h_samples) if x >= 0]
        if len(points) < 2:
            continue
        cv2.polylines(mask, [np.array(points, dtype=np.int32)], isClosed=False,
                      color=lane_id, thickness=thickness)
    return mask


def load_tusimple_labels(label_files: Sequence[Path]) -> List[Dict]:
    """Read TuSimple label files (one JSON object per line)."""
    records = []
    for label_file in label_files:
        with open(label_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


class LaneDataset(Dataset):
    """
    TuSimple-format lane detection dataset.

    Expects the layout of the official download, e.g. for the training set:
    ```
    train_set/
        label_data_0313.json
        label_data_0531.json
        label_data_0601.json
        clips/0313-1/6040/20.jpg
        ...
    ```
    Each label line looks like
    `{"lanes": [[-2, 632, 625, ...], ...], "h_samples": [240, 250, ...], "raw_file": "clips/..."}`.
    For the test set, point `data_dir` at `test_set/` and pass
    `label_files=["test_label.json"]` (paths are relative to `data_dir`).
    """

    def __init__(self, data_dir: str, split: str = 'train', image_size: Tuple[int, int] = (384, 640),
                 augment: bool = True, use_mock_data: bool = False,
                 label_files: Optional[Sequence[str]] = None, lane_thickness: int = 5):
        """
        Args:
            data_dir: Root directory (TuSimple train_set/ or test_set/)
            split: 'train' enables augmentation; anything else disables it
            image_size: Target image size (height, width)
            augment: Apply augmentations
            use_mock_data: Generate synthetic data instead of reading TuSimple
            label_files: Label JSON files, relative to data_dir. Defaults to
                label_data_*.json, then test_label.json, then any *.json.
            lane_thickness: Lane line thickness in pixels at source resolution
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_size = image_size
        self.augment = augment
        self.use_mock_data = use_mock_data
        self.lane_thickness = lane_thickness
        self.records: List[Dict] = []

        # Create mock data if requested
        if use_mock_data:
            self._create_mock_data()
        else:
            self._validate_dataset()
            self.label_files = self._find_label_files(label_files)

        self.image_paths, self.labels = self._load_dataset()

        # Setup augmentations
        self.transform = self._get_transforms()

    def _create_mock_data(self) -> None:
        """Create a synthetic dataset (noise images + random polylines) for smoke tests."""
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
            for lane_id in (1, 2):
                y_start = np.random.randint(0, h // 2)
                x_start = np.random.randint(w // 4, 3 * w // 4)
                points = []
                for y in range(y_start, h, 20):
                    x = x_start + np.random.randint(-20, 20)
                    x = np.clip(x, 0, w - 1)
                    points.append([x, y])
                if points:
                    points = np.array(points, dtype=np.int32)
                    cv2.polylines(mask, [points], False, lane_id, 2)

            self.image_paths.append(str(img_path))
            self.labels.append(mask)

    def _validate_dataset(self) -> None:
        """Validate that dataset directory exists."""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.data_dir}")

    def _find_label_files(self, label_files: Optional[Sequence[str]]) -> List[Path]:
        """Resolve the TuSimple label JSON files to read."""
        if label_files:
            files = [self.data_dir / f for f in label_files]
        else:
            files = (sorted(self.data_dir.glob('label_data_*.json'))
                     or sorted(self.data_dir.glob('test_label.json'))
                     or sorted(self.data_dir.glob('*.json')))
        missing = [f for f in files if not f.exists()]
        if not files or missing:
            raise FileNotFoundError(
                f"No TuSimple label files found in {self.data_dir} "
                f"(missing: {[str(m) for m in missing]})"
            )
        return files

    def _load_dataset(self) -> Tuple[List[str], List]:
        """Load image paths and per-image lane annotations."""
        if self.use_mock_data:
            return self.image_paths, self.labels

        self.records = load_tusimple_labels(self.label_files)
        image_paths = [str(self.data_dir / r['raw_file']) for r in self.records]
        # Masks are rasterized lazily in __getitem__ to keep memory flat
        labels = [(r['lanes'], r['h_samples']) for r in self.records]
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
            ])
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
            Dictionary with 'image' (3xHxW float), 'mask' (1xHxW float, 1 = lane)
            and 'instance' (HxW long, 0 = background, i = lane i)
        """
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")

        if self.use_mock_data:
            instance = self.labels[idx]
        else:
            lanes, h_samples = self.labels[idx]
            instance = rasterize_tusimple_lanes(lanes, h_samples, image.shape[:2],
                                                thickness=self.lane_thickness)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize/flip the label map with the image (nearest-neighbour for masks)
        transformed = self.transform(image=image, mask=instance)
        instance = torch.as_tensor(np.asarray(transformed['mask']), dtype=torch.long)

        return {
            'image': transformed['image'],
            'mask': (instance > 0).float().unsqueeze(0),
            'instance': instance,
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
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # Smoke test with synthetic data
    print("Creating dataset with synthetic data...")
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
