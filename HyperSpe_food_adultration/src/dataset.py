from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from .preprocessing import load_image


class ImageAdulterationDataset(Dataset):
    def __init__(self, root: str | Path, transform=None, labels: dict[str, int] | None = None):
        self.root = Path(root)
        self.transform = transform
        self.labels = labels or {'PER0': 0, 'PER5': 1, 'PER10': 2, 'PER15': 3}
        self.samples = []

        for class_dir in sorted(self.root.iterdir()):
            if not class_dir.is_dir():
                continue
            label_id = self.labels.get(class_dir.name)
            if label_id is None:
                continue
            for file_path in sorted(class_dir.rglob('*.jpg')):
                self.samples.append((file_path, label_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = load_image(img_path)
        if self.transform:
            image = self.transform(image)
        return image, label


def build_data_loaders(data_root: str | Path, batch_size: int = 16, image_size: int = 224, seed: int = 42):
    from .preprocessing import build_transforms

    data_root = Path(data_root)
    all_dirs = sorted([p for p in data_root.iterdir() if p.is_dir()])
    if not all_dirs:
        raise FileNotFoundError(f'No class folders found under {data_root}')

    samples = []
    labels = []
    class_to_idx = {}
    for class_dir in all_dirs:
        label = class_dir.name
        if label not in {'PER0', 'PER5', 'PER10', 'PER15'}:
            continue
        class_to_idx[label] = len(class_to_idx)
        for img in sorted(class_dir.rglob('*.jpg')):
            samples.append(str(img))
            labels.append(class_to_idx[label])

    train_idx, temp_idx = train_test_split(
        list(range(len(samples))),
        test_size=0.30,
        stratify=labels,
        random_state=seed,
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        stratify=[labels[i] for i in temp_idx],
        random_state=seed,
    )

    train_paths = [samples[i] for i in train_idx]
    val_paths = [samples[i] for i in val_idx]
    test_paths = [samples[i] for i in test_idx]

    train_labels = [labels[i] for i in train_idx]
    val_labels = [labels[i] for i in val_idx]
    test_labels = [labels[i] for i in test_idx]

    train_tfms, eval_tfms = build_transforms(image_size)

    train_ds = CustomImageFolderDataset(train_paths, train_labels, train_tfms)
    val_ds = CustomImageFolderDataset(val_paths, val_labels, eval_tfms)
    test_ds = CustomImageFolderDataset(test_paths, test_labels, eval_tfms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader, class_to_idx


class CustomImageFolderDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label
