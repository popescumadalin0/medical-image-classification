"""Dataset loading, stratified validation split, augmentation and DataLoaders."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .config import CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD, Config

_IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def _list_samples(split_dir: Path) -> tuple[list[Path], list[int]]:
    """Return (paths, labels) for one split folder with NORMAL/PNEUMONIA classes."""
    paths: list[Path] = []
    labels: list[int] = []
    for label, cls in enumerate(CLASS_NAMES):
        cls_dir = split_dir / cls
        if not cls_dir.is_dir():
            continue
        for p in sorted(cls_dir.iterdir()):
            if p.suffix.lower() in _IMG_EXT:
                paths.append(p)
                labels.append(label)
    return paths, labels


def _subset(paths, labels, n, seed):
    """Stratified subset of at most n samples (used in quick mode)."""
    if n is None or n >= len(paths):
        return paths, labels
    idx = np.arange(len(paths))
    keep, _ = train_test_split(
        idx, train_size=n, stratify=labels, random_state=seed
    )
    return [paths[i] for i in keep], [labels[i] for i in keep]


class ChestXrayDataset(Dataset):
    def __init__(self, paths: list[Path], labels: list[int], transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        # Grayscale X-rays are converted to 3-channel RGB for ImageNet backbones.
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), self.labels[i]


def _build_transforms(cfg: Config):
    norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    train_tf = transforms.Compose(
        [
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            norm,
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.ToTensor(),
            norm,
        ]
    )
    return train_tf, eval_tf


def build_dataloaders(dataset_root: Path, cfg: Config, seed: int):
    """Create train/val/test DataLoaders plus class weights for balanced loss.

    A stratified ``val_fraction`` slice of the official train split is used as
    the internal validation set (the official val split has only 16 images).
    """
    train_paths, train_labels = _list_samples(dataset_root / "train")
    test_paths, test_labels = _list_samples(dataset_root / "test")

    tr_p, va_p, tr_l, va_l = train_test_split(
        train_paths,
        train_labels,
        test_size=cfg.val_fraction,
        stratify=train_labels,
        random_state=seed,
    )

    if cfg.subset_per_split:
        tr_p, tr_l = _subset(tr_p, tr_l, cfg.subset_per_split.get("train"), seed)
        va_p, va_l = _subset(va_p, va_l, cfg.subset_per_split.get("val"), seed)
        test_paths, test_labels = _subset(
            test_paths, test_labels, cfg.subset_per_split.get("test"), seed
        )

    train_tf, eval_tf = _build_transforms(cfg)
    train_ds = ChestXrayDataset(tr_p, tr_l, train_tf)
    val_ds = ChestXrayDataset(va_p, va_l, eval_tf)
    test_ds = ChestXrayDataset(test_paths, test_labels, eval_tf)

    def _loader(ds, shuffle):
        return DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            num_workers=cfg.num_workers,
            pin_memory=False,
        )

    loaders = {
        "train": _loader(train_ds, True),
        "val": _loader(val_ds, False),
        "test": _loader(test_ds, False),
    }

    # Class weights inversely proportional to frequency (balanced cross-entropy).
    counts = np.bincount(tr_l, minlength=len(CLASS_NAMES)).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (len(CLASS_NAMES) * counts)
    class_weights = torch.tensor(weights, dtype=torch.float32)

    sizes = {k: len(v.dataset) for k, v in loaders.items()}
    print(
        f"  Samples -> train={sizes['train']} val={sizes['val']} test={sizes['test']} "
        f"| class weights={class_weights.tolist()}"
    )
    return loaders, class_weights
