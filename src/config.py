"""Configuration objects for the two run modes: quick (smoke test) and full.

The full configuration mirrors the protocol described in the project report
(two-stage transfer learning on the Kermany pediatric chest X-ray dataset).
The quick configuration runs a tiny subset for a couple of epochs so the whole
pipeline can be exercised on a CPU in a few minutes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Project root = parent of the src/ package.
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"

# ImageNet normalization (the timm backbones are pre-trained with these stats).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CLASS_NAMES = ("NORMAL", "PNEUMONIA")


@dataclass
class Config:
    """Hyper-parameters for one experiment run."""

    mode: str  # "quick" or "full"
    models: list[str]
    image_size: int = 224
    batch_size: int = 32

    # Stage 1: feature extraction (frozen backbone).
    stage1_epochs: int = 15
    stage1_lr: float = 1e-3

    # Stage 2: fine-tuning (last conv block unfrozen).
    stage2_epochs: int = 25
    stage2_lr: float = 1e-5

    # Callbacks.
    lr_factor: float = 0.2
    lr_patience: int = 3
    early_stop_patience: int = 7

    dropout: float = 0.3
    head_units: int = 256

    # Number of random seeds to repeat each experiment with.
    seeds: list[int] = field(default_factory=lambda: [42])

    # Internal validation split fraction taken from the training set.
    val_fraction: float = 0.10

    # Number of data-loading worker processes (0 is safest on Windows/CPU).
    num_workers: int = 0

    # Quick mode caps the number of images per split (None = use everything).
    subset_per_split: dict[str, int] | None = None

    # How many Grad-CAM overlays to generate.
    gradcam_samples: int = 6


# Backbone names as known to the `timm` library.
TIMM_NAMES = {
    "vgg16": "vgg16",
    "resnet50": "resnet50",
    "efficientnet_b0": "efficientnet_b0",
    "densenet121": "densenet121",
}

ALL_MODELS = ["vgg16", "resnet50", "efficientnet_b0", "densenet121"]


def quick_config(models: list[str] | None = None) -> Config:
    """Tiny configuration to verify the pipeline end-to-end on CPU."""
    return Config(
        mode="quick",
        models=models or ["densenet121"],
        batch_size=16,
        stage1_epochs=1,
        stage2_epochs=1,
        seeds=[42],
        subset_per_split={"train": 200, "val": 60, "test": 100},
        gradcam_samples=4,
    )


def full_config(models: list[str] | None = None, seeds: int = 3) -> Config:
    """Faithful configuration matching the report's protocol (slow on CPU)."""
    return Config(
        mode="full",
        models=models or ALL_MODELS,
        batch_size=32,
        stage1_epochs=15,
        stage2_epochs=25,
        seeds=list(range(42, 42 + seeds)),
        subset_per_split=None,
        gradcam_samples=6,
    )
