"""Build ImageNet-pretrained backbones with a custom classification head."""
from __future__ import annotations

import timm
import torch
from torch import nn

from .config import TIMM_NAMES, Config


class CnnClassifier(nn.Module):
    """timm backbone (global-avg-pooled features) + custom 2-class head."""

    def __init__(self, model_name: str, cfg: Config, pretrained: bool = True):
        super().__init__()
        timm_name = TIMM_NAMES[model_name]
        # num_classes=0 + global_pool='avg' -> backbone outputs a feature vector.
        self.backbone = timm.create_model(
            timm_name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )
        feat_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Dropout(cfg.dropout),
            nn.Linear(feat_dim, cfg.head_units),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_units, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


def freeze_backbone(model: CnnClassifier) -> None:
    """Stage 1: freeze the whole backbone, train only the head."""
    for p in model.backbone.parameters():
        p.requires_grad = False


def unfreeze_last_block(model: CnnClassifier) -> None:
    """Stage 2: unfreeze the last convolutional block for fine-tuning.

    timm modules expose ``group_matcher`` / named children that let us target
    the deepest stage generically. We unfreeze the last top-level child module
    of the backbone, which corresponds to the final conv block for the four
    architectures used here (VGG features tail, ResNet layer4, DenseNet
    denseblock4+norm5, EfficientNet final blocks/conv_head).
    """
    children = list(model.backbone.named_children())
    # Unfreeze roughly the last third of top-level backbone modules so the
    # deepest conv block(s) become trainable regardless of architecture.
    n_unfreeze = max(1, len(children) // 3)
    to_unfreeze = {name for name, _ in children[-n_unfreeze:]}
    for name, module in model.backbone.named_children():
        requires = name in to_unfreeze
        for p in module.parameters():
            p.requires_grad = requires


def trainable_parameters(model: nn.Module):
    return [p for p in model.parameters() if p.requires_grad]
