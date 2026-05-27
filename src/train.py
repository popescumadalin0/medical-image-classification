"""Two-stage transfer-learning training loop with callbacks."""
from __future__ import annotations

import copy

import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from .config import Config
from .models import (
    CnnClassifier,
    freeze_backbone,
    trainable_parameters,
    unfreeze_last_block,
)


def _run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    torch.set_grad_enabled(train)
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        if train:
            optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        if train:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        n += images.size(0)
    torch.set_grad_enabled(True)
    return total_loss / max(n, 1), correct / max(n, 1)


def _train_stage(
    model, loaders, criterion, device, epochs, lr, cfg: Config, stage_name: str
):
    optimizer = torch.optim.Adam(trainable_parameters(model), lr=lr)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=cfg.lr_factor, patience=cfg.lr_patience
    )
    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    history = []
    bar = tqdm(range(1, epochs + 1), desc=stage_name, leave=False)
    for epoch in bar:
        tr_loss, tr_acc = _run_epoch(model, loaders["train"], criterion, optimizer, device, True)
        va_loss, va_acc = _run_epoch(model, loaders["val"], criterion, optimizer, device, False)
        scheduler.step(va_loss)
        history.append(
            {"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
             "val_loss": va_loss, "val_acc": va_acc}
        )
        bar.set_postfix(tr_loss=f"{tr_loss:.3f}", va_loss=f"{va_loss:.3f}", va_acc=f"{va_acc:.3f}")

        if va_loss < best_val - 1e-4:
            best_val = va_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.early_stop_patience:
                bar.write(f"    early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    return history


def train_model(model_name: str, loaders, class_weights, cfg: Config, device, seed: int):
    """Run Stage 1 (feature extraction) then Stage 2 (fine-tuning).

    Returns the trained model plus the two-stage training history.
    """
    torch.manual_seed(seed)
    model = CnnClassifier(model_name, cfg, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    print(f"  [Stage 1] feature extraction ({cfg.stage1_epochs} epochs)")
    freeze_backbone(model)
    hist1 = _train_stage(
        model, loaders, criterion, device,
        cfg.stage1_epochs, cfg.stage1_lr, cfg, "stage1",
    )

    print(f"  [Stage 2] fine-tuning ({cfg.stage2_epochs} epochs)")
    unfreeze_last_block(model)
    hist2 = _train_stage(
        model, loaders, criterion, device,
        cfg.stage2_epochs, cfg.stage2_lr, cfg, "stage2",
    )

    return model, {"stage1": hist1, "stage2": hist2}
