"""Grad-CAM saliency overlays for qualitative analysis."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from .config import CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD

# Architecture-specific target-layer selectors (per timm model structure).
# Each function receives the CnnClassifier's .backbone and returns the target module.
_TARGET_LAYER_FN = {
    "densenet121":    lambda b: b.features.denseblock4,
    "resnet50":       lambda b: b.layer4,
    "vgg16":          lambda b: b.features[-3],  # last Conv2d before final max-pool
    "efficientnet_b0": lambda b: list(b.blocks.children())[-1],
}


def _get_target_layer(model, model_name: str):
    fn = _TARGET_LAYER_FN.get(model_name)
    if fn is not None:
        try:
            return fn(model.backbone)
        except Exception:  # noqa: BLE001
            pass
    # Generic fallback: deepest Conv2d in the backbone.
    last = None
    for m in model.backbone.modules():
        if isinstance(m, torch.nn.Conv2d):
            last = m
    if last is None:
        raise RuntimeError("No Conv2d layer found in backbone for Grad-CAM.")
    return last


def _denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Tensor (C,H,W) normalized -> HxWx3 float image in [0,1]."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = (tensor.cpu() * std + mean).clamp(0, 1)
    return img.permute(1, 2, 0).numpy()


def generate_gradcam(model, loader, device, model_name: str, out_dir: Path, n_samples: int):
    """Save Grad-CAM overlays for the first n_samples test images."""
    out_dir = out_dir / "gradcam"
    out_dir.mkdir(parents=True, exist_ok=True)
    target_layer = _get_target_layer(model, model_name)
    cam = GradCAM(model=model, target_layers=[target_layer])

    saved = 0
    for images, labels in loader:
        for i in range(images.size(0)):
            if saved >= n_samples:
                cam_cleanup(cam)
                return
            img_t = images[i : i + 1].to(device)
            try:
                with torch.enable_grad():
                    result = cam(input_tensor=img_t)
                grayscale = result[0] if result is not None else None
                if grayscale is None:
                    continue
            except Exception as exc:  # noqa: BLE001
                print(f"  Grad-CAM error on sample {saved}: {exc}")
                continue
            rgb = _denormalize(images[i])
            overlay = show_cam_on_image(rgb, grayscale, use_rgb=True)

            with torch.no_grad():
                pred = int(model(img_t).argmax(1).item())
            true = int(labels[i].item())

            fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
            axes[0].imshow(rgb)
            axes[0].set_title(f"true: {CLASS_NAMES[true]}")
            axes[0].axis("off")
            axes[1].imshow(overlay)
            axes[1].set_title(f"Grad-CAM pred: {CLASS_NAMES[pred]}")
            axes[1].axis("off")
            fig.tight_layout()
            tag = "correct" if pred == true else "wrong"
            fig.savefig(out_dir / f"{model_name}_{saved:02d}_{tag}.png", dpi=110)
            plt.close(fig)
            saved += 1
    cam_cleanup(cam)


def cam_cleanup(cam) -> None:
    # pytorch-grad-cam registers hooks; release them explicitly.
    try:
        cam.activations_and_grads.release()
    except Exception:  # noqa: BLE001
        pass
