"""CLI entry point: orchestrates download -> data -> train -> evaluate -> Grad-CAM.

Examples
--------
    python main.py --download                  # only fetch the dataset
    python main.py --quick                      # fast smoke test (default)
    python main.py --full                       # faithful run (slow on CPU)
    python main.py --models densenet121 resnet50 --seeds 1
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from src.config import ALL_MODELS, OUTPUTS_DIR, full_config, quick_config
from src.data import build_dataloaders
from src.download_data import ensure_dataset
from src.evaluate import evaluate_model, write_summary_table
from src.gradcam import generate_gradcam
from src.train import train_model


def parse_args():
    p = argparse.ArgumentParser(description="Chest X-ray pneumonia classification (CNN transfer learning).")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="fast smoke test (default)")
    mode.add_argument("--full", action="store_true", help="faithful full training (slow on CPU)")
    p.add_argument("--download", action="store_true", help="only download the dataset, then exit")
    p.add_argument("--models", nargs="+", choices=ALL_MODELS, help="subset of backbones to run")
    p.add_argument("--seeds", type=int, default=None, help="number of seeds (full mode)")
    return p.parse_args()


def main():
    args = parse_args()

    dataset_root = ensure_dataset()
    if args.download:
        return

    if args.full:
        cfg = full_config(models=args.models, seeds=args.seeds or 3)
    else:
        cfg = quick_config(models=args.models)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nMode: {cfg.mode} | device: {device} | models: {cfg.models} | seeds: {cfg.seeds}\n")

    out_dir = OUTPUTS_DIR / cfg.mode
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: list[dict] = []
    for model_name in cfg.models:
        print(f"=== {model_name} ===")
        per_seed: list[dict] = []
        last_model = last_loaders = None
        for seed in cfg.seeds:
            np.random.seed(seed)
            torch.manual_seed(seed)
            loaders, class_weights = build_dataloaders(dataset_root, cfg, seed)
            model, _history = train_model(model_name, loaders, class_weights, cfg, device, seed)
            metrics = evaluate_model(model, loaders["test"], device, model_name, out_dir)
            metrics["seed"] = seed
            per_seed.append(metrics)
            last_model, last_loaders = model, loaders

        # Aggregate across seeds (mean) for the summary row.
        agg = {"model": model_name}
        for key in ("accuracy", "precision", "recall", "specificity", "f1", "auc"):
            agg[key] = float(np.mean([m[key] for m in per_seed]))
        agg["accuracy_std"] = float(np.std([m["accuracy"] for m in per_seed]))
        summary.append(agg)

        # Grad-CAM on the last trained model of this backbone.
        try:
            generate_gradcam(last_model, last_loaders["test"], device, model_name,
                             out_dir, cfg.gradcam_samples)
        except Exception as exc:  # noqa: BLE001 - keep pipeline alive if CAM fails
            print(f"  Grad-CAM skipped for {model_name}: {exc}")

    write_summary_table(summary, out_dir)
    print(f"\nDone. Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
