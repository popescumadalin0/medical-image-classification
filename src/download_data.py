"""Download and locate the Chest X-Ray Images (Pneumonia) dataset.

Acquisition strategy (in order):
  1. kagglehub  -> dataset "paultimothymooney/chest-xray-pneumonia"
  2. direct ZIP from Mendeley Data (Kermany et al., no credentials needed)
  3. print clear manual instructions

After acquisition the dataset is normalized so that
``data/chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}/*.jpeg`` exists.
"""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

from .config import DATA_DIR

DATASET_ROOT = DATA_DIR / "chest_xray"

# Direct cached ZIP of Kermany et al. on Mendeley (ChestXRay2017.zip ~1.2 GB).
MENDELEY_URL = (
    "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/"
    "rscbjbr9sj/3/files/ChestXRay2017.zip"
)


def _has_valid_dataset(root: Path) -> bool:
    """True if root looks like a usable chest_xray dataset."""
    for split in ("train", "test"):
        for cls in ("NORMAL", "PNEUMONIA"):
            d = root / split / cls
            if not d.is_dir() or not any(d.iterdir()):
                return False
    return True


def _find_dataset_root(start: Path) -> Path | None:
    """Search downloaded/extracted tree for a folder holding train/test splits."""
    candidates = [start, *(p for p in start.rglob("*") if p.is_dir())]
    for c in candidates:
        if _has_valid_dataset(c):
            return c
    return None


def _normalize_into_place(found: Path) -> Path:
    """Copy the located dataset into data/chest_xray (idempotent)."""
    if found.resolve() == DATASET_ROOT.resolve():
        return DATASET_ROOT
    DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        src = found / split
        if src.is_dir():
            dst = DATASET_ROOT / split
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
    return DATASET_ROOT


def _try_kagglehub() -> Path | None:
    try:
        import kagglehub  # noqa: PLC0415
    except Exception:  # ImportError or import-time failures in kagglesdk
        print("  kagglehub not available, skipping.")
        return None
    try:
        print("  Trying kagglehub: paultimothymooney/chest-xray-pneumonia ...")
        path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
        found = _find_dataset_root(Path(path))
        if found:
            print(f"  kagglehub OK, dataset at: {found}")
            return found
        print("  kagglehub download did not contain expected structure.")
    except Exception as exc:  # noqa: BLE001 - network/auth errors are expected
        print(f"  kagglehub failed: {exc}")
    return None


def _try_mendeley() -> Path | None:
    import requests

    zip_path = DATA_DIR / "ChestXRay2017.zip"
    extract_dir = DATA_DIR / "mendeley_extract"
    try:
        print(f"  Trying direct download from Mendeley:\n    {MENDELEY_URL}")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with requests.get(MENDELEY_URL, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            done = 0
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        print(f"\r    {done / 1e6:.0f}/{total / 1e6:.0f} MB", end="")
            print()
        print("  Extracting ...")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(extract_dir)
        found = _find_dataset_root(extract_dir)
        if found:
            print(f"  Mendeley OK, dataset at: {found}")
            return found
        print("  Mendeley archive did not contain expected structure.")
    except Exception as exc:  # noqa: BLE001
        print(f"  Mendeley download failed: {exc}")
    return None


def _manual_instructions() -> None:
    print(
        "\n"
        "=========================================================================\n"
        "Automatic download failed. Please obtain the dataset manually:\n\n"
        "  Kaggle: https://www.kaggle.com/datasets/paultimothymooney/"
        "chest-xray-pneumonia\n"
        "  (or Mendeley: https://data.mendeley.com/datasets/rscbjbr9sj/3)\n\n"
        "Unzip it so that the following folders exist:\n"
        f"  {DATASET_ROOT / 'train' / 'NORMAL'}\n"
        f"  {DATASET_ROOT / 'train' / 'PNEUMONIA'}\n"
        f"  {DATASET_ROOT / 'test'  / 'NORMAL'}\n"
        f"  {DATASET_ROOT / 'test'  / 'PNEUMONIA'}\n"
        "Then re-run the program.\n"
        "=========================================================================\n"
    )


def ensure_dataset() -> Path:
    """Return the dataset root, downloading it if necessary.

    Raises FileNotFoundError if the dataset could not be obtained.
    """
    if _has_valid_dataset(DATASET_ROOT):
        print(f"Dataset already present at {DATASET_ROOT}")
        return DATASET_ROOT

    print("Acquiring Chest X-Ray Pneumonia dataset ...")
    found = _try_kagglehub() or _try_mendeley()
    if found is None:
        _manual_instructions()
        raise FileNotFoundError("Dataset not available.")

    root = _normalize_into_place(found)
    if not _has_valid_dataset(root):
        _manual_instructions()
        raise FileNotFoundError("Dataset structure invalid after download.")
    print(f"Dataset ready at {root}")
    return root


if __name__ == "__main__":
    ensure_dataset()
