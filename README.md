# Medical Image Classification — Chest X-Ray Pneumonia (CNN transfer learning)

Implements the pipeline described in the accompanying report
(`Computer Vision -  Project-Exam.docx`): binary pneumonia classification on the
Kermany pediatric chest X-ray dataset using four ImageNet-pretrained CNN
backbones (**VGG16, ResNet50, EfficientNet-B0, DenseNet121**) with a two-stage
transfer-learning protocol (frozen feature extraction → partial fine-tuning),
class-balanced loss, data augmentation, and full evaluation
(accuracy / precision / recall / specificity / F1 / AUC, confusion matrix, ROC,
Grad-CAM).

## Project layout
```
src/
  config.py         hyper-parameters (quick vs full)
  download_data.py  dataset acquisition (kagglehub -> Mendeley -> manual)
  data.py           stratified split, augmentation, DataLoaders, class weights
  models.py         timm backbone + custom head, freeze/unfreeze helpers
  train.py          two-stage training loop with callbacks
  evaluate.py       metrics + confusion matrix + ROC + summary table
  gradcam.py        Grad-CAM overlays
main.py             CLI entry point
```

## Setup (Windows, PyCharm)

This machine has **no NVIDIA GPU**, so the CPU build of PyTorch is used and
training runs on the CPU.

1. Create a virtual environment with **Python 3.12** (PyTorch has no 3.14 wheels yet):
   ```powershell
   py -3.12 -m venv .venv
   ```
2. Install PyTorch (CPU) and the rest of the dependencies:
   ```powershell
   .venv\Scripts\python -m pip install --upgrade pip
   .venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   .venv\Scripts\python -m pip install -r requirements.txt
   ```
3. In PyCharm: *File → Settings → Project → Python Interpreter → Add Interpreter
   → Existing → select* `.venv\Scripts\python.exe`.

## Get the dataset
```powershell
.venv\Scripts\python main.py --download
```
Downloads ~1.2 GB into `data/chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}`.
If automatic download fails, download manually from
[Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
or [Mendeley](https://data.mendeley.com/datasets/rscbjbr9sj/3) and unzip into
that path.

## Run

Quick smoke test (default — one backbone, tiny subset, ~minutes on CPU):
```powershell
.venv\Scripts\python main.py --quick
```

Full faithful run (all four backbones, full epochs, 3 seeds — **slow on CPU**):
```powershell
.venv\Scripts\python main.py --full
```

Subset of models / seeds:
```powershell
.venv\Scripts\python main.py --models densenet121 resnet50 --full --seeds 1
```

Outputs (metrics JSON, confusion matrices, ROC curves, Grad-CAM overlays, and a
comparative `summary.md`) are written to `outputs/<mode>/`.
