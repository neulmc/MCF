# MCF-Net: Multi-Modal Cross-Attention Fusion Network with Difference Convolution for Object Detection

This repository contains the official implementation of **MCF-Net**, 
a multi-modal object detection method based on cross-modal attention and difference convolution. 
The framework supports both bi-modal and tri-modal configurations.

---

## 📁 Directory Structure
```
MCF-Net/
├── ultralytics/ # Modified Ultralytics YOLO framework
│── train/
│ │── train_mcf.py  Training scripts
│ │── predict_mcf.py  Inference scripts
│ │── eval_mcf(bi_modal).py  # Evaluation scripts (bi-modal)
│ └── eval_mcf(tri_modal).py  # Evaluation scripts (tri-modal)
├── runs/ # Training logs and weights
├── ...
└── README.md
```
---

## 🚀 Quick Start

### 1. Environment Setup

```bash
conda create -n mcfnet python=3.8
conda activate mcfnet
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics numpy opencv-python matplotlib pillow pyyaml tqdm seaborn pandas scipy thop albumentations
```

### 2. Data Preparation
Organize the dataset in the following structure:

For bi-modal (RGB+IR **M3FD**) or tri-modal (RGB+IR+Depth **AIC2026**):

```
dataset/
├── images/
│   ├── train/
│   └── val/
├── infrared/
│   ├── train/
│   └── val/
├── depth/  ## For AIC2026 only
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```
### 3. Training & Evaluation
Our model is built upon the YOLO11 architecture. After training, the model weights and logs will be saved in the `runs/` directory. You can then run the evaluation scripts to test the trained model on bi-modal or tri-modal settings.
```bash
# Training
python train/train_mcf.py

# Evaluation (bi-modal)
python train/eval_mcf_bi.py

# Evaluation (tri-modal)
python train/eval_mcf_tri.py
```
The training and evaluation scripts share two custom arguments for modality control:
```python
lmc_multi_modal = True          # Enable tri-modal (RGB+IR+Depth). Set to False for bi-modal or single-modal.
lmc_multi_modal_m3data = False  # False for AIC2026, True for M3FD bi-modal (RGB+IR) dataset.
```
Switching these values will change the data loading behavior to accommodate different modality configurations.

### Core Modules

Our core modules (DC-Bottleneck, CMGA, CMSF) are implemented in MCF-Net/ultralytics/nn/modules/lmc_multi.py, with model definitions and task logic in MCF-Net/ultralytics/nn/tasks.py. The AM-Backbone is defined via YAML configuration files in MCF-Net/ultralytics/cfg/models/.

| Module | Description |
| :--- | :--- |
| AM-Backbone | Asymmetric backbone with lightweight IR/Depth branches (channel reduction ratio r=0.5) |
| DC-Bottleneck | Difference Convolution Bottleneck for edge and structural feature extraction |
| CMGA | Cross-Modal Grid Attention Enhancement for inter-modal feature interaction |
| CMSF | Cross-Modal Spatial Fusion for adaptive multi-modal feature fusion |

The dataset loading pipeline has been rewritten to support both bi-modal (RGB+IR, RGB+Depth) and tri-modal (RGB+IR+Depth) inputs. To ensure correct data loading and model behavior, please use the entire repository as provided. Simply copying individual modules may cause dependency errors.
 
## 📊 Results
### AIC2026 Challenge Dataset

| Method           | mAP50  | mAP50-95 | Params (M) |
|:-----------------|:-------|:---------|:-----------|
| MCF-Net (Ours)   | 63.3   | 39.0     | 3.9        |
| MCF-Net-S (Ours) | 64.2   | 39.6     | 14.7       |

### M3FD Public Dataset

| Method | mAP50 | mAP50-95 | Params (M) |
| :--- | :--- | :--- |:-----------|
| MCF-Net (Ours) | 88.9 | 59.7 | 3.4        |
| MCF-Net-S (Ours) | 91.1 | 62.4 | 12.4       |

We have released our pre-trained model and visualization results on the M3FD public dataset, along with the dataset split details, at https://pan.baidu.com/s/1aug5GcZqYubE8WZlVHhISA code: jqt5. 
The AIC2026 dataset is not publicly available for direct download, but it can be accessed by participating in the competition and applying through its official website (https://www.aicomp.cn/tracks/3633.html).

## 📧 Contact
For questions or issues, please open an issue or contact the authors.


## 🙏 Acknowledgements
We thank the following open-source projects and datasets for their valuable contributions:

[1] **(Ultralytics YOLO)** [Ultralytics YOLO Framework](https://github.com/ultralytics/ultralytics)

[2] **(DEYOLO)** [Dual-Feature-Enhancement YOLO for Cross-Modality Object Detection](https://github.com/chips96/DEYOLO)

[3] **(M3FD Dataset)** [Target-aware Dual Adversarial Learning and a Multi-scenario Multi-Modality Benchmark to Fuse Infrared and Visible for Object Detection](https://github.com/JinyuanLiu-CV/TarDAL)

[4] **(AIC2026 Dataset)** [Global Campus Artificial Intelligence Algorithm Elite Competition](https://www.aicomp.cn/tracks/3633.html)
