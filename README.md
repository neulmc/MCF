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


### Detailed AP Scores at All IoU Thresholds (AIC2026) — Ablation Study

The table below presents the detailed AP scores across 10 IoU thresholds from 0.50 to 0.95 for each ablation configuration. 
These results supplement **Table 6** in the paper.

| Configuration | **mAP50-95 (primary metric)** | AP50 | AP55 | AP60 | AP65 | AP70 | AP75 | AP80 | AP85 | AP90 | AP95 |
| :--- |:-----------:| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline |    37.69    | 61.68 | 59.35 | 56.69 | 52.63 | 46.21 | 38.21 | 27.83 | 19.70 | **11.67** | 2.94 |
| + DC |    38.26    | 63.01 | 60.60 | 57.86 | 53.08 | 48.06 | 37.82 | **30.22** | 18.64 | 10.53 | 2.80 |
| + CMGA |    38.57    | 63.25 | **61.30** | **58.36** | 53.18 | 46.86 | **39.50** | 29.23 | 19.94 | 11.16 | 2.96 |
| + CMSF |    38.38    | 61.93 | 59.86 | 57.44 | 54.41 | 47.02 | 37.43 | **30.07** | **21.09** | 10.61 | **3.98** |
| DC + CMGA |  **38.83**  | **64.51** | **61.94** | 58.07 | **55.14** | **48.60** | **39.58** | 29.84 | 17.78 | 9.95 | 2.89 |
| DC + CMSF |    38.51    | 63.70 | 61.17 | **58.16** | 53.56 | 48.30 | 36.84 | 29.13 | 19.83 | **11.63** | 2.73 |
| CMGA + CMSF |    38.78    | **63.83** | 61.31 | 57.94 | 53.55 | 48.50 | 39.39 | 29.48 | 19.24 | 11.40 | 3.11 |
| Full (DC+CMGA+CMSF) |  **39.01**  | 63.30 | 60.98 | 58.12 | **55.12** | **49.86** | 39.17 | 29.12 | **20.05** | 11.10 | **3.31** |

This detailed per‑threshold comparison further reveals the complementary behavior of the three proposed modules. DC+CMGA achieves the highest AP50 (64.51%) and AP75 (39.58%), demonstrating its strength in moderate‑quality localization. However, the full model, which further incorporates CMSF, attains the best performance at higher IoU thresholds (AP65: 55.12%, AP70: 49.86%) and the highest mAP50‑95 (39.01%). This indicates that CMSF prioritizes overall performance across varying localization quality requirements rather than optimizing for a single threshold. Overall, the three modules contribute differently, and their combination yields the most balanced detection capability.


### Detailed AP Scores at All IoU Thresholds (AIC2026) — Modality Analysis

The table below presents the detailed AP scores across 10 IoU thresholds from 0.50 to 0.95 for each modality configuration. 
These results supplement **Table 7** in the paper.

| Configuration | **mAP50-95 (primary metric)**  | AP50 | AP55 | AP60 | AP65 | AP70 | AP75 | AP80 | AP85 | AP90 | AP95 |
| :--- |:---------:| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| RGB |   36.84   | 60.63 | 58.32 | 55.37 | 51.13 | 45.03 | 37.46 | 28.84 | 19.58 | 10.69 | 1.38 |
| IR |   21.93   | 38.97 | 35.80 | 32.43 | 29.06 | 24.78 | 21.29 | 15.97 | 11.63 | 7.66 | 1.69 |
| Depth |   21.03   | 35.39 | 33.81 | 31.58 | 29.03 | 24.33 | 20.41 | 15.98 | 12.00 | 5.38 | 2.37 |
| RGB + IR |   38.63   | **63.32** | **61.44** | **58.53** | 54.51 | 48.49 | **40.50** | 29.11 | 19.10 | 8.96 | 2.38 |
| RGB + Depth |   38.09   | 63.13 | 60.77 | 57.25 | 52.32 | 46.28 | 39.85 | **30.50** | 18.78 | 9.18 | 2.87 |
| RGB + IR + Depth | **39.01** | 63.30 | 60.98 | 58.12 | **55.12** | **49.86** | 39.17 | 29.12 | **20.05** | **11.10** | **3.31** |

The per‑threshold results provide a more nuanced view of modality contributions. RGB+IR achieves the highest AP50 (63.32%), AP55 (61.44%), AP60 (58.53%), and AP75 (40.50%), indicating that thermal information from IR effectively complements RGB texture for most targets. RGB+Depth shows a notable advantage at AP80 (30.50%), suggesting that geometric cues from Depth may offer benefits under stricter localization criteria. The tri‑modal configuration achieves the best results at AP65 (55.12%), AP70 (49.86%), AP85 (20.05%), AP90 (11.10%), and AP95 (3.31%). Overall, the tri-modal configuration achieves the highest mAP50‑95 (39.01%), indicating complementary benefits from the three modalities.

## 📧 Contact
For questions or issues, please open an issue or contact the authors.


## 🙏 Acknowledgements
We thank the following open-source projects and datasets for their valuable contributions:

[1] **(Ultralytics YOLO)** [Ultralytics YOLO Framework](https://github.com/ultralytics/ultralytics)

[2] **(DEYOLO)** [Dual-Feature-Enhancement YOLO for Cross-Modality Object Detection](https://github.com/chips96/DEYOLO)

[3] **(M3FD Dataset)** [Target-aware Dual Adversarial Learning and a Multi-scenario Multi-Modality Benchmark to Fuse Infrared and Visible for Object Detection](https://github.com/JinyuanLiu-CV/TarDAL)

[4] **(AIC2026 Dataset)** [Global Campus Artificial Intelligence Algorithm Elite Competition](https://www.aicomp.cn/tracks/3633.html)
