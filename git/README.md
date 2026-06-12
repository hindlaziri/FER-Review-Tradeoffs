# Facial Emotion Recognition (FER): A Multi-Criteria Trade-off Analysis

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official implementation and experimental framework for the paper: **"Experimental Comparative Analysis of Conventional Feature Extraction vs. Deep Learning Models for Facial Emotion Recognition"** (Submitted to *Multimedia Tools and Applications*).

## 🚀 Overview

This project introduces a **Decision-Making Framework for FER Deployment**, which systematically quantifies the trade-offs between discriminative power and computational efficiency. We evaluate models ranging from handcrafted feature extractors to compound-scaled deep architectures, providing a practical engineering tool for selecting FER models based on specific hardware constraints.

## 📊 Key Results (FER-2013)

| Model | Accuracy (%) | Weighted F1 (%) | Inference Time (s) |
| :--- | :---: | :---: | :---: |
| LBP + SVM | 27.96 ± 0.12 | 21.48 ± 0.15 | 28.76 |
| Basic CNN | 62.16 ± 0.45 | 61.34 ± 0.38 | 1.61 |
| **EfficientNet-B0** | **72.45 ± 0.32** | **71.92 ± 0.28** | **2.45** |
| **MobileViT-XXS** | **75.12 ± 0.25** | **74.85 ± 0.22** | **4.12** |

*Note: Results represent mean ± standard deviation over 5 independent runs. Statistical significance confirmed via Student's t-test (p < 0.001).*

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/hindlaziri/FER-Review-Tradeoffs.git
cd FER-Review-Tradeoffs

# Install dependencies
pip install -r requirements.txt
```

## 💻 Usage

### 1. Training and Evaluation
To reproduce the comparative analysis:
```bash
python integrate_sota.py
```

### 2. Generate Visualizations
To regenerate the confusion matrices and F1-score plots:
```bash
python fix_figures_and_add_vit.py
```

## 📂 Repository Structure
- `sota_model_pytorch.py`: Core model implementations (EfficientNet, MobileViT).
- `integrate_sota.py`: Main execution script for training and benchmarking.
- `fix_figures_and_add_vit.py`: Visualization utilities.
- `fer_pipeline.png`: Visual representation of the preprocessing stage.
- `fig_confusion_matrices.png`: Comparative confusion matrices.
- `fig_per_class_f1.png`: Per-class performance analysis.

## 📜 Citation
If you find this work useful, please cite:
```bibtex
@article{laziri2026fer,
  title={Experimental Comparative Analysis of Conventional Feature Extraction vs. Deep Learning Models for Facial Emotion Recognition},
  author={Laziri, Hind and Riffi, Mohammed Essaid},
  journal={Multimedia Tools and Applications},
  year={2026}
}
```
