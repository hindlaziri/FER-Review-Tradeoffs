# Experimental Comparative Analysis of Conventional vs. Deep Learning Models for Facial Emotion Recognition

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Journal: SN Computer Science](https://img.shields.io/badge/Journal-SN_Computer_Science-green.svg)](https://www.springer.com/journal/42979)

This repository contains the official implementation and manuscript for our study on **Facial Emotion Recognition (FER)**. We provide a rigorous comparative analysis across three distinct paradigms: conventional handcrafted features (LBP+SVM), baseline deep learning (CNN), and state-of-the-art optimized architectures (EfficientNet-B0).

## 🚀 Key Contributions
- **Multi-Paradigm Benchmarking**: Systematic comparison of three different FER approaches on the FER-2013 dataset.
- **SOTA Integration**: Implementation of an optimized EfficientNet-B0 pipeline achieving **72.45% accuracy**.
- **Efficiency Analysis**: Detailed evaluation of the trade-offs between accuracy, F1-score, and real-time inference latency.
- **Reproducibility**: Complete LaTeX source and PyTorch code provided for full transparency.

## 📊 Experimental Results

Our findings demonstrate the clear superiority of optimized deep architectures for complex affective computing tasks.

![FER Pipeline](https://private-us-east-1.manuscdn.com/sessionFile/OHSWHhhm4aHxSkziZCMQd3/sandbox/jmdNfmShtlDPWaxrYCHCZo-images_1781275807770_na1fn_L2hvbWUvdWJ1bnR1L0ZFUl9GaW5hbF9HaXRIdWJfUmVwby9hc3NldHMvaW1hZ2VzL2Zlcl9waXBlbGluZQ.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvT0hTV0hoaG00YUh4U2t6aVpDTVFkMy9zYW5kYm94L2ptZE5mbVNodGxEUFdheHJZQ0hDWm8taW1hZ2VzXzE3ODEyNzU4MDc3NzBfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwwWkZVbDlHYVc1aGJGOUhhWFJJZFdKZlVtVndieTloYzNObGRITXZhVzFoWjJWekwyWmxjbDl3YVhCbGJHbHVaUS5wbmciLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3OTg3NjE2MDB9fX1dfQ__&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=VlTf6PIXjF6qeEBYOybPdjkRJPe1EoGa9mGbXhRWClT1qVYfErI1Mo~HLcJMpUcEFKWdr3zdmWoqzw9auWYOvqnu2XmPUzpvfxUjEEuMwj3MyH4ABpw-Xe~-Kw98j1TOzIPo~CuY39E9-D~Ns88PtpFUm1oNGW3oE1y1k-PHoIwl6mX8zu4eWdzVhPdPP4RK9mnq3CtxjiVSnbUuZsFs1-yZpYg-hRw9ACzYQ8he1sYB~DrBhn7p5yVkwAHEeIJlqMiKNnrKu~wrOd3c6m0qPa3sN~Vh6XFjLUTA68tklM1lXg7xYt~DnNpvsUbUCb~mrqgzbg5Bx6Uzix5h9KMTiQ__)
*Figure 1: Overview of the proposed Facial Emotion Recognition pipeline.*

| Methodology | Accuracy (%) | Weighted F1 (%) | Training Time (s) | Inference Time (s) |
| :--- | :---: | :---: | :---: | :---: |
| **LBP + SVM** | 27.96 | 21.48 | **124.2** | **0.85** |
| **Basic CNN** | 62.16 | 61.34 | 845.3 | 1.12 |
| **EfficientNet-B0** | **72.45** | **71.92** | 3420.5 | 2.45 |

> *Note: Inference time measured on the full test set (~7,000 images).*

## 🛠️ Project Structure
```text
.
├── src/                    # PyTorch implementation
│   ├── sota_model_pytorch.py  # EfficientNet-B0 & Data Pipeline
│   └── integrate_sota.py      # Training & Evaluation script
├── manuscript/             # Academic Paper
│   ├── latex/                 # LaTeX source files (.tex, .bib)
│   └── pdf/                   # Final compiled manuscript
├── assets/                 # Visual Materials
│   ├── images/                # Figures (Pipeline, Architecture)
│   └── styles/                # Springer Nature templates
└── requirements.txt        # Environment dependencies
```

## 💻 Getting Started

### Prerequisites
- Python 3.8+
- CUDA-enabled GPU (recommended for EfficientNet)

### Installation
```bash
git clone https://github.com/yourusername/FER-Comparative-Analysis.git
cd FER-Comparative-Analysis
pip install -r requirements.txt
```

### Usage
To reproduce the SOTA results and generate the evaluation metrics:
```bash
python src/integrate_sota.py
```

## 📖 Citation


```bibtex
@article{laziri2026experimental,
  title={Experimental Comparative Analysis of Conventional Feature Extraction vs. Deep Learning Models for Facial Emotion Recognition},
  author={Laziri, Hind and Riffi, Mohammed Essaid},
  journal={SN Computer Science},
  volume={X},
  number={X},
  pages={XXX--XXX},
  year={2026},
  publisher={Springer}
}
```

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
