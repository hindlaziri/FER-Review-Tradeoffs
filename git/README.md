# Facial Emotion Recognition: Conventional vs. Deep Learning Analysis

This repository contains the complete source code and manuscript for the comparative study of Facial Emotion Recognition (FER) techniques, ranging from conventional feature extraction to state-of-the-art deep learning architectures.

## Project Overview
This study evaluates three distinct paradigms on the FER-2013 dataset:
1. **Conventional**: LBP + SVM
2. **Baseline Deep Learning**: Basic CNN
3. **State-of-the-Art**: EfficientNet-B0

## Repository Structure
- `src/`: PyTorch implementation of the models and evaluation pipeline.
- `manuscript/`: 
  - `latex/`: LaTeX source files and bibliography.
  - `pdf/`: Final compiled manuscript.
- `assets/`:
  - `images/`: Figures used in the paper.
  - `styles/`: Springer Nature LaTeX class and BibTeX styles.
- `requirements.txt`: Python dependencies.

## Results Summary
| Model | Accuracy | Weighted F1 |
| :--- | :---: | :---: |
| LBP + SVM | 27.96% | 21.48% |
| Basic CNN | 62.16% | 61.34% |
| **EfficientNet-B0** | **72.45%** | **71.92%** |

## Getting Started
1. Install dependencies: `pip install -r requirements.txt`
2. Run evaluation: `python src/integrate_sota.py`

## Citation
Please cite our work if you use this code:
```bibtex
@article{laziri2026experimental,
  title={Experimental Comparative Analysis of Conventional Feature Extraction vs. Deep Learning Models for Facial Emotion Recognition},
  author={Laziri, Hind and Riffi, Mohammed Essaid},
  journal={SN Computer Science},
  year={2026}
}
```
