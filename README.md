# Facial Emotion Recognition: Conventional vs. Deep Learning Approaches

**Paper ID**: 27738  
**Journal**: TELKOMNIKA  
**Status**: Under Review (Minor Revisions)

---

## 📋 Overview

This repository contains the complete implementation and experimental code for the paper:

> **"A Comparative Study of Conventional vs. Deep Learning Approaches for Facial Emotion Recognition"**

The paper presents a comprehensive comparison between:
- **Conventional Methods**: Local Binary Pattern (LBP) + Support Vector Machine (SVM)
- **Deep Learning Methods**: Convolutional Neural Networks (CNN)

Experiments are conducted on the **FER2013** dataset with detailed performance analysis.

---

## 🎯 Key Results

| Method | Accuracy | F1-Score | Training Time | Inference Time |
|--------|----------|----------|---------------|----------------|
| **LBP + SVM** | 27.96% | 21.48% | 160.9s | 28.76s |
| **Basic CNN** | 62.16% | 61.60% | 1517.8s | 1.61s |
| **Improvement** | **+34.20 pp** | **+40.12 pp** | ×9.4 longer | ×17.8 faster |

---

## 📁 Repository Structure

```
.
├── lbp_svm_classifier.py          # LBP+SVM implementation
├── deep_learning_cnn.py           # Deep Learning CNN implementation
├── train_and_evaluate.py          # Complete training pipeline
├── generate_figures.py            # Generate publication figures
├── generate_exp_figures.py        # Generate experimental result figures
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── results/
│   ├── experiment_results.json    # Raw experimental metrics
│   ├── experiment_report.txt      # Text report
│   ├── fig_confusion_matrices.png # Confusion matrices
│   ├── fig_cnn_training_curves.png # Training curves
│   └── fig_per_class_f1.png       # Per-class F1-scores
└── data/
    └── fer2013/
        └── fer2013.csv            # FER2013 dataset (download required)
```

---

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/fer-comparative-study.git
cd fer-comparative-study
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download FER2013 Dataset
The FER2013 dataset is too large to include in the repository. Follow these steps:

#### Option A: Download from Kaggle (Recommended)
```bash
# 1. Install Kaggle CLI
pip install kaggle

# 2. Set up Kaggle API credentials
# - Go to https://www.kaggle.com/settings/account
# - Click "Create New API Token"
# - Place kaggle.json in ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# 3. Download dataset
kaggle datasets download -d deadskull7/fer2013
unzip fer2013.zip
mkdir -p data/fer2013
mv fer2013.csv data/fer2013/
```

#### Option B: Manual Download
1. Visit [Kaggle FER2013 Dataset](https://www.kaggle.com/datasets/deadskull7/fer2013)
2. Download `fer2013.csv`
3. Create directory: `mkdir -p data/fer2013`
4. Place file: `mv fer2013.csv data/fer2013/`

#### Option C: Alternative Sources
- **Google Drive**: [FER2013 Mirror](https://drive.google.com/file/d/1example/view?usp=sharing)
- **Academic Dataset**: Available through institutional access

### 4. Run Experiments
```bash
# Run complete pipeline
python train_and_evaluate.py

# Or run individual experiments
python -c "from lbp_svm_classifier import LBPSVMClassifier; print('LBP+SVM module loaded')"
python -c "from deep_learning_cnn import BasicCNNModel; print('CNN module loaded')"
```

### 5. Generate Figures
```bash
python generate_figures.py        # Conceptual figures
python generate_exp_figures.py    # Experimental result figures
```

---

## 📊 Dataset Information

### FER2013 Dataset
- **Total Samples**: 35,887 images
- **Resolution**: 48×48 pixels (grayscale)
- **Emotion Classes**: 7 (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral)
- **Train/Test Split**: 28,709 train / 7,178 test
- **License**: Public Domain
- **Size**: ~300 MB (CSV format)

### Class Distribution
```
Angry:    3,995 (11.1%)
Disgust:    436 (1.2%)
Fear:     4,097 (11.4%)
Happy:    7,215 (20.1%)
Sad:      6,077 (16.9%)
Surprise: 4,002 (11.2%)
Neutral:  6,198 (17.3%)
```

---

## 💻 System Requirements

### Minimum
- **CPU**: 4-core processor
- **RAM**: 8 GB
- **Storage**: 500 MB (dataset + outputs)
- **Python**: 3.8+

### Recommended
- **CPU**: 8-core processor or GPU
- **RAM**: 16 GB
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended)
- **Python**: 3.10+

---

## 📦 Dependencies

All dependencies are listed in `requirements.txt`:

```
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.1
scikit-image==0.21.0
tensorflow==2.13.0
keras==2.13.1
matplotlib==3.7.2
seaborn==0.12.2
opencv-python==4.8.0.76
```

Install with:
```bash
pip install -r requirements.txt
```

---

## 🔬 Module Documentation

### LBP+SVM Classifier (`lbp_svm_classifier.py`)

**Main Classes**:
- `LBPFeatureExtractor`: Extracts LBP features from images
- `LBPSVMClassifier`: Complete LBP+SVM pipeline

**Usage Example**:
```python
from lbp_svm_classifier import LBPSVMClassifier
import numpy as np

# Initialize classifier
classifier = LBPSVMClassifier(
    lbp_params={'radius': 1, 'n_points': 8, 'n_bins': 256},
    svm_params={'kernel': 'rbf', 'C': 10}
)

# Train
classifier.fit(X_train, y_train)

# Evaluate
results = classifier.evaluate(X_test, y_test)

# Save model
classifier.save('lbp_svm_model.pkl')
```

### Deep Learning CNN (`deep_learning_cnn.py`)

**Main Class**:
- `BasicCNNModel`: CNN architecture and training pipeline

**Usage Example**:
```python
from deep_learning_cnn import BasicCNNModel
import numpy as np

# Initialize model
model = BasicCNNModel(input_shape=(48, 48, 1), n_classes=7)

# Build and compile
model.build_model()
model.compile_model(optimizer='adam', learning_rate=1e-3)

# Train
model.train(X_train, y_train, epochs=30, batch_size=64)

# Evaluate
results = model.evaluate(X_test, y_test)

# Save model
model.save('cnn_model.h5')
```

### Complete Pipeline (`train_and_evaluate.py`)

**Main Class**:
- `ComparisonExperiment`: Orchestrates both experiments and comparison

**Usage**:
```bash
python train_and_evaluate.py
```

---

## 📈 Expected Output

### Console Output
```
Loading FER2013 Dataset
============================================================
Reading CSV from data/fer2013/fer2013.csv...
  Total samples: 35887
  Train samples: 28709
  Test samples: 7178

============================================================
EXPERIMENT 1: LBP + SVM (Conventional Method)
============================================================
LBP+SVM Training Pipeline
...
Results:
  Accuracy:  27.96%
  F1-Score:  21.48%

============================================================
EXPERIMENT 2: Deep Learning CNN
============================================================
Deep Learning CNN Training Pipeline
...
Results:
  Accuracy:  62.16%
  F1-Score:  61.60%

============================================================
COMPARATIVE ANALYSIS
============================================================
CNN Accuracy Improvement: +34.20 percentage points
```

### Generated Files
- `results/experiment_results.json` - Raw metrics
- `results/experiment_report.txt` - Text report
- `results/fig_confusion_matrices.png` - Confusion matrices
- `results/fig_cnn_training_curves.png` - Training curves
- `results/fig_per_class_f1.png` - Per-class metrics

---

## 🔧 Configuration

### LBP Parameters
Edit in `lbp_svm_classifier.py`:
```python
lbp_params = {
    'radius': 1,           # LBP neighborhood radius
    'n_points': 8,         # Number of neighborhood points
    'n_bins': 256,         # Histogram bins
    'method': 'uniform'    # 'uniform', 'nri_uniform', 'var'
}
```

### SVM Parameters
Edit in `lbp_svm_classifier.py`:
```python
svm_params = {
    'kernel': 'rbf',       # 'linear', 'rbf', 'poly'
    'C': 1.0,              # Regularization parameter (default: 1.0)
    'gamma': 'scale'       # 'scale', 'auto', or float
}
```

### CNN Architecture
Edit in `deep_learning_cnn.py`:
```python
# Modify build_model() method:
# - Change number of conv filters: 32, 64, 128
# - Change FC layer sizes: 512, 128
# - Adjust dropout rates: 0.25, 0.5
# - Change activation functions: 'relu', 'elu', etc.
```

### Training Parameters
Edit in `train_and_evaluate.py`:
```python
cnn.train(
    X_train_norm, y_train,
    validation_split=0.1,    # Validation split
    epochs=30,               # Number of epochs
    batch_size=64            # Batch size
)
```

---

## 🐛 Troubleshooting

### Issue: "No module named 'tensorflow'"
```bash
pip install tensorflow
# For GPU support:
pip install tensorflow-gpu
```

### Issue: "FER2013 dataset not found"
```bash
# Verify dataset location
ls -la data/fer2013/fer2013.csv

# If missing, download from Kaggle (see Quick Start section 3)
```

### Issue: "CUDA out of memory"
```python
# Reduce batch size in train_and_evaluate.py:
batch_size=32  # Instead of 64

# Or use CPU only:
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

### Issue: "Slow training on CPU"
```bash
# Install GPU support
pip install tensorflow-gpu

# Or use Google Colab with GPU:
# https://colab.research.google.com/
```

---

## 📊 Reproducing Results

### Step-by-Step Reproduction
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download FER2013 (see section 3 above)
mkdir -p data/fer2013
# Place fer2013.csv in data/fer2013/

# 3. Run experiments
python train_and_evaluate.py

# 4. Expected runtime:
# - LBP+SVM: ~3-5 minutes
# - CNN: ~30-40 minutes
# - Total: ~45-50 minutes (CPU)
# - Total: ~10-15 minutes (GPU)

# 5. Check results
cat results/experiment_report.txt
```

### Verifying Results
```python
import json

# Load results
with open('results/experiment_results.json') as f:
    results = json.load(f)

# Check LBP+SVM
lbp_acc = results['lbp_svm']['evaluation']['accuracy']
print(f"LBP+SVM Accuracy: {lbp_acc*100:.2f}%")  # Expected: ~27.96%

# Check CNN
cnn_acc = results['deep_learning_cnn']['evaluation']['accuracy']
print(f"CNN Accuracy: {cnn_acc*100:.2f}%")      # Expected: ~62.16%
```

---

## 📚 References

### Key Papers
1. Goodfellow et al. (2013) - FER2013 Dataset
2. Ojala et al. (2002) - LBP Features
3. LeCun et al. (1998) - CNN Fundamentals
4. Krizhevsky et al. (2012) - Deep Learning for Image Recognition

### Datasets
- **FER2013**: https://www.kaggle.com/datasets/deadskull7/fer2013
- **CK+**: https://www.jeffcohn.com/databases/
- **AffectNet**: http://mohammadmahoor.com/affectnet/

### Tools & Frameworks
- **TensorFlow**: https://www.tensorflow.org/
- **scikit-learn**: https://scikit-learn.org/
- **OpenCV**: https://opencv.org/

---

## 📄 Citation

If you use this code or dataset in your research, please cite:

```bibtex
@article{paper27738,
  title={A Comparative Study of Conventional vs. Deep Learning Approaches for Facial Emotion Recognition},
  journal={TELKOMNIKA},
  year={2026},
  note={Paper ID: 27738},
  author={Your Name}
}
```

---

## 📞 Support & Contact

For issues, questions, or contributions:

1. **GitHub Issues**: Open an issue on the repository
2. **Email**: your.email@institution.edu
3. **Discussion**: Start a discussion in the repository

---

## 📝 License

This project is licensed under the **MIT License** - see LICENSE file for details.

---

## 🙏 Acknowledgments

- **FER2013 Dataset**: Ian Goodfellow et al.
- **TELKOMNIKA Journal**: For publication opportunity
- **Research Community**: For feedback and support

---

**Last Updated**: 2026-03-25  
**Status**: Production Ready  
**Version**: 1.0
