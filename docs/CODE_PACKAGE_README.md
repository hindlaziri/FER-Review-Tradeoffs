# Code Package for FER Manuscript (Paper ID: 27738)
## "A Comparative Study of Conventional vs. Deep Learning Approaches for Facial Emotion Recognition"

---

## 📁 File Structure

```
/home/ubuntu/
├── run_experiments.py              # Main experiment: LBP+SVM vs Basic CNN on FER2013
├── generate_figures.py             # Generate 3 conceptual figures (pipeline, CNN arch, trade-off)
├── generate_exp_figures.py         # Generate 4 experimental result figures
├── update_refs.py                  # Update and fix bibliography references
├── apply_final_fixes.py            # Apply final manuscript refinements
├── convert_to_telkomnika_docx.py   # Convert LaTeX to Word format
├── experiment_results.json         # Output: experimental metrics and results
├── fer2013_data/                   # FER2013 dataset directory
└── [Generated figures and outputs]
```

---

## 🔬 Scripts Description

### 1. **run_experiments.py** (Main Experiment)
**Purpose**: Train and evaluate LBP+SVM and Basic CNN on FER2013

**Key Features**:
- Loads FER2013 dataset (28,709 train + 7,178 test samples)
- Extracts LBP features with 256 bins
- Trains SVM with RBF kernel (C=10)
- Builds and trains Basic CNN (3 conv blocks + 2 FC layers)
- Computes accuracy, F1-score, confusion matrices, per-class metrics
- Saves results to `experiment_results.json`

**Dependencies**:
```bash
pip install numpy pandas scikit-learn scikit-image tensorflow keras
```

**Output**:
- `experiment_results.json`: Complete metrics, confusion matrices, training history

**Runtime**: ~30-40 minutes (depends on hardware)

---

### 2. **generate_figures.py** (Conceptual Figures)
**Purpose**: Generate 3 publication-quality figures for the manuscript

**Figures Generated**:
1. **fer_pipeline.png** - FER processing pipeline flowchart
2. **cnn_architecture.png** - CNN architecture diagram
3. **fer_comparison_chart.png** - Trade-off accuracy vs computational complexity

**Dependencies**:
```bash
pip install matplotlib numpy
```

**Output**:
- 3 PNG files (180 DPI, publication-ready)

**Runtime**: < 5 seconds

---

### 3. **generate_exp_figures.py** (Experimental Results Figures)
**Purpose**: Generate 4 figures from experimental results

**Figures Generated**:
1. **fig_confusion_matrices.png** - Side-by-side confusion matrices (LBP+SVM vs CNN)
2. **fig_cnn_training_curves.png** - Training/validation accuracy and loss curves
3. **fig_per_class_f1.png** - Per-class F1-score comparison bar chart
4. **fig_comparison_table.png** - Summary comparison table (visual)

**Dependencies**:
```bash
pip install matplotlib numpy seaborn
```

**Input**: `experiment_results.json` (from run_experiments.py)

**Output**:
- 4 PNG files (150 DPI, publication-ready)

**Runtime**: < 10 seconds

---

### 4. **update_refs.py** (Bibliography Management)
**Purpose**: Update and standardize references in LaTeX

**Features**:
- Replaces 8 conference/arXiv references with journal articles (2024-2026)
- Fixes title casing to IEEE sentence case
- Ensures sequential citation numbering [1]-[25]
- Removes duplicate references

**Dependencies**: None (regex-based)

**Input**: `27738-71535-1-SP_revised.tex`

**Output**: Updated LaTeX file with corrected bibliography

**Runtime**: < 1 second

---

### 5. **apply_final_fixes.py** (Manuscript Refinement)
**Purpose**: Apply final quality improvements to LaTeX

**Fixes Applied**:
1. Remove duplicate "REFERENCES" section
2. Add CNN performance explanation (62% vs 72-83% literature)
3. Strengthen experimental limitation statements
4. Add scientific justification for CNN superiority
5. Deepen dataset bias discussion
6. Improve review methodology description
7. Polish academic language (remove subjective terms)
8. Verify figure references and captions
9. Ensure IEEE reference format consistency

**Dependencies**: None (regex-based)

**Input**: `27738-71535-1-SP_revised.tex`

**Output**: Refined LaTeX file

**Runtime**: < 1 second

---

### 6. **convert_to_telkomnika_docx.py** (Format Conversion)
**Purpose**: Convert LaTeX to TELKOMNIKA-compliant Word format

**Features**:
- Uses Pandoc for LaTeX → DOCX conversion
- Integrates with TELKOMNIKA template
- Preserves formatting and references

**Dependencies**:
```bash
sudo apt-get install pandoc
pip install python-docx
```

**Input**: `27738-71535-1-SP_revised.tex`

**Output**: `27738-TELKOMNIKA-GOLD-STANDARD.docx`

**Runtime**: < 5 seconds

---

## 🚀 Quick Start Guide

### Step 1: Prepare Environment
```bash
# Install dependencies
pip install numpy pandas scikit-learn scikit-image tensorflow keras matplotlib seaborn

# Download FER2013 dataset
# (Requires Kaggle API credentials or manual download)
# Place in: /home/ubuntu/fer2013_data/fer2013/fer2013.csv
```

### Step 2: Run Experiments
```bash
cd /home/ubuntu
python3.11 run_experiments.py
# Output: experiment_results.json
```

### Step 3: Generate Figures
```bash
# Generate conceptual figures
python3.11 generate_figures.py

# Generate experimental result figures
python3.11 generate_exp_figures.py
```

### Step 4: Update Manuscript
```bash
# Update references
python3.11 update_refs.py

# Apply final fixes
python3.11 apply_final_fixes.py

# Convert to Word
python3.11 convert_to_telkomnika_docx.py
```

### Step 5: Compile LaTeX (Optional)
```bash
pdflatex -interaction=nonstopmode 27738-71535-1-SP_revised.tex
```

---

## 📊 Expected Results

### Experimental Metrics (FER2013)
| Method | Accuracy | F1-Score | Train Time | Infer Time |
|--------|----------|----------|-----------|-----------|
| LBP + SVM | 27.96% | 21.48% | 160.9s | 28.76s |
| Basic CNN | 62.16% | 61.60% | 1517.8s | 1.61s |

### CNN Architecture
- Input: 48×48 grayscale images
- 3 Convolutional blocks (32→64→128 filters)
- Max pooling (2×2) after each block
- 2 Fully connected layers (512, 128)
- Output: 7 emotion classes (softmax)
- Total parameters: ~1.2M

---

## 🔧 Customization

### Modify LBP Parameters
Edit `run_experiments.py` line 49:
```python
def extract_lbp_features(images, radius=1, n_points=8, n_bins=256):
    # Adjust radius, n_points, n_bins as needed
```

### Modify CNN Architecture
Edit `run_experiments.py` line 117:
```python
def build_basic_cnn(input_shape=(48, 48, 1), n_classes=7):
    # Add/remove layers, adjust filters, dropout rates
```

### Modify Figure Styles
Edit `generate_figures.py` and `generate_exp_figures.py`:
- Colors: `colors = ['#4B2D8A', ...]`
- Fonts: `fontsize=10`
- DPI: `dpi=180`

---

## 📝 Output Files

### Figures
- `fer_pipeline.png` (7×12 inches, 180 DPI)
- `cnn_architecture.png` (14×6 inches, 180 DPI)
- `fer_comparison_chart.png` (9×6 inches, 180 DPI)
- `fig_confusion_matrices.png` (14×6 inches, 150 DPI)
- `fig_cnn_training_curves.png` (12×4.5 inches, 150 DPI)
- `fig_per_class_f1.png` (10×5 inches, 150 DPI)

### Data
- `experiment_results.json` (~500 KB)

### Manuscript
- `27738-71535-1-SP_revised.tex` (LaTeX source)
- `27738-71535-1-SP_revised.pdf` (Compiled PDF)
- `27738-TELKOMNIKA-GOLD-STANDARD.docx` (Word format)

---

## ⚠️ Requirements & Constraints

### Hardware
- **Minimum**: 8 GB RAM, 4-core CPU
- **Recommended**: 16 GB RAM, GPU (NVIDIA CUDA)
- **Storage**: 500 MB for FER2013 dataset + outputs

### Software
- Python 3.8+
- TensorFlow 2.10+
- scikit-learn 1.0+
- Matplotlib 3.5+

### Dataset
- FER2013 (35,887 images, 7 emotions)
- Download: https://www.kaggle.com/datasets/deadskull7/fer2013
- License: Public domain

---

## 🐛 Troubleshooting

### Issue: "No module named 'tensorflow'"
```bash
pip install tensorflow
```

### Issue: "FER2013 dataset not found"
```bash
# Download from Kaggle and place at:
/home/ubuntu/fer2013_data/fer2013/fer2013.csv
```

### Issue: "Pandoc not found"
```bash
sudo apt-get install pandoc
```

### Issue: "CUDA out of memory"
```python
# In run_experiments.py, reduce batch size:
batch_size=32  # Instead of 64
```

---

## 📚 References

- **FER2013 Dataset**: Goodfellow et al., 2013
- **LBP Features**: Ojala et al., 2002
- **CNN Architecture**: Inspired by VGG, ResNet
- **TELKOMNIKA Journal**: https://telkomnika.uad.ac.id/

---

## 📄 Citation

If you use this code, please cite:
```bibtex
@article{paper27738,
  title={A Comparative Study of Conventional vs. Deep Learning Approaches for Facial Emotion Recognition},
  journal={TELKOMNIKA},
  year={2026},
  note={Paper ID: 27738}
}
```

---

## 📞 Support

For issues or questions:
1. Check this README
2. Review script comments
3. Verify dependencies
4. Check error logs

---

**Generated**: 2026-03-25  
**Version**: 1.0  
**Status**: Production-ready
