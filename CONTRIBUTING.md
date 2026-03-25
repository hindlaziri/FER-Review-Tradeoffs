# Contributing to FER Comparative Study

Thank you for your interest in contributing to this project!

## How to Contribute

### 1. Report Bugs
If you find a bug, please open an issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs. actual behavior
- Python version and environment details

### 2. Suggest Enhancements
Enhancement suggestions are welcome! Please include:
- Clear description of the enhancement
- Use cases and benefits
- Possible implementation approach

### 3. Submit Code Changes
To submit code changes:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes and commit: `git commit -m 'Add your feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

### Code Style
- Follow PEP 8 conventions
- Use descriptive variable and function names
- Add docstrings to all functions and classes
- Include comments for complex logic

### Testing
Before submitting a PR:
- Test your changes locally
- Verify all existing tests still pass
- Add new tests for new functionality

## Development Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/fer-comparative-study.git
cd fer-comparative-study

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run experiments
python code/train_and_evaluate.py
```

## Citation
If you use this code in your research, please cite:

```bibtex
@article{paper27738,
  title={A Comparative Study of Conventional vs. Deep Learning Approaches for Facial Emotion Recognition},
  journal={TELKOMNIKA},
  year={2026},
  note={Paper ID: 27738}
}
```

## Questions?
Feel free to open an issue or contact the maintainers.

Thank you for contributing! 🙏
