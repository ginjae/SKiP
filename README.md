# SKiP - SVM with K-nearest neighbor and Probabilistic weighting

Robust SVM classifier implementation with noise-resistant weighting schemes.

## Environment Setup

### Create Conda Environment with Python 3.10

```bash
conda create -n skip python=3.10
conda activate skip
```

### Install Required Packages

```bash
pip install numpy scikit-learn matplotlib cvxpy jupyter
```

## Usage

Run the Jupyter Notebook to see examples:

```bash
jupyter notebook Naive-SVM.ipynb
```

## Files

- `svm_models.py`: NaiveSVM, ProbSVM, KNNSVM, SKiP implementations
- `datasets.py`: Noise injection utilities
- `utils.py`: Visualization functions
- `Naive-SVM.ipynb`: Demo notebook
