# VectorML Core: Machine Learning & Numerical Optimization from First Principles

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A production-grade, vectorized implementation of core machine learning algorithms and mathematical optimization routines built in pure NumPy without high-level ML framework dependencies.

## Key Modules
- **`VectorKNNRegressor`**: Vectorized Euclidean distance calculation using algebraic matrix decomposition ($||a - b||^2 = ||a||^2 + ||b||^2 - 2a^T b$) and $k$-NN continuous aggregation.
- **`VectorNeuron`**: Single-neuron artificial perceptron with numerical stability clipping, forward sigmoid propagation, and analytical backpropagation.
- **`GradientOptimizer`**: Momentum-accelerated gradient descent optimizer.
- **Interactive Notebooks**: Complete derivation notebooks for multivariate calculus, probability distributions, and object-oriented ML design.

## Quickstart

```bash
git clone https://github.com/KhotChaitanya/vectorml-core.git
cd vectorml-core
pip install -r requirements.txt
python main.py
```

## Running the Unit Tests

```bash
pytest
```
