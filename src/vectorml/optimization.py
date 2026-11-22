"""Numerical optimization and gradient descent algorithms."""
import numpy as np

class GradientOptimizer:
    """Numerical gradient descent optimizer with momentum."""
    def __init__(self, lr: float = 0.05, momentum: float = 0.9):
        self.lr = lr
        self.momentum = momentum
        self.velocity = 0.0

    def step(self, grad: float) -> float:
        self.velocity = self.momentum * self.velocity + self.lr * grad
        return self.velocity
