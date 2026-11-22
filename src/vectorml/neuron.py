"""Single-Neuron Artificial Perceptron with forward & backward propagation."""
import numpy as np

class VectorNeuron:
    """Single Artificial Neuron with Sigmoid activation and gradient descent updates."""
    def __init__(self, input_dim: int, learning_rate: float = 0.01):
        self.weights = np.random.randn(input_dim) * 0.01
        self.bias = 0.0
        self.learning_rate = learning_rate

    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def forward(self, X: np.ndarray) -> np.ndarray:
        return self.sigmoid(np.dot(X, self.weights) + self.bias)

    def train_step(self, X: np.ndarray, y: np.ndarray) -> float:
        m = X.shape[0]
        y_pred = self.forward(X)
        loss = -np.mean(y * np.log(y_pred + 1e-15) + (1 - y) * np.log(1 - y_pred + 1e-15))
        error = y_pred - y
        dw = (1 / m) * np.dot(X.T, error)
        db = (1 / m) * np.sum(error)
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db
        return float(loss)
