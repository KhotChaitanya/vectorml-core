"""Vectorized K-Nearest Neighbors Regressor implemented from mathematical first principles."""
import numpy as np

class VectorKNNRegressor:
    """K-Nearest Neighbors Regressor utilizing vectorized Euclidean distance computations.
    
    Parameters
    ----------
    n_neighbors : int, default=3
        Number of nearest neighbors to consider for continuous target aggregation.
    """
    def __init__(self, n_neighbors: int = 3):
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit training points into the regressor memory."""
        self.X_train = np.asarray(X, dtype=np.float64)
        self.y_train = np.asarray(y, dtype=np.float64)
        return self

    def _euclidean_distances(self, X_test: np.ndarray) -> np.ndarray:
        """Compute vectorized pair-wise Euclidean distances between test samples and training samples.
        Uses formulation: ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * a . b
        """
        X_test = np.asarray(X_test, dtype=np.float64)
        dists = np.sqrt(
            np.sum(X_test**2, axis=1, keepdims=True) +
            np.sum(self.X_train**2, axis=1, keepdims=True).T -
            2 * np.dot(X_test, self.X_train.T)
        )
        return np.maximum(dists, 0.0)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict target values by averaging the k nearest neighbor targets."""
        dists = self._euclidean_distances(X_test)
        predictions = []
        for row in dists:
            knn_indices = np.argsort(row)[:self.n_neighbors]
            pred = np.mean(self.y_train[knn_indices])
            predictions.append(pred)
        return np.array(predictions)
