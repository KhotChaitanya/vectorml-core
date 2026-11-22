import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import unittest
import numpy as np
from src.vectorml import VectorKNNRegressor

class TestVectorKNN(unittest.TestCase):
    def test_knn_fit_predict(self):
        X_train = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        y_train = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        X_test = np.array([[1.5], [3.5]])
        
        knn = VectorKNNRegressor(n_neighbors=2)
        knn.fit(X_train, y_train)
        preds = knn.predict(X_test)
        
        self.assertEqual(len(preds), 2)
        self.assertEqual(preds[0], 3.0)
        self.assertEqual(preds[1], 7.0)

    def test_distance_symmetry(self):
        knn = VectorKNNRegressor(n_neighbors=1)
        X = np.array([[0.0, 0.0], [3.0, 4.0]])
        y = np.array([0.0, 5.0])
        knn.fit(X, y)
        dists = knn._euclidean_distances(X)
        self.assertTrue(np.isclose(dists[0, 1], 5.0))
        self.assertTrue(np.isclose(dists[1, 0], 5.0))

if __name__ == "__main__":
    unittest.main()
