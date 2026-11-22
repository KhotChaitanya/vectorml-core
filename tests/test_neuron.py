import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import unittest
import numpy as np
from src.vectorml import VectorNeuron

class TestVectorNeuron(unittest.TestCase):
    def test_sigmoid_numerical_stability(self):
        neuron = VectorNeuron(input_dim=2)
        z = np.array([-1000.0, 0.0, 1000.0])
        sig = neuron.sigmoid(z)
        self.assertGreaterEqual(sig[0], 0.0)
        self.assertEqual(sig[1], 0.5)
        self.assertLessEqual(sig[2], 1.0)

    def test_neuron_and_gate_learning(self):
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
        y = np.array([0, 0, 0, 1], dtype=np.float64)
        neuron = VectorNeuron(input_dim=2, learning_rate=0.8)
        initial_loss = neuron.train_step(X, y)
        for _ in range(150):
            final_loss = neuron.train_step(X, y)
        self.assertLess(final_loss, initial_loss)

if __name__ == "__main__":
    unittest.main()
