"""VectorML CLI Demonstration and Benchmark Runner."""
import numpy as np
from src.vectorml import VectorKNNRegressor, VectorNeuron

def main():
    print("=" * 60)
    print(" VectorML Core: Mathematical Foundations Benchmark")
    print("=" * 60)
    
    # 1. KNN Regressor Benchmark
    X_train = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y_train = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    X_test = np.array([[1.5], [3.5], [6.0]])
    
    knn = VectorKNNRegressor(n_neighbors=2)
    knn.fit(X_train, y_train)
    preds = knn.predict(X_test)
    print("\n[1] VectorKNNRegressor Benchmark:")
    print(f"  Training Points: X={X_train.ravel().tolist()}, y={y_train.tolist()}")
    print(f"  Test Inputs:     {X_test.ravel().tolist()}")
    print(f"  Predictions:     {preds.tolist()}")
    
    # 2. VectorNeuron Benchmark
    X_neuron = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
    y_neuron = np.array([0, 0, 0, 1], dtype=np.float64)
    neuron = VectorNeuron(input_dim=2, learning_rate=0.5)
    for epoch in range(100):
        loss = neuron.train_step(X_neuron, y_neuron)
    print("\n[2] VectorNeuron (AND Gate Convergence):")
    print(f"  Epoch 100 Loss:  {loss:.4f}")
    print(f"  Outputs:         {np.round(neuron.forward(X_neuron), 3).tolist()}")
    print("\n[SUCCESS] VectorML benchmarks completed successfully.")

if __name__ == "__main__":
    main()
