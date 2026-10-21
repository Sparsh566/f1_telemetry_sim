"""
tests/test_backpropagation.py
=============================
Unit tests for forward propagation, analytical backpropagation,
finite-difference gradient checks, and training convergence.
"""

import numpy as np
from ml.neural_network_numpy import (
    DenseLayer,
    NeuralNetwork,
    check_network_gradients
)
from ml.optimizers import AdamOptimizer


def test_dense_layer_forward_backward_shapes():
    layer = DenseLayer(in_features=4, out_features=2, activation_name="ReLU", seed=42)
    X = np.random.randn(5, 4)

    A = layer.forward(X)
    assert A.shape == (5, 2)

    dA = np.ones((5, 2))
    dX = layer.backward(dA)

    assert dX.shape == (5, 4)
    assert layer.dW.shape == (4, 2)
    assert layer.db.shape == (1, 2)


def test_analytical_backprop_matches_finite_differences():
    # Construct small test network for precise numerical gradient checking
    net = NeuralNetwork(layer_sizes=[3, 4, 1], activations=["Tanh", "Linear"], seed=123)
    X = np.random.randn(8, 3)
    y = np.random.randn(8, 1)

    is_verified, max_error = check_network_gradients(net, X, y, eps=1e-5)
    assert is_verified, f"Backpropagation failed finite difference check (max_rel_error={max_error})"


def test_neural_network_training_reduces_loss():
    # Simple linear relationship: y = 2*x1 - 3*x2 + 1
    rng = np.random.RandomState(42)
    X = rng.randn(100, 2)
    y = 2.0 * X[:, 0] - 3.0 * X[:, 1] + 1.0

    net = NeuralNetwork(layer_sizes=[2, 8, 1], activations=["ReLU", "Linear"], seed=42)
    hist = net.fit(X, y, epochs=50, batch_size=16, optimizer=AdamOptimizer(lr=0.03))

    initial_loss = hist["train_loss"][0]
    final_loss = hist["train_loss"][-1]

    assert final_loss < initial_loss, f"Training failed to reduce loss (initial={initial_loss}, final={final_loss})"
    assert final_loss < 0.5, f"Expected final loss < 0.5, got {final_loss}"
