"""
tests/test_activations.py
=========================
Unit tests for first-principles activation functions and analytical derivatives.
"""

import numpy as np
from ml.activations import ACTIVATIONS


def test_relu_forward_and_derivative():
    relu = ACTIVATIONS["ReLU"]
    z = np.array([-3.0, -0.5, 0.0, 0.5, 4.0])
    
    # Forward check
    fwd = relu(z)
    np.testing.assert_allclose(fwd, [0.0, 0.0, 0.0, 0.5, 4.0])
    
    # Derivative check
    deriv = relu.derivative(z)
    np.testing.assert_allclose(deriv, [0.0, 0.0, 0.0, 1.0, 1.0])


def test_leaky_relu_forward_and_derivative():
    leaky = ACTIVATIONS["Leaky ReLU"]
    z = np.array([-2.0, 0.0, 2.0])
    
    fwd = leaky(z)
    np.testing.assert_allclose(fwd, [-0.10, 0.0, 2.0])
    
    deriv = leaky.derivative(z)
    np.testing.assert_allclose(deriv, [0.05, 0.05, 1.0])


def test_sigmoid_forward_and_derivative():
    sigmoid = ACTIVATIONS["Sigmoid"]
    z = np.array([0.0])
    
    # At z=0, sigmoid is 0.5, derivative is 0.5 * 0.5 = 0.25
    assert abs(float(sigmoid(z)[0]) - 0.5) < 1e-5
    assert abs(float(sigmoid.derivative(z)[0]) - 0.25) < 1e-5
    
    # Test numerical stability with large magnitude inputs
    z_extreme = np.array([-100.0, 100.0])
    fwd_extreme = sigmoid(z_extreme)
    assert not np.isnan(fwd_extreme).any()
    assert 0.0 <= fwd_extreme[0] <= 1.0
    assert 0.0 <= fwd_extreme[1] <= 1.0


def test_tanh_forward_and_derivative():
    tanh = ACTIVATIONS["Tanh"]
    z = np.array([0.0])
    
    # At z=0, tanh is 0.0, derivative is 1 - 0 = 1.0
    assert abs(float(tanh(z)[0]) - 0.0) < 1e-5
    assert abs(float(tanh.derivative(z)[0]) - 1.0) < 1e-5


def test_linear_forward_and_derivative():
    linear = ACTIVATIONS["Linear"]
    z = np.array([-5.2, 0.0, 42.0])
    
    np.testing.assert_allclose(linear(z), z)
    np.testing.assert_allclose(linear.derivative(z), [1.0, 1.0, 1.0])


def test_step_forward():
    step = ACTIVATIONS["Step"]
    z = np.array([-2.0, -0.001, 0.0, 0.001, 5.0])
    
    np.testing.assert_allclose(step(z), [0.0, 0.0, 1.0, 1.0, 1.0])
