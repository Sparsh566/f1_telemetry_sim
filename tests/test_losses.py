"""
tests/test_losses.py
====================
Unit tests for MSE, MAE, Binary Cross-Entropy, and analytical gradient verification.
"""

import numpy as np
from ml.losses import LossFunction, check_loss_gradient_finite_diff


def test_mse_value_and_finite_diff_gradient():
    y_pred = np.array([2.0, 4.0, 6.0])
    y_true = np.array([1.0, 3.0, 5.0])
    
    # Each error is 1.0, squared is 1.0, mean is 1.0
    loss = LossFunction.mse(y_pred, y_true)
    assert abs(loss - 1.0) < 1e-6
    
    # Analytical gradient: 2 * (y_pred - y_true) / 3 = 2/3 ≈ 0.66667
    grad = LossFunction.mse_gradient(y_pred, y_true)
    np.testing.assert_allclose(grad, [2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0])
    
    # Finite difference test
    is_valid, max_diff = check_loss_gradient_finite_diff(
        loss_fn=LossFunction.mse,
        loss_grad_fn=LossFunction.mse_gradient,
        y_pred=y_pred.copy(),
        y_true=y_true
    )
    assert is_valid, f"MSE analytical gradient failed finite difference check (diff={max_diff})"


def test_mae_value():
    y_pred = np.array([10.0, 20.0, 30.0])
    y_true = np.array([8.0, 25.0, 30.0])
    
    # abs errors: [2, 5, 0], mean = 7 / 3
    loss = LossFunction.mae(y_pred, y_true)
    assert abs(loss - (7.0 / 3.0)) < 1e-6


def test_bce_value_and_finite_diff():
    y_pred = np.array([0.8, 0.2])
    y_true = np.array([1.0, 0.0])
    
    # Loss should be positive and low since predictions match targets well
    loss = LossFunction.binary_cross_entropy(y_pred, y_true)
    assert loss > 0.0
    
    # Finite difference check
    is_valid, max_diff = check_loss_gradient_finite_diff(
        loss_fn=LossFunction.binary_cross_entropy,
        loss_grad_fn=LossFunction.bce_gradient,
        y_pred=y_pred.copy(),
        y_true=y_true
    )
    assert is_valid, f"BCE analytical gradient failed finite difference check (diff={max_diff})"
