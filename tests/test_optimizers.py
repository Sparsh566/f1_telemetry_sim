"""
tests/test_optimizers.py
========================
Unit tests for SGD, Momentum, and Adam optimizer updates and state resets.
"""

import numpy as np
from ml.optimizers import SGDOptimizer, SGDMomentumOptimizer, AdamOptimizer


def test_sgd_update():
    opt = SGDOptimizer(lr=0.1)
    W = np.array([[1.0, 2.0]])
    b = np.array([[0.5]])
    dW = np.array([[0.2, -0.4]])
    db = np.array([[0.1]])

    W_new, b_new = opt.update(0, W, b, dW, db)

    # W_new = W - 0.1 * dW = [1.0 - 0.02, 2.0 + 0.04] = [0.98, 2.04]
    np.testing.assert_allclose(W_new, [[0.98, 2.04]])
    np.testing.assert_allclose(b_new, [[0.49]])


def test_sgd_momentum_accumulation():
    opt = SGDMomentumOptimizer(lr=0.1, momentum=0.9)
    W = np.array([[1.0]])
    b = np.array([[0.0]])
    dW = np.array([[1.0]])
    db = np.array([[1.0]])

    # Step 1: v_W = 0.9 * 0 + 0.1 * 1.0 = 0.1 -> W_new = 1.0 - 0.1 = 0.9
    W_1, b_1 = opt.update(0, W, b, dW, db)
    np.testing.assert_allclose(W_1, [[0.9]])

    # Step 2: v_W = 0.9 * 0.1 + 0.1 * 1.0 = 0.19 -> W_new = 0.9 - 0.19 = 0.71
    W_2, b_2 = opt.update(0, W_1, b_1, dW, db)
    np.testing.assert_allclose(W_2, [[0.71]])


def test_adam_update_and_reset():
    opt = AdamOptimizer(lr=0.01)
    W = np.zeros((2, 2))
    b = np.zeros((1, 2))
    dW = np.ones((2, 2))
    db = np.ones((1, 2))

    W_new, b_new = opt.update(0, W, b, dW, db)

    # With non-zero gradient, parameters must move in negative direction
    assert (W_new < 0).all()
    assert (b_new < 0).all()
    assert opt.t[0] == 1

    # Reset must clear internal step count and accumulators
    opt.reset()
    assert len(opt.t) == 0
    assert len(opt.m_W) == 0
