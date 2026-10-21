"""
ml/losses.py
============
First-Principles Loss Functions and Analytical Gradients (NumPy).

Includes Mean Squared Error (MSE), Mean Absolute Error (MAE), and
Binary Cross-Entropy (BCE) with analytical derivatives for backpropagation.
"""

from typing import Tuple
import numpy as np


class LossFunction:
    """Base Loss Function Interface."""
    @staticmethod
    def mse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Mean Squared Error: (1 / N) * sum((y_pred - y_true)^2)
        Appropriate for regression where large errors should be penalized quadratically.
        """
        y_p = np.asarray(y_pred)
        y_t = np.asarray(y_true)
        return float(np.mean((y_p - y_t) ** 2))

    @staticmethod
    def mse_gradient(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Analytical gradient dL/d(y_pred): 2 * (y_pred - y_true) / N
        """
        y_p = np.asarray(y_pred)
        y_t = np.asarray(y_true)
        N = max(1, y_p.shape[0])
        return 2.0 * (y_p - y_t) / N

    @staticmethod
    def mae(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Mean Absolute Error: (1 / N) * sum(|y_pred - y_true|)
        Appropriate for robust regression where outliers should not skew gradients.
        """
        y_p = np.asarray(y_pred)
        y_t = np.asarray(y_true)
        return float(np.mean(np.abs(y_p - y_t)))

    @staticmethod
    def mae_gradient(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Subgradient dL/d(y_pred): sign(y_pred - y_true) / N
        """
        y_p = np.asarray(y_pred)
        y_t = np.asarray(y_true)
        N = max(1, y_p.shape[0])
        return np.sign(y_p - y_t) / N

    @staticmethod
    def binary_cross_entropy(y_pred: np.ndarray, y_true: np.ndarray, eps: float = 1e-12) -> float:
        """
        Binary Cross-Entropy: -(1 / N) * sum(y * log(p) + (1 - y) * log(1 - p))
        Appropriate for binary classification probability outputs in [0, 1].
        """
        p = np.clip(np.asarray(y_pred), eps, 1.0 - eps)
        y_t = np.asarray(y_true)
        return float(-np.mean(y_t * np.log(p) + (1.0 - y_t) * np.log(1.0 - p)))

    @staticmethod
    def bce_gradient(y_pred: np.ndarray, y_true: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        """
        Analytical gradient of BCE with respect to predicted probability p.
        dL/dp = ((p - y) / (p * (1 - p))) / N
        """
        p = np.clip(np.asarray(y_pred), eps, 1.0 - eps)
        y_t = np.asarray(y_true)
        N = max(1, p.shape[0])
        return ((p - y_t) / (p * (1.0 - p))) / N


def check_loss_gradient_finite_diff(
    loss_fn,
    loss_grad_fn,
    y_pred: np.ndarray,
    y_true: np.ndarray,
    eps: float = 1e-5
) -> Tuple[bool, float]:
    """
    Validates analytical gradient against numerical central difference:
      dL/dy ≈ (L(y + eps) - L(y - eps)) / (2 * eps)
    """
    analytical_grad = loss_grad_fn(y_pred, y_true)
    numerical_grad = np.zeros_like(y_pred)

    it = np.nditer(y_pred, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        orig_val = y_pred[idx]

        y_pred[idx] = orig_val + eps
        loss_plus = loss_fn(y_pred, y_true)

        y_pred[idx] = orig_val - eps
        loss_minus = loss_fn(y_pred, y_true)

        y_pred[idx] = orig_val  # restore
        numerical_grad[idx] = (loss_plus - loss_minus) / (2.0 * eps)
        it.iternext()

    max_diff = float(np.max(np.abs(analytical_grad - numerical_grad)))
    is_valid = bool(max_diff < 1e-4)
    return is_valid, max_diff
