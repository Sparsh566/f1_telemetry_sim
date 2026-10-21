"""
ml/optimizers.py
================
First-Principles Optimization Algorithms (Pure NumPy).

Includes:
  1. Vanilla Stochastic Gradient Descent (SGD)
  2. SGD with Polyak Momentum
  3. Adam (Adaptive Moment Estimation)
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np


class Optimizer:
    """Base optimizer interface for updating weights and biases."""
    def reset(self) -> None:
        """Resets optimizer internal accumulator states."""
        pass

    def update(
        self,
        layer_idx: int,
        W: np.ndarray,
        b: np.ndarray,
        dW: np.ndarray,
        db: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class SGDOptimizer(Optimizer):
    """
    Vanilla Stochastic Gradient Descent:
      W = W - lr * dW
      b = b - lr * db
    """
    def __init__(self, lr: float = 0.01):
        self.name = "SGD (Vanilla)"
        self.lr = float(lr)

    def reset(self) -> None:
        pass

    def update(
        self,
        layer_idx: int,
        W: np.ndarray,
        b: np.ndarray,
        dW: np.ndarray,
        db: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        W_new = W - self.lr * dW
        b_new = b - self.lr * db
        return W_new, b_new


class SGDMomentumOptimizer(Optimizer):
    """
    SGD with Polyak Momentum:
      v_W = momentum * v_W + lr * dW
      W = W - v_W
    """
    def __init__(self, lr: float = 0.01, momentum: float = 0.9):
        self.name = "SGD + Momentum"
        self.lr = float(lr)
        self.momentum = float(momentum)
        self.v_W: Dict[int, np.ndarray] = {}
        self.v_b: Dict[int, np.ndarray] = {}

    def reset(self) -> None:
        self.v_W.clear()
        self.v_b.clear()

    def update(
        self,
        layer_idx: int,
        W: np.ndarray,
        b: np.ndarray,
        dW: np.ndarray,
        db: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if layer_idx not in self.v_W:
            self.v_W[layer_idx] = np.zeros_like(W)
            self.v_b[layer_idx] = np.zeros_like(b)

        self.v_W[layer_idx] = self.momentum * self.v_W[layer_idx] + self.lr * dW
        self.v_b[layer_idx] = self.momentum * self.v_b[layer_idx] + self.lr * db

        W_new = W - self.v_W[layer_idx]
        b_new = b - self.v_b[layer_idx]
        return W_new, b_new


class AdamOptimizer(Optimizer):
    """
    Adam (Adaptive Moment Estimation):
      m_t = beta1 * m_{t-1} + (1 - beta1) * g_t          (1st moment: mean)
      v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2        (2nd moment: uncentered variance)
      m_hat = m_t / (1 - beta1^t)                        (bias-corrected 1st moment)
      v_hat = v_t / (1 - beta2^t)                        (bias-corrected 2nd moment)
      theta_t = theta_{t-1} - lr * m_hat / (sqrt(v_hat) + eps)
    """
    def __init__(
        self,
        lr: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8
    ):
        self.name = "Adam"
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)

        self.m_W: Dict[int, np.ndarray] = {}
        self.v_W: Dict[int, np.ndarray] = {}
        self.m_b: Dict[int, np.ndarray] = {}
        self.v_b: Dict[int, np.ndarray] = {}
        self.t: Dict[int, int] = {}

    def reset(self) -> None:
        self.m_W.clear()
        self.v_W.clear()
        self.m_b.clear()
        self.v_b.clear()
        self.t.clear()

    def update(
        self,
        layer_idx: int,
        W: np.ndarray,
        b: np.ndarray,
        dW: np.ndarray,
        db: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if layer_idx not in self.m_W:
            self.m_W[layer_idx] = np.zeros_like(W)
            self.v_W[layer_idx] = np.zeros_like(W)
            self.m_b[layer_idx] = np.zeros_like(b)
            self.v_b[layer_idx] = np.zeros_like(b)
            self.t[layer_idx] = 0

        self.t[layer_idx] += 1
        t = self.t[layer_idx]

        # First and second moment updates for weights
        self.m_W[layer_idx] = self.beta1 * self.m_W[layer_idx] + (1.0 - self.beta1) * dW
        self.v_W[layer_idx] = self.beta2 * self.v_W[layer_idx] + (1.0 - self.beta2) * (dW ** 2)

        # First and second moment updates for biases
        self.m_b[layer_idx] = self.beta1 * self.m_b[layer_idx] + (1.0 - self.beta1) * db
        self.v_b[layer_idx] = self.beta2 * self.v_b[layer_idx] + (1.0 - self.beta2) * (db ** 2)

        # Bias corrections
        m_W_hat = self.m_W[layer_idx] / (1.0 - (self.beta1 ** t))
        v_W_hat = self.v_W[layer_idx] / (1.0 - (self.beta2 ** t))

        m_b_hat = self.m_b[layer_idx] / (1.0 - (self.beta1 ** t))
        v_b_hat = self.v_b[layer_idx] / (1.0 - (self.beta2 ** t))

        # Parameter updates
        W_new = W - self.lr * m_W_hat / (np.sqrt(v_W_hat) + self.eps)
        b_new = b - self.lr * m_b_hat / (np.sqrt(v_b_hat) + self.eps)

        return W_new, b_new
