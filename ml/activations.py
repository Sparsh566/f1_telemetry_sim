"""
ml/activations.py
=================
First-Principles Activation Functions and Analytical Derivatives (NumPy).

Includes Step, Sigmoid, Tanh, ReLU, Leaky ReLU, and Linear with exact
analytical gradients, numerical stability protections, and metadata.
"""

from typing import Callable, Dict, Any, Tuple
import numpy as np


class ActivationFunction:
    """Container for activation function forward call and backward derivative."""
    def __init__(
        self,
        name: str,
        forward: Callable[[np.ndarray], np.ndarray],
        backward: Callable[[np.ndarray], np.ndarray],
        output_range: Tuple[float, float],
        formula: str,
        derivative_formula: str,
        description: str,
        limitation: str
    ):
        self.name = name
        self.forward = forward
        self.backward = backward
        self.output_range = output_range
        self.formula = formula
        self.derivative_formula = derivative_formula
        self.description = description
        self.limitation = limitation

    def __call__(self, z: np.ndarray) -> np.ndarray:
        return self.forward(z)

    def derivative(self, z: np.ndarray) -> np.ndarray:
        return self.backward(z)


# 1. Step Activation (Rosenblatt Perceptron)
def _step_fwd(z: np.ndarray) -> np.ndarray:
    return np.where(z >= 0.0, 1.0, 0.0)

def _step_bwd(z: np.ndarray) -> np.ndarray:
    # Derivative is 0 everywhere except undefined at z=0; returns 0 for gradient descent
    return np.zeros_like(z)


# 2. Sigmoid (Logistic)
def _sigmoid_fwd(z: np.ndarray) -> np.ndarray:
    z_clipped = np.clip(z, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z_clipped))

def _sigmoid_bwd(z: np.ndarray) -> np.ndarray:
    s = _sigmoid_fwd(z)
    return s * (1.0 - s)


# 3. Hyperbolic Tangent (Tanh)
def _tanh_fwd(z: np.ndarray) -> np.ndarray:
    return np.tanh(z)

def _tanh_bwd(z: np.ndarray) -> np.ndarray:
    return 1.0 - np.tanh(z) ** 2


# 4. Rectified Linear Unit (ReLU)
def _relu_fwd(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, z)

def _relu_bwd(z: np.ndarray) -> np.ndarray:
    return (z > 0.0).astype(float)


# 5. Leaky ReLU
def _leaky_relu_fwd(z: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    return np.where(z > 0.0, z, z * alpha)

def _leaky_relu_bwd(z: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    return np.where(z > 0.0, 1.0, alpha)


# 6. Linear Identity
def _linear_fwd(z: np.ndarray) -> np.ndarray:
    return z

def _linear_bwd(z: np.ndarray) -> np.ndarray:
    return np.ones_like(z)


ACTIVATIONS: Dict[str, ActivationFunction] = {
    "ReLU": ActivationFunction(
        name="ReLU",
        forward=_relu_fwd,
        backward=_relu_bwd,
        output_range=(0.0, float("inf")),
        formula=r"f(z) = \max(0, z)",
        derivative_formula=r"f'(z) = 1 \text{ if } z > 0 \text{ else } 0",
        description="Standard hidden layer activation. Fast to compute and avoids vanishing gradient for positive activations.",
        limitation="Can suffer from the 'Dying ReLU' problem if neurons receive large negative gradients and become permanently inactive."
    ),
    "Leaky ReLU": ActivationFunction(
        name="Leaky ReLU",
        forward=_leaky_relu_fwd,
        backward=_leaky_relu_bwd,
        output_range=(float("-inf"), float("inf")),
        formula=r"f(z) = \max(\alpha z, z), \quad \alpha=0.05",
        derivative_formula=r"f'(z) = 1 \text{ if } z > 0 \text{ else } 0.05",
        description="Allows a small positive gradient (0.05) when the unit is inactive, curing the dying ReLU problem.",
        limitation="Requires an additional hyperparameter (alpha), slightly slower than standard ReLU."
    ),
    "Sigmoid": ActivationFunction(
        name="Sigmoid",
        forward=_sigmoid_fwd,
        backward=_sigmoid_bwd,
        output_range=(0.0, 1.0),
        formula=r"f(z) = \frac{1}{1 + e^{-z}}",
        derivative_formula=r"f'(z) = f(z)(1 - f(z))",
        description="Squashes real values into probabilities (0 to 1). Ideal for binary decisions such as pit window entry.",
        limitation="Suffers from severe vanishing gradients at saturating extremes (|z| > 4) and is not zero-centered."
    ),
    "Tanh": ActivationFunction(
        name="Tanh",
        forward=_tanh_fwd,
        backward=_tanh_bwd,
        output_range=(-1.0, 1.0),
        formula=r"f(z) = \tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}",
        derivative_formula=r"f'(z) = 1 - \tanh^2(z)",
        description="Zero-centered activation mapped into [-1, 1]. Stronger gradients around zero than Sigmoid.",
        limitation="Still suffers from vanishing gradients when neurons saturate at large positive or negative values."
    ),
    "Linear": ActivationFunction(
        name="Linear",
        forward=_linear_fwd,
        backward=_linear_bwd,
        output_range=(float("-inf"), float("inf")),
        formula=r"f(z) = z",
        derivative_formula=r"f'(z) = 1",
        description="Identity activation used for regression output layers (e.g., predicting continuous lap pace in seconds).",
        limitation="Cannot produce non-linear representations if used in hidden layers."
    ),
    "Step": ActivationFunction(
        name="Step",
        forward=_step_fwd,
        backward=_step_bwd,
        output_range=(0.0, 1.0),
        formula=r"f(z) = 1 \text{ if } z \ge 0 \text{ else } 0",
        derivative_formula=r"f'(z) = 0 \text{ for } z \neq 0",
        description="Original Rosenblatt Perceptron thresholding for binary linear classification.",
        limitation="Derivative is zero almost everywhere, making it impossible to train with gradient-based backpropagation."
    )
}
