"""
f1_neural_network.py
====================
From-Scratch Deep Learning & Model Deployment Engine for Formula 1 Telemetry.

Architected with pure NumPy (no heavy external deep learning frameworks):
  1. Perceptrons & Decision Boundaries:
     - Single-layer Rosenblatt & Sigmoid Perceptron for binary strategy decisions (e.g. Pit Window trigger).
  2. Activation Functions & Exact Derivatives:
     - Linear, ReLU, Leaky ReLU, Sigmoid, Tanh.
  3. Forward & Backward Propagation:
     - Dense layers with He/Xavier Glorot initialization.
     - Analytical chain rule gradient backpropagation.
  4. Loss Functions & Optimizers:
     - Losses: Mean Squared Error (MSE), Binary Cross-Entropy (BCE), MAE.
     - Optimizers: Vanilla SGD, SGD with Polyak Momentum, Adam (Adaptive Moment Estimation).
  5. Multi-Layer Perceptron (MLP) from scratch:
     - Multi-layer regression predicting F1 lap pace and tyre degradation.
  6. Model Deployment Basics (Pickle vs. Joblib):
     - Serialization benchmarking (dump/load latency, payload size, memory mapping).
     - Export, import, metadata schema, and live telemetry serving.
"""

from typing import Dict, Any, List, Tuple, Optional, Callable
import time
import io
import pickle
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# 1. Feature Definition & Normalization Scaler (Pure NumPy)
# -----------------------------------------------------------------------------
NN_FEATURE_COLUMNS = [
    "speed",
    "throttle",
    "brake",
    "tyre_wear",
    "fuel",
    "brake_temp",
    "aggression"
]

NN_FEATURE_LABELS = {
    "speed": "⚡ Speed (km/h)",
    "throttle": "🟢 Throttle (%)",
    "brake": "🛑 Brake (%)",
    "tyre_wear": "📉 Tyre Wear (%)",
    "fuel": "⛽ Fuel Remaining (%)",
    "brake_temp": "🌡️ Brake Temp (°C)",
    "aggression": "🔥 Driver Aggression"
}


class NumpyStandardScaler:
    """
    Transparent Pure-NumPy Standard Scaler: z = (x - mean) / std.
    Preserves mean and scale for reproducible deployment and inference.
    """
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
        self.feature_names = None

    def fit(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> "NumpyStandardScaler":
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        self.scale_[self.scale_ == 0.0] = 1.0  # Avoid zero division
        self.feature_names = feature_names
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("Scaler must be fitted before transforming.")
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> np.ndarray:
        return self.fit(X, feature_names).transform(X)

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        return (X_scaled * self.scale_) + self.mean_


# -----------------------------------------------------------------------------
# 2. Activation Functions and Analytical Derivatives
# -----------------------------------------------------------------------------
class Activation:
    """Activation function and its corresponding analytical gradient."""
    def __init__(self, name: str, forward: Callable, backward: Callable, description: str, formula: str):
        self.name = name
        self.forward = forward
        self.backward = backward
        self.description = description
        self.formula = formula

    def __call__(self, z: np.ndarray) -> np.ndarray:
        return self.forward(z)

    def derivative(self, z: np.ndarray) -> np.ndarray:
        return self.backward(z)


def _relu_fwd(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, z)

def _relu_bwd(z: np.ndarray) -> np.ndarray:
    return (z > 0.0).astype(float)

def _leaky_relu_fwd(z: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    return np.where(z > 0.0, z, z * alpha)

def _leaky_relu_bwd(z: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    return np.where(z > 0.0, 1.0, alpha)

def _sigmoid_fwd(z: np.ndarray) -> np.ndarray:
    # Stable sigmoid clip to prevent numerical overflow in exp
    z_clip = np.clip(z, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z_clip))

def _sigmoid_bwd(z: np.ndarray) -> np.ndarray:
    s = _sigmoid_fwd(z)
    return s * (1.0 - s)

def _tanh_fwd(z: np.ndarray) -> np.ndarray:
    return np.tanh(z)

def _tanh_bwd(z: np.ndarray) -> np.ndarray:
    return 1.0 - np.tanh(z) ** 2

def _linear_fwd(z: np.ndarray) -> np.ndarray:
    return z

def _linear_bwd(z: np.ndarray) -> np.ndarray:
    return np.ones_like(z)


ACTIVATIONS = {
    "ReLU": Activation(
        name="ReLU",
        forward=_relu_fwd,
        backward=_relu_bwd,
        description="Rectified Linear Unit: Computes max(0, z). Eliminates vanishing gradients in positive domain; ideal for hidden layers modeling non-linear vehicle dynamics.",
        formula=r"f(z) = \max(0, z), \quad f'(z) = \mathbf{1}_{z > 0}"
    ),
    "Leaky ReLU": Activation(
        name="Leaky ReLU",
        forward=_leaky_relu_fwd,
        backward=_leaky_relu_bwd,
        description="Leaky ReLU: Allows a small non-zero slope (0.05) for negative inputs, curing the 'dying neuron' issue when cold brake temperatures produce negative signals.",
        formula=r"f(z) = \begin{cases} z & \text{if } z > 0 \\ 0.05z & \text{otherwise} \end{cases}"
    ),
    "Sigmoid": Activation(
        name="Sigmoid",
        forward=_sigmoid_fwd,
        backward=_sigmoid_bwd,
        description="Sigmoid: Squashes any real number to [0, 1]. Essential for probabilities, such as pit window trigger probability or DRS activation.",
        formula=r"f(z) = \frac{1}{1 + e^{-z}}, \quad f'(z) = f(z)(1 - f(z))"
    ),
    "Tanh": Activation(
        name="Tanh",
        forward=_tanh_fwd,
        backward=_tanh_bwd,
        description="Hyperbolic Tangent: Zero-centered output in [-1, +1]. Stronger gradients than Sigmoid; ideal for telemetry signals centered around target pace.",
        formula=r"f(z) = \tanh(z), \quad f'(z) = 1 - \tanh^2(z)"
    ),
    "Linear": Activation(
        name="Linear",
        forward=_linear_fwd,
        backward=_linear_bwd,
        description="Linear Identity: Passes values directly f(z) = z. Standard output activation for continuous regression tasks (e.g. Lap Time in seconds).",
        formula=r"f(z) = z, \quad f'(z) = 1"
    )
}


# -----------------------------------------------------------------------------
# 3. Loss Functions and Analytical Gradients
# -----------------------------------------------------------------------------
class LossFunction:
    """Loss function container computing scalar loss and backward gradient."""
    @staticmethod
    def mse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Mean Squared Error: (1 / 2N) * sum((y_pred - y_true)^2)"""
        return float(np.mean((y_pred - y_true) ** 2))

    @staticmethod
    def mse_gradient(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Analytical gradient dL/d(y_pred): 2 * (y_pred - y_true) / N"""
        N = y_pred.shape[0]
        return 2.0 * (y_pred - y_true) / max(1, N)

    @staticmethod
    def mae(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Mean Absolute Error: (1 / N) * sum(|y_pred - y_true|)"""
        return float(np.mean(np.abs(y_pred - y_true)))

    @staticmethod
    def r2_score(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Coefficient of determination R² score."""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot == 0.0:
            return 1.0 if ss_res == 0.0 else 0.0
        return float(1.0 - (ss_res / ss_tot))

    @staticmethod
    def binary_cross_entropy(y_pred: np.ndarray, y_true: np.ndarray, eps: float = 1e-12) -> float:
        """Binary Cross-Entropy Loss for probabilistic strategy classification."""
        p = np.clip(y_pred, eps, 1.0 - eps)
        return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))

    @staticmethod
    def bce_gradient(y_pred: np.ndarray, y_true: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        """Gradient of BCE with respect to probability predictions."""
        p = np.clip(y_pred, eps, 1.0 - eps)
        N = y_pred.shape[0]
        return ((p - y_true) / (p * (1.0 - p))) / max(1, N)


# -----------------------------------------------------------------------------
# 4. First-Principles Optimizers: SGD, SGD+Momentum, Adam
# -----------------------------------------------------------------------------
class Optimizer:
    """Base optimizer interface."""
    def update(self, layer_idx: int, W: np.ndarray, b: np.ndarray, dW: np.ndarray, db: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class SGDOptimizer(Optimizer):
    """
    Vanilla Stochastic Gradient Descent:
      W = W - lr * dW
      b = b - lr * db
    """
    def __init__(self, lr: float = 0.01):
        self.name = "SGD (Vanilla)"
        self.lr = lr

    def update(self, layer_idx: int, W: np.ndarray, b: np.ndarray, dW: np.ndarray, db: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        W_new = W - self.lr * dW
        b_new = b - self.lr * db
        return W_new, b_new


class SGDMomentumOptimizer(Optimizer):
    """
    SGD with Polyak Momentum:
      v_W = momentum * v_W + lr * dW
      W = W - v_W
    Accumulates past telemetry gradient vectors to roll through shallow plateaus.
    """
    def __init__(self, lr: float = 0.01, momentum: float = 0.9):
        self.name = "SGD + Momentum"
        self.lr = lr
        self.momentum = momentum
        self.v_W: Dict[int, np.ndarray] = {}
        self.v_b: Dict[int, np.ndarray] = {}

    def update(self, layer_idx: int, W: np.ndarray, b: np.ndarray, dW: np.ndarray, db: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
    def __init__(self, lr: float = 0.01, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.name = "Adam"
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m_W: Dict[int, np.ndarray] = {}
        self.v_W: Dict[int, np.ndarray] = {}
        self.m_b: Dict[int, np.ndarray] = {}
        self.v_b: Dict[int, np.ndarray] = {}
        self.t: Dict[int, int] = {}

    def update(self, layer_idx: int, W: np.ndarray, b: np.ndarray, dW: np.ndarray, db: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if layer_idx not in self.m_W:
            self.m_W[layer_idx] = np.zeros_like(W)
            self.v_W[layer_idx] = np.zeros_like(W)
            self.m_b[layer_idx] = np.zeros_like(b)
            self.v_b[layer_idx] = np.zeros_like(b)
            self.t[layer_idx] = 0

        self.t[layer_idx] += 1
        t = self.t[layer_idx]

        # Weights 1st and 2nd moments
        self.m_W[layer_idx] = self.beta1 * self.m_W[layer_idx] + (1.0 - self.beta1) * dW
        self.v_W[layer_idx] = self.beta2 * self.v_W[layer_idx] + (1.0 - self.beta2) * (dW ** 2)

        # Biases 1st and 2nd moments
        self.m_b[layer_idx] = self.beta1 * self.m_b[layer_idx] + (1.0 - self.beta1) * db
        self.v_b[layer_idx] = self.beta2 * self.v_b[layer_idx] + (1.0 - self.beta2) * (db ** 2)

        # Bias corrections
        m_W_hat = self.m_W[layer_idx] / (1.0 - (self.beta1 ** t))
        v_W_hat = self.v_W[layer_idx] / (1.0 - (self.beta2 ** t))
        m_b_hat = self.m_b[layer_idx] / (1.0 - (self.beta1 ** t))
        v_b_hat = self.v_b[layer_idx] / (1.0 - (self.beta2 ** t))

        # Weight updates
        W_new = W - self.lr * m_W_hat / (np.sqrt(v_W_hat) + self.eps)
        b_new = b - self.lr * m_b_hat / (np.sqrt(v_b_hat) + self.eps)

        return W_new, b_new


# -----------------------------------------------------------------------------
# 5. Dense Layer & Backpropagation Engine
# -----------------------------------------------------------------------------
class DenseLayer:
    """
    Fully Connected Dense Layer with He / Xavier weight initialization
    and forward/backward activation caches.
    """
    def __init__(self, in_features: int, out_features: int, activation_name: str = "ReLU", seed: Optional[int] = None):
        self.in_features = in_features
        self.out_features = out_features
        self.activation_name = activation_name
        self.activation = ACTIVATIONS[activation_name]

        # Weight Initialization:
        # He Normal for ReLU / Leaky ReLU; Xavier/Glorot for Sigmoid / Tanh / Linear
        rng = np.random.RandomState(seed)
        if activation_name in ["ReLU", "Leaky ReLU"]:
            std = np.sqrt(2.0 / in_features)
        else:
            std = np.sqrt(1.0 / in_features)

        self.W = rng.normal(0.0, std, size=(in_features, out_features))
        self.b = np.zeros((1, out_features))

        # Caches for backpropagation
        self.X_cache: Optional[np.ndarray] = None
        self.Z_cache: Optional[np.ndarray] = None
        self.A_cache: Optional[np.ndarray] = None

        # Gradients computed during backward pass
        self.dW: Optional[np.ndarray] = None
        self.db: Optional[np.ndarray] = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward Pass:
          Z = X * W + b
          A = activation(Z)
        """
        self.X_cache = X
        self.Z_cache = np.dot(X, self.W) + self.b
        self.A_cache = self.activation(self.Z_cache)
        return self.A_cache

    def backward(self, dA: np.ndarray) -> np.ndarray:
        """
        Backward Pass via the Chain Rule:
          dZ = dA * activation'(Z)
          dW = (1/N) * X^T * dZ
          db = (1/N) * sum(dZ, axis=0)
          dX = dZ * W^T
        """
        dZ = dA * self.activation.derivative(self.Z_cache)
        N = self.X_cache.shape[0]

        self.dW = np.dot(self.X_cache.T, dZ)
        self.db = np.sum(dZ, axis=0, keepdims=True)
        dX = np.dot(dZ, self.W.T)
        return dX


# -----------------------------------------------------------------------------
# 6. From-Scratch Neural Network (Multi-Layer Perceptron)
# -----------------------------------------------------------------------------
class NeuralNetwork:
    """
    Complete pure-NumPy Multi-Layer Perceptron.
    Supports arbitrary layer configurations, optimizers, and tracking metrics.
    """
    def __init__(self, layer_sizes: List[int], activations: List[str], seed: int = 42):
        if len(layer_sizes) < 2:
            raise ValueError("Network must have at least an input and output layer.")
        if len(activations) != len(layer_sizes) - 1:
            raise ValueError("Number of activations must equal number of layers minus 1.")

        self.layer_sizes = layer_sizes
        self.activations = activations
        self.layers: List[DenseLayer] = []
        self.history: Dict[str, List[float]] = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "train_r2": [],
            "val_r2": [],
            "grad_norm": []
        }

        # Initialize network layers
        for i in range(len(layer_sizes) - 1):
            layer_seed = (seed + i * 37) if seed is not None else None
            layer = DenseLayer(
                in_features=layer_sizes[i],
                out_features=layer_sizes[i + 1],
                activation_name=activations[i],
                seed=layer_seed
            )
            self.layers.append(layer)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Pass input through all layers sequentially."""
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A

    def backward(self, dL_dY: np.ndarray):
        """Propagate gradients backward from output layer to input layer."""
        dA = dL_dY
        for layer in reversed(self.layers):
            dA = layer.backward(dA)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        optimizer: Optional[Optimizer] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = False
    ) -> Dict[str, List[float]]:
        """
        Trains the neural network using mini-batch gradient descent.
        """
        if optimizer is None:
            optimizer = AdamOptimizer(lr=0.01)

        N = X_train.shape[0]
        y_train = y_train.reshape(-1, 1) if y_train.ndim == 1 else y_train
        if y_val is not None and y_val.ndim == 1:
            y_val = y_val.reshape(-1, 1)

        self.history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "train_r2": [],
            "val_r2": [],
            "grad_norm": []
        }

        for epoch in range(1, epochs + 1):
            # Shuffle training batches
            indices = np.random.permutation(N)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            epoch_grad_norms = []

            for start_idx in range(0, N, batch_size):
                end_idx = min(start_idx + batch_size, N)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                # 1. Forward Pass
                y_batch_pred = self.forward(X_batch)

                # 2. Compute Output Loss Gradient
                dL_dY = LossFunction.mse_gradient(y_batch_pred, y_batch)

                # 3. Backward Pass (Chain rule backpropagation)
                self.backward(dL_dY)

                # 4. Weight Updates via Optimizer
                batch_gnorm = 0.0
                for l_idx, layer in enumerate(self.layers):
                    gn = float(np.linalg.norm(layer.dW) + np.linalg.norm(layer.db))
                    batch_gnorm += gn
                    layer.W, layer.b = optimizer.update(
                        layer_idx=l_idx,
                        W=layer.W,
                        b=layer.b,
                        dW=layer.dW,
                        db=layer.db
                    )
                epoch_grad_norms.append(batch_gnorm)

            # Evaluate epoch metrics
            y_train_pred = self.forward(X_train)
            train_loss = LossFunction.mse(y_train_pred, y_train)
            train_r2 = LossFunction.r2_score(y_train_pred, y_train)

            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(train_loss)
            self.history["train_r2"].append(train_r2)
            self.history["grad_norm"].append(float(np.mean(epoch_grad_norms)))

            if X_val is not None and y_val is not None:
                y_val_pred = self.forward(X_val)
                val_loss = LossFunction.mse(y_val_pred, y_val)
                val_r2 = LossFunction.r2_score(y_val_pred, y_val)
                self.history["val_loss"].append(val_loss)
                self.history["val_r2"].append(val_r2)
            else:
                self.history["val_loss"].append(train_loss)
                self.history["val_r2"].append(train_r2)

        return self.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Run forward inference and return continuous predictions."""
        preds = self.forward(X)
        return preds.ravel() if preds.shape[1] == 1 else preds

    def get_parameter_count(self) -> int:
        """Returns total number of trainable weights and biases."""
        return sum(layer.W.size + layer.b.size for layer in self.layers)


# -----------------------------------------------------------------------------
# 7. Single Perceptron & Pit Window Decision Boundary
# -----------------------------------------------------------------------------
class SinglePerceptron:
    """
    Classic Single Perceptron for linear binary classification:
      z = w1 * x1 + w2 * x2 + b
      p = sigmoid(z) or step(z)
    Used to classify Pit Window decisions based on Tyre Wear and Pace Drop.
    """
    def __init__(self, in_features: int = 2, lr: float = 0.05, seed: int = 42):
        rng = np.random.RandomState(seed)
        self.weights = rng.normal(0.0, 0.5, size=(in_features,))
        self.bias = 0.0
        self.lr = lr
        self.in_features = in_features

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Computes linear score z = X * w + b."""
        return np.dot(X, self.weights) + self.bias

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Sigmoid probability output."""
        z = self.forward(X)
        return _sigmoid_fwd(z)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Binary classification output (0 or 1)."""
        return (self.predict_proba(X) >= threshold).astype(int)

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 50) -> List[float]:
        """Trains perceptron using logistic loss gradient descent."""
        N = X.shape[0]
        losses = []
        for _ in range(epochs):
            p = self.predict_proba(X)
            loss = LossFunction.binary_cross_entropy(p, y)
            losses.append(loss)

            # Gradient for cross-entropy with sigmoid
            error = p - y
            dw = np.dot(X.T, error) / N
            db = np.sum(error) / N

            self.weights -= self.lr * dw
            self.bias -= self.lr * db
        return losses

    def get_decision_boundary_line(self, x1_range: Tuple[float, float]) -> Tuple[List[float], List[float]]:
        """
        Derives the 2D linear decision boundary line where w1 * x1 + w2 * x2 + b = 0.
        Therefore: x2 = -(w1 * x1 + b) / w2
        """
        w1, w2 = self.weights[0], self.weights[1]
        if abs(w2) < 1e-6:
            w2 = 1e-6
        x1_vals = [x1_range[0], x1_range[1]]
        x2_vals = [float(-(w1 * x + self.bias) / w2) for x in x1_vals]
        return x1_vals, x2_vals


# -----------------------------------------------------------------------------
# 8. Model Deployment Basics (Pickle vs. Joblib)
# -----------------------------------------------------------------------------
def serialize_model_payload(
    model: Any,
    scaler: Any,
    metadata: Dict[str, Any],
    method: str = "joblib"
) -> bytes:
    """
    Serializes a complete deployment bundle:
      - Trained neural net architecture, weights, and biases
      - Feature normalizer scaler parameters
      - Metadata: training timestamp, accuracy metrics, git/model version
    Supports both Python's built-in `pickle` and Scikit-Learn's `joblib`.
    """
    bundle = {
        "model": model,
        "scaler": scaler,
        "metadata": metadata,
        "serialization_format": method,
        "timestamp": time.time()
    }

    buf = io.BytesIO()
    if method == "joblib":
        import joblib
        joblib.dump(bundle, buf, compress=3)
    elif method == "pickle":
        pickle.dump(bundle, buf, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        raise ValueError(f"Unsupported serialization method: {method}")

    buf.seek(0)
    return buf.read()


def deserialize_model_payload(payload_bytes: bytes, method: str = "joblib") -> Dict[str, Any]:
    """
    Loads and reconstitutes a saved model bundle from raw bytes.
    """
    buf = io.BytesIO(payload_bytes)
    if method == "joblib":
        import joblib
        bundle = joblib.load(buf)
    elif method == "pickle":
        bundle = pickle.load(buf)
    else:
        raise ValueError(f"Unsupported deserialization method: {method}")
    return bundle


def benchmark_pickle_vs_joblib(model: Any, scaler: Any) -> pd.DataFrame:
    """
    Direct head-to-head empirical benchmark comparing `pickle` vs `joblib`
    on the current neural network weights array:
      - Dump latency (ms)
      - Load latency (ms)
      - File size (Bytes)
      - Compression factor
    """
    metadata = {
        "model_type": "F1_NumPy_NeuralNet",
        "parameters": model.get_parameter_count() if hasattr(model, "get_parameter_count") else 100,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    # 1. Benchmark Pickle
    t0 = time.perf_counter()
    pkl_bytes = serialize_model_payload(model, scaler, metadata, method="pickle")
    t_pkl_dump = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    _ = deserialize_model_payload(pkl_bytes, method="pickle")
    t_pkl_load = (time.perf_counter() - t0) * 1000.0

    # 2. Benchmark Joblib
    t0 = time.perf_counter()
    joblib_bytes = serialize_model_payload(model, scaler, metadata, method="joblib")
    t_joblib_dump = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    _ = deserialize_model_payload(joblib_bytes, method="joblib")
    t_joblib_load = (time.perf_counter() - t0) * 1000.0

    df_bench = pd.DataFrame([
        {
            "Serialization Tool": "Python Pickle (`pickle.dump`)",
            "File Size (Bytes)": len(pkl_bytes),
            "Dump Time (ms)": round(t_pkl_dump, 2),
            "Load Time (ms)": round(t_pkl_load, 2),
            "Memory Mapping": "❌ No (Full copy into memory)",
            "Best Used For": "General Python objects & primitives"
        },
        {
            "Serialization Tool": "Joblib (`joblib.dump`)",
            "File Size (Bytes)": len(joblib_bytes),
            "Dump Time (ms)": round(t_joblib_dump, 2),
            "Load Time (ms)": round(t_joblib_load, 2),
            "Memory Mapping": "✅ Yes (Zero-copy memmap for large NumPy arrays)",
            "Best Used For": "Machine Learning pipelines & heavy numeric matrices"
        }
    ])
    return df_bench


# -----------------------------------------------------------------------------
# 9. Multi-Optimizer Telemetry Convergence Benchmark
# -----------------------------------------------------------------------------
def run_optimizer_convergence_battle(
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 70,
    lr: float = 0.01,
    hidden_dims: Optional[List[int]] = None
) -> Tuple[pd.DataFrame, Dict[str, NeuralNetwork]]:
    """
    Fits three identical neural networks with:
      1. Vanilla SGD
      2. SGD with Momentum (0.9)
      3. Adam (Adaptive Moment Estimation)
    under exact same initial random weights to visualize the convergence race.
    """
    if hidden_dims is None:
        hidden_dims = [24, 12]

    in_dim = X_train.shape[1]
    layer_sizes = [in_dim] + hidden_dims + [1]
    activations = ["ReLU"] * len(hidden_dims) + ["Linear"]

    optimizers = [
        ("Vanilla SGD", SGDOptimizer(lr=lr)),
        ("SGD + Momentum", SGDMomentumOptimizer(lr=lr, momentum=0.9)),
        ("Adam Optimizer", AdamOptimizer(lr=lr))
    ]

    history_dict = {}
    fitted_nets = {}

    for opt_name, opt_inst in optimizers:
        # Fixed seed guarantees identical weight initialization across runs
        net = NeuralNetwork(layer_sizes=layer_sizes, activations=activations, seed=101)
        hist = net.fit(
            X_train=X_train,
            y_train=y_train,
            epochs=epochs,
            batch_size=32,
            optimizer=opt_inst,
            verbose=False
        )
        history_dict[opt_name] = hist["train_loss"]
        fitted_nets[opt_name] = net

    df_convergence = pd.DataFrame(history_dict, index=[f"Epoch {e}" for e in range(1, epochs + 1)])
    df_convergence.index.name = "Epoch"
    return df_convergence, fitted_nets


# -----------------------------------------------------------------------------
# 10. Student Deep Learning & Deployment Master Guide
# -----------------------------------------------------------------------------
NN_STUDY_GUIDE = {
    "perceptron_basics": r"""
### ⚡ 1. The Perceptron & The Linearity Limit

The **Perceptron** (Frank Rosenblatt, 1958) is the fundamental building block of all modern deep neural networks.

$$z = \sum_{i=1}^D w_i x_i + b = \mathbf{w}^T \mathbf{x} + b$$

$$\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}$$

#### Why a Single Perceptron Fails on Complex F1 Telemetry:
* **Linear Separability Constraint**: A single perceptron can only draw a straight hyperplane boundary. 
* **The XOR / Non-Linear Dynamic**: In racing, a car with low fuel can go fast, and a car with high grip can go fast. But a car with high tyre wear and high fuel suffers an **exponential** lap-time penalty (not a linear sum). Single perceptrons cannot model interactions or non-linear curves without hidden layers.
""",

    "activations_derivatives": r"""
### 🧬 2. Activation Functions: Non-Linearity & Vanishing Gradients

Without non-linear activation functions, stacking 100 neural network layers collapses mathematically into a single linear matrix multiplication:
$$W_2 \cdot (W_1 \cdot X) = (W_2 W_1) \cdot X = W_{net} \cdot X$$

| Activation | Mathematical Formula | Derivative $f'(z)$ | Range | When to Use in Racing Telemetry |
| :--- | :--- | :--- | :--- | :--- |
| **ReLU** | $\max(0, z)$ | $1$ if $z > 0$ else $0$ | $[0, \infty)$ | **Standard Hidden Layers**: Fastest compute, zero vanishing gradient for positive values. |
| **Leaky ReLU** | $z$ if $z > 0$ else $0.05z$ | $1$ if $z > 0$ else $0.05$ | $(-\infty, \infty)$ | **Prevent Dead Neurons**: When cold telemetry features (sub-zero delta pace) cause neurons to deactivate permanently. |
| **Sigmoid** | $\frac{1}{1 + e^{-z}}$ | $f(z)(1 - f(z))$ | $(0, 1)$ | **Binary Probabilities**: Pit window entry chance, overtake probability. |
| **Tanh** | $\frac{e^z - e^{-z}}{e^z + e^{-z}}$ | $1 - \tanh^2(z)$ | $(-1, 1)$ | **Zero-Centered Dynamics**: Telemetry signals oscillating around delta lap time (faster or slower). |
| **Linear** | $z$ | $1$ | $(-\infty, \infty)$ | **Regression Output**: Predicting absolute lap time in seconds. |
""",

    "backpropagation_chain_rule": r"""
### 🔄 3. Forward & Backward Propagation (Analytical Chain Rule)

Deep neural networks learn by calculating the gradient of the loss function with respect to every single weight: $\frac{\partial L}{\partial W^{[l]}}$.

#### 1. Forward Pass (Layer $l$):
$$Z^{[l]} = A^{[l-1]} W^{[l]} + b^{[l]}$$
$$A^{[l]} = g^{[l]}(Z^{[l]})$$

#### 2. Backward Pass via Analytical Chain Rule:
To compute the error signal $\delta^{[l]} = \frac{\partial L}{\partial Z^{[l]}}$:
$$\delta^{[l]} = \left( \delta^{[l+1]} (W^{[l+1]})^T \right) \odot g'^{[l]}(Z^{[l]})$$

#### 3. Parameter Gradients & Weight Updates:
$$\frac{\partial L}{\partial W^{[l]}} = \frac{1}{N} (A^{[l-1]})^T \delta^{[l]}$$
$$\frac{\partial L}{\partial b^{[l]}} = \frac{1}{N} \sum_{i=1}^N \delta^{[l]}_i$$
$$W^{[l]} \leftarrow W^{[l]} - \eta \frac{\partial L}{\partial W^{[l]}}$$
""",

    "optimizers_sgd_adam": r"""
### 🚀 4. Loss Functions & Optimizers: SGD vs. Adam

The optimization landscape of telemetry neural networks contains steep ravines, local flat plateaus, and noisy sensor oscillations.

#### 1. Vanilla SGD (Stochastic Gradient Descent)
$$W_{t+1} = W_t - \eta \nabla L(W_t)$$
* **Limitation**: Suffers from high variance oscillations across narrow canyons; can easily get stuck on saddles.

#### 2. SGD with Momentum (Polyak, 1964)
$$v_t = \beta v_{t-1} + \eta \nabla L(W_t)$$
$$W_{t+1} = W_t - v_t$$
* **Advantage**: Simulates a heavy rolling ball. Dampens orthogonal oscillations and accelerates through consistent downhill gradient directions.

#### 3. Adam (Adaptive Moment Estimation - Kingma & Ba, 2014)
Combines **Momentum** (1st moment, directional mean) and **RMSprop** (2nd moment, uncentered variance):
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \quad \text{(First moment)}$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \quad \text{(Second moment)}$$
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t} \quad \text{(Bias corrections for step } t=0)$$
$$W_{t+1} = W_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$
* **Why Adam Wins**: Features with infrequent or noisy spikes (e.g. sudden brake locks) receive smaller scaled updates, while smooth consistent features get steady progress.
""",

    "deployment_pickle_joblib": r"""
### 📦 5. Model Deployment Fundamentals: Pickle vs. Joblib

In production race engineering, training a model is only 20% of the lifecycle. The model must be serialized, versioned, and served in real time to the pit wall.

#### Comparison: `pickle` vs. `joblib`
1. **Python `pickle`**:
   - Python's standard object serializer.
   - Converts arbitrary Python objects into a byte stream.
   - **Flaw with ML**: Serializes large NumPy arrays by creating full object copies, leading to high memory overhead and slower I/O on multi-gigabyte models.

2. **Joblib (`joblib.dump` / `joblib.load`)**:
   - Specifically optimized for **scientific Python** (NumPy, Scikit-Learn).
   - Uses memory mapping (`memmap`) on disk for large arrays, allowing workers to read shared model weights with **zero-copy RAM footprint**.
   - Built-in compression algorithms (`zlib`, `lz4`).

#### Production Serving Checklist:
* ✅ **Always serialize preprocessing pipelines alongside weights** (e.g., standard scalers with exact `mean_` and `scale_`).
* ✅ **Record metadata schema**: Feature names, package versions (`scikit-learn==1.6.x`), git commit hash, and validation metrics.
* ⚠️ **Security Warning**: Never `unpickle` or `joblib.load` untrusted model files downloaded from the internet, as deserialization can execute arbitrary malicious code via Python bytecode opcodes.
"""
}


# -----------------------------------------------------------------------------
# 11. Interactive UI Helper Utilities
# -----------------------------------------------------------------------------
def generate_pit_window_dataset(n_samples: int = 250, seed: int = 42) -> pd.DataFrame:
    """
    Generates a 2D telemetry dataset for Single Perceptron classification:
      Feature 1: Tyre Wear (%) in [15, 85]
      Feature 2: Lap Pace Drop (seconds slower than optimal) in [0.2, 4.5]
      Target: Pit Stop Recommendation (1 = Box Box Box, 0 = Stay Out)
    """
    rng = np.random.RandomState(seed)
    tyre_wear = rng.uniform(15.0, 85.0, size=n_samples)
    pace_drop = rng.uniform(0.1, 4.5, size=n_samples)

    score = 0.06 * tyre_wear + 0.9 * pace_drop - 5.0 + rng.normal(0, 0.35, size=n_samples)
    pit_decision = (score > 0).astype(int)

    return pd.DataFrame({
        "tyre_wear": np.round(tyre_wear, 1),
        "pace_drop": np.round(pace_drop, 2),
        "pit_decision": pit_decision,
        "pit_label": ["🛑 PIT STOP (Box)" if p == 1 else "🟢 STAY OUT (Track)" for p in pit_decision]
    })


def get_activation_plot_data(activation_name: str, z_min: float = -5.0, z_max: float = 5.0, n_points: int = 200) -> pd.DataFrame:
    """
    Generates coordinate series for plotting an activation function and its derivative.
    """
    act = ACTIVATIONS.get(activation_name, ACTIVATIONS["ReLU"])
    z_vals = np.linspace(z_min, z_max, n_points)
    f_vals = act(z_vals)
    df_vals = act.derivative(z_vals)
    return pd.DataFrame({
        "z (Input Score)": z_vals,
        f"f(z) [{activation_name}]": f_vals,
        f"f'(z) Derivative": df_vals
    })


def predict_live_grid_neural_net(
    model: NeuralNetwork,
    scaler: NumpyStandardScaler,
    state: Dict[str, Any],
    driver_labels: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """
    Evaluates real-time neural network lap-time predictions for all cars
    currently active on the circuit.
    """
    if driver_labels is None:
        driver_labels = {}

    rows = []
    for code, car in sorted(state.items(), key=lambda kv: kv[1]["position"]):
        speed = float(car.get("speed", 250.0))
        throttle = float(car.get("throttle", 0.8) * 100.0)
        brake = float(car.get("brake", 0.0) * 100.0)
        tyre_wear = float(car.get("tyre_wear", 0.1) * 100.0)
        fuel = float(car.get("fuel", 50.0))
        brake_temp = float(car.get("brake_temp", 500.0))
        aggression = float(car.get("aggression", 1.0))

        feat_vector = np.array([[speed, throttle, brake, tyre_wear, fuel, brake_temp, aggression]])
        feat_scaled = scaler.transform(feat_vector)
        pred_lap = float(model.predict(feat_scaled)[0])
        current_lap = float(car.get("lap_time", 85.0))
        delta = pred_lap - current_lap

        rows.append({
            "Driver": driver_labels.get(code, code),
            "Tyre": car.get("tyre", "Medium"),
            "Wear (%)": f"{tyre_wear:.1f}%",
            "Speed": f"{speed:.1f} km/h",
            "Current Lap Time": f"{current_lap:.3f}s",
            "NN Predicted Pace": f"{pred_lap:.3f}s",
            "Delta (NN vs Live)": f"{delta:+.3f}s"
        })
    return pd.DataFrame(rows)

