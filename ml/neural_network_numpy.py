"""
ml/neural_network_numpy.py
==========================
From-Scratch Deep Learning & Neural Network Engine (Pure NumPy).

No external deep learning frameworks (no PyTorch, TensorFlow, or Keras).
Implements:
  1. SinglePerceptron & XOR Non-Linearity Limit Demonstration
  2. DenseLayer with He/Xavier Glorot initialization and forward/backward caches
  3. Multi-Layer Perceptron (NeuralNetwork) with analytical backpropagation
  4. Finite-difference numerical gradient checking utility
  5. Weight serialization and deserialization
"""

from typing import List, Dict, Any, Tuple, Optional, Callable
import time
import io
import pickle
import numpy as np
import pandas as pd

from ml.activations import ACTIVATIONS, ActivationFunction
from ml.losses import LossFunction
from ml.optimizers import Optimizer, AdamOptimizer, SGDOptimizer, SGDMomentumOptimizer


class NumpyStandardScaler:
    """
    Pure-NumPy Feature Scaler: z = (x - mean) / std.
    Preserves fitted parameters strictly from the training dataset.
    """
    def __init__(self):
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self.feature_names: Optional[List[str]] = None

    def fit(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> "NumpyStandardScaler":
        arr = np.asarray(X, dtype=float)
        self.mean_ = np.mean(arr, axis=0)
        self.scale_ = np.std(arr, axis=0)
        self.scale_[self.scale_ == 0.0] = 1.0  # Avoid division by zero
        self.feature_names = feature_names
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("Scaler must be fitted before transforming.")
        arr = np.asarray(X, dtype=float)
        return (arr - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> np.ndarray:
        return self.fit(X, feature_names).transform(X)

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("Scaler must be fitted before inverse transforming.")
        return (np.asarray(X_scaled, dtype=float) * self.scale_) + self.mean_


class SinglePerceptron:
    """
    Frank Rosenblatt's 1958 Perceptron for binary classification:
      z = w1 * x1 + w2 * x2 + ... + b
      y = step(z) or sigmoid(z)
    """
    def __init__(self, in_features: int = 2, lr: float = 0.05, seed: int = 42):
        rng = np.random.RandomState(seed)
        self.in_features = in_features
        self.weights = rng.normal(0.0, 0.5, size=(in_features,))
        self.bias = 0.0
        self.lr = lr

    def weighted_sum(self, X: np.ndarray) -> np.ndarray:
        """Computes linear score z = X * w + b."""
        return np.dot(X, self.weights) + self.bias

    def predict_step(self, X: np.ndarray) -> np.ndarray:
        """Step activation prediction (0 or 1)."""
        z = self.weighted_sum(X)
        return (z >= 0.0).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Sigmoid probability prediction in [0, 1]."""
        z = self.weighted_sum(X)
        z_clip = np.clip(z, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-z_clip))

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Thresholded binary prediction."""
        return (self.predict_proba(X) >= threshold).astype(int)

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 50) -> List[float]:
        """Trains perceptron using logistic loss gradient descent."""
        N = X.shape[0]
        losses = []
        for _ in range(epochs):
            p = self.predict_proba(X)
            loss = LossFunction.binary_cross_entropy(p, y)
            losses.append(loss)

            error = p - y
            dw = np.dot(X.T, error) / N
            db = np.sum(error) / N

            self.weights -= self.lr * dw
            self.bias -= self.lr * db
        return losses

    def get_decision_boundary(self, x1_range: Tuple[float, float]) -> Tuple[List[float], List[float]]:
        """
        Derives linear boundary line where w1 * x1 + w2 * x2 + b = 0.
        x2 = -(w1 * x1 + b) / w2
        """
        w1, w2 = self.weights[0], self.weights[1]
        w2_safe = w2 if abs(w2) > 1e-6 else 1e-6
        x1_vals = [float(x1_range[0]), float(x1_range[1])]
        x2_vals = [float(-(w1 * x + self.bias) / w2_safe) for x in x1_vals]
        return x1_vals, x2_vals

    @staticmethod
    def demonstrate_xor_limitation() -> Dict[str, Any]:
        """
        Demonstrates why a single-layer perceptron CANNOT solve XOR (Minsky & Papert, 1969).
        """
        X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y_xor = np.array([0, 1, 1, 0])

        perc = SinglePerceptron(in_features=2, lr=0.1, seed=42)
        losses = perc.fit(X_xor, y_xor, epochs=100)
        preds = perc.predict(X_xor)
        accuracy = float(np.mean(preds == y_xor))

        return {
            "inputs": X_xor.tolist(),
            "targets": y_xor.tolist(),
            "predictions": preds.tolist(),
            "accuracy": accuracy,
            "explanation": (
                "A single perceptron can only produce a linear hyperplane boundary. "
                "Because XOR points (0,1) and (1,0) cannot be separated from (0,0) and (1,1) "
                "by any single straight line, single-layer perceptrons fail (accuracy ~50%). "
                "Solving XOR requires at least one non-linear hidden layer (Multi-Layer Perceptron)."
            )
        }


class DenseLayer:
    """
    Fully Connected Feedforward Layer with He / Xavier weight initialization
    and explicit caches for analytical backpropagation.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation_name: str = "ReLU",
        seed: Optional[int] = None
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.activation_name = activation_name

        if activation_name not in ACTIVATIONS:
            raise ValueError(f"Unknown activation '{activation_name}'. Available: {list(ACTIVATIONS.keys())}")
        self.activation = ACTIVATIONS[activation_name]

        # Initialization
        rng = np.random.RandomState(seed)
        if activation_name in ["ReLU", "Leaky ReLU"]:
            # He Normal Initialization: std = sqrt(2 / n_in)
            std = np.sqrt(2.0 / in_features)
        else:
            # Xavier/Glorot Normal Initialization: std = sqrt(1 / n_in)
            std = np.sqrt(1.0 / in_features)

        self.W = rng.normal(0.0, std, size=(in_features, out_features))
        self.b = np.zeros((1, out_features))

        # Caches
        self.X_cache: Optional[np.ndarray] = None
        self.Z_cache: Optional[np.ndarray] = None
        self.A_cache: Optional[np.ndarray] = None

        # Gradients
        self.dW: Optional[np.ndarray] = None
        self.db: Optional[np.ndarray] = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward Pass:
          Z = X · W + b
          A = activation(Z)
        """
        self.X_cache = X
        self.Z_cache = np.dot(X, self.W) + self.b
        self.A_cache = self.activation(self.Z_cache)
        return self.A_cache

    def backward(self, dA: np.ndarray) -> np.ndarray:
        """
        Backward Pass (Chain Rule):
          dZ = dA ⊙ activation'(Z)
          dW = X^T · dZ
          db = sum(dZ, axis=0)
          dX = dZ · W^T
        """
        dZ = dA * self.activation.derivative(self.Z_cache)
        N = self.X_cache.shape[0]

        self.dW = np.dot(self.X_cache.T, dZ)
        self.db = np.sum(dZ, axis=0, keepdims=True)
        dX = np.dot(dZ, self.W.T)
        return dX


class NeuralNetwork:
    """
    Pure NumPy Multi-Layer Perceptron (MLP) for continuous regression
    or classification.
    """
    def __init__(
        self,
        layer_sizes: List[int],
        activations: List[str],
        seed: int = 42
    ):
        if len(layer_sizes) < 2:
            raise ValueError("Network must have at least an input and an output layer.")
        if len(activations) != len(layer_sizes) - 1:
            raise ValueError(f"Expected {len(layer_sizes)-1} activations for {len(layer_sizes)} layer sizes, got {len(activations)}.")

        self.layer_sizes = list(layer_sizes)
        self.activations = list(activations)
        self.layers: List[DenseLayer] = []
        self.history: Dict[str, List[float]] = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "train_mae": [],
            "val_mae": [],
            "grad_norm": []
        }

        for i in range(len(layer_sizes) - 1):
            layer_seed = (seed + i * 47) if seed is not None else None
            layer = DenseLayer(
                in_features=layer_sizes[i],
                out_features=layer_sizes[i + 1],
                activation_name=activations[i],
                seed=layer_seed
            )
            self.layers.append(layer)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Propagates input forward through all layers."""
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A

    def backward(self, dL_dY: np.ndarray) -> None:
        """Backpropagates gradient from output layer to first layer."""
        dA = dL_dY
        for layer in reversed(self.layers):
            dA = layer.backward(dA)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 60,
        batch_size: int = 32,
        optimizer: Optional[Optimizer] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = False
    ) -> Dict[str, List[float]]:
        """
        Trains network using mini-batch gradient descent.
        """
        if optimizer is None:
            optimizer = AdamOptimizer(lr=0.01)

        optimizer.reset()

        N = X_train.shape[0]
        y_tr = y_train.reshape(-1, 1) if y_train.ndim == 1 else y_train
        y_v = (y_val.reshape(-1, 1) if y_val.ndim == 1 else y_val) if y_val is not None else None

        self.history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "train_mae": [],
            "val_mae": [],
            "grad_norm": []
        }

        for epoch in range(1, epochs + 1):
            # Batch shuffle
            indices = np.random.permutation(N)
            X_shuffled = X_train[indices]
            y_shuffled = y_tr[indices]

            epoch_gnorms = []

            for start_idx in range(0, N, batch_size):
                end_idx = min(start_idx + batch_size, N)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                # 1. Forward Pass
                y_batch_pred = self.forward(X_batch)

                # 2. Output Loss Gradient
                dL_dY = LossFunction.mse_gradient(y_batch_pred, y_batch)

                # 3. Backward Pass
                self.backward(dL_dY)

                # 4. Parameter Updates
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
                epoch_gnorms.append(batch_gnorm)

            # Record epoch statistics
            y_tr_pred = self.forward(X_train)
            train_loss = LossFunction.mse(y_tr_pred, y_tr)
            train_mae = LossFunction.mae(y_tr_pred, y_tr)

            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(train_loss)
            self.history["train_mae"].append(train_mae)
            self.history["grad_norm"].append(float(np.mean(epoch_gnorms)))

            if X_val is not None and y_v is not None:
                y_v_pred = self.forward(X_val)
                val_loss = LossFunction.mse(y_v_pred, y_v)
                val_mae = LossFunction.mae(y_v_pred, y_v)
                self.history["val_loss"].append(val_loss)
                self.history["val_mae"].append(val_mae)
            else:
                self.history["val_loss"].append(train_loss)
                self.history["val_mae"].append(train_mae)

        return self.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Inference returning flat 1D array for continuous output."""
        preds = self.forward(X)
        return preds.ravel() if preds.shape[1] == 1 else preds

    def get_parameter_count(self) -> int:
        """Returns total trainable weights and biases count."""
        return sum(layer.W.size + layer.b.size for layer in self.layers)


def check_network_gradients(
    net: NeuralNetwork,
    X: np.ndarray,
    y: np.ndarray,
    eps: float = 1e-5
) -> Tuple[bool, float]:
    """
    Mathematically verifies analytical backpropagation gradients against
    numerical central difference finite approximations:
      dL/dW ≈ (L(W + eps) - L(W - eps)) / (2 * eps)
    """
    y_target = y.reshape(-1, 1) if y.ndim == 1 else y

    # 1. Forward and analytical backward pass
    y_pred = net.forward(X)
    dL_dY = LossFunction.mse_gradient(y_pred, y_target)
    net.backward(dL_dY)

    max_rel_error = 0.0

    # 2. Check each layer's weights
    for layer in net.layers:
        analytical_dW = layer.dW.copy()
        numerical_dW = np.zeros_like(layer.W)

        it = np.nditer(layer.W, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            orig_val = layer.W[idx]

            layer.W[idx] = orig_val + eps
            loss_plus = LossFunction.mse(net.forward(X), y_target)

            layer.W[idx] = orig_val - eps
            loss_minus = LossFunction.mse(net.forward(X), y_target)

            layer.W[idx] = orig_val  # restore
            numerical_dW[idx] = (loss_plus - loss_minus) / (2.0 * eps)
            it.iternext()

        denom = np.maximum(np.abs(analytical_dW) + np.abs(numerical_dW), 1e-8)
        rel_diff = np.max(np.abs(analytical_dW - numerical_dW) / denom)
        max_rel_error = max(max_rel_error, float(rel_diff))

    is_verified = bool(max_rel_error < 1e-4)
    return is_verified, max_rel_error


def save_numpy_network_weights(net: NeuralNetwork, filepath: str) -> str:
    """Saves raw parameters of the NumPy neural network to disk."""
    bundle = {
        "layer_sizes": net.layer_sizes,
        "activations": net.activations,
        "parameters": [{"W": layer.W, "b": layer.b} for layer in net.layers],
        "parameter_count": net.get_parameter_count(),
        "timestamp": time.time()
    }
    with open(filepath, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    return filepath


def load_numpy_network_weights(filepath: str) -> NeuralNetwork:
    """Reconstructs and populates a NumPy neural network from saved parameters."""
    with open(filepath, "rb") as f:
        bundle = pickle.load(f)

    net = NeuralNetwork(
        layer_sizes=bundle["layer_sizes"],
        activations=bundle["activations"]
    )
    for layer, p in zip(net.layers, bundle["parameters"]):
        layer.W = p["W"].copy()
        layer.b = p["b"].copy()

    return net
