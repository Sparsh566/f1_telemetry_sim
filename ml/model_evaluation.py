"""
ml/model_evaluation.py
======================
Standardized Evaluation Metrics & Comparative Benchmarking for F1 Telemetry.

Computes R², MAE, RMSE, and compares ML models against an empirical
baseline (mean training lap time).
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def calculate_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str = ""
) -> Dict[str, float]:
    """
    Computes R², MAE, and RMSE for regression predictions.
    """
    y_true_arr = np.asarray(y_true).ravel()
    y_pred_arr = np.asarray(y_pred).ravel()

    if len(y_true_arr) == 0:
        return {f"{prefix}r2": 0.0, f"{prefix}mae": 0.0, f"{prefix}rmse": 0.0}

    r2 = float(r2_score(y_true_arr, y_pred_arr))
    mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
    rmse = float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)))

    return {
        f"{prefix}r2": round(r2, 4),
        f"{prefix}mae": round(mae, 4),
        f"{prefix}rmse": round(rmse, 4)
    }


class BaselineMeanLapRegressor:
    """
    Simple naive baseline predicting the mean training lap time.
    Provides a scientific benchmark to ensure ML models deliver real lift.
    """
    def __init__(self):
        self.mean_lap_: float = 0.0

    def fit(self, X: Any, y: np.ndarray) -> "BaselineMeanLapRegressor":
        self.mean_lap_ = float(np.mean(y))
        return self

    def predict(self, X: Any) -> np.ndarray:
        n_samples = len(X)
        return np.full(shape=(n_samples,), fill_value=self.mean_lap_)


def evaluate_model_against_baseline(
    model: Any,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    model_name: str = "Trained Model"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Evaluates any fitted estimator against the simple mean baseline on identical test data.
    """
    # 1. Model predictions
    y_test_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)

    train_metrics = calculate_regression_metrics(y_train, y_train_pred, prefix="train_")
    test_metrics = calculate_regression_metrics(y_test, y_test_pred, prefix="test_")

    # 2. Baseline model
    baseline = BaselineMeanLapRegressor().fit(X_train, y_train)
    y_base_pred = baseline.predict(X_test)
    base_metrics = calculate_regression_metrics(y_test, y_base_pred, prefix="test_")

    # 3. Create comparison summary table
    df_compare = pd.DataFrame([
        {
            "Model": "Baseline (Training Set Mean)",
            "Test R²": base_metrics["test_r2"],
            "Test MAE (s)": base_metrics["test_mae"],
            "Test RMSE (s)": base_metrics["test_rmse"],
            "Status": "Reference Baseline"
        },
        {
            "Model": model_name,
            "Test R²": test_metrics["test_r2"],
            "Test MAE (s)": test_metrics["test_mae"],
            "Test RMSE (s)": test_metrics["test_rmse"],
            "Status": "Evaluated Estimator"
        }
    ])

    summary = {
        "model_name": model_name,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "baseline_metrics": base_metrics,
        "lift_mae_seconds": round(base_metrics["test_mae"] - test_metrics["test_mae"], 4)
    }

    return df_compare, summary
