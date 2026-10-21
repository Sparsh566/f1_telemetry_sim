"""
ml Package Initialization
=========================
F1 Telemetry Machine Learning & Deep Learning Suite.
"""

from ml.dataset import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
    FEATURE_LABELS,
    validate_feature_schema,
    build_leak_free_race_dataset,
    split_by_race_group
)

from ml.model_evaluation import (
    calculate_regression_metrics,
    BaselineMeanLapRegressor,
    evaluate_model_against_baseline
)

from ml.model_training import (
    create_f1_regression_pipeline,
    train_f1_lap_predictor
)

from ml.model_persistence import (
    save_model_joblib,
    load_model_joblib,
    save_model_pickle,
    load_model_pickle,
    verify_prediction_equivalence,
    benchmark_serialization_formats,
    SECURITY_WARNING
)

from ml.activations import ACTIVATIONS, ActivationFunction
from ml.losses import LossFunction, check_loss_gradient_finite_diff
from ml.optimizers import SGDOptimizer, SGDMomentumOptimizer, AdamOptimizer

from ml.neural_network_numpy import (
    NumpyStandardScaler,
    SinglePerceptron,
    DenseLayer,
    NeuralNetwork,
    check_network_gradients,
    save_numpy_network_weights,
    load_numpy_network_weights
)

__all__ = [
    "ALL_FEATURES",
    "CATEGORICAL_FEATURES",
    "NUMERICAL_FEATURES",
    "TARGET_COLUMN",
    "FEATURE_LABELS",
    "validate_feature_schema",
    "build_leak_free_race_dataset",
    "split_by_race_group",
    "calculate_regression_metrics",
    "BaselineMeanLapRegressor",
    "evaluate_model_against_baseline",
    "create_f1_regression_pipeline",
    "train_f1_lap_predictor",
    "save_model_joblib",
    "load_model_joblib",
    "save_model_pickle",
    "load_model_pickle",
    "verify_prediction_equivalence",
    "benchmark_serialization_formats",
    "SECURITY_WARNING",
    "ACTIVATIONS",
    "ActivationFunction",
    "LossFunction",
    "check_loss_gradient_finite_diff",
    "SGDOptimizer",
    "SGDMomentumOptimizer",
    "AdamOptimizer",
    "NumpyStandardScaler",
    "SinglePerceptron",
    "DenseLayer",
    "NeuralNetwork",
    "check_network_gradients",
    "save_numpy_network_weights",
    "load_numpy_network_weights"
]
