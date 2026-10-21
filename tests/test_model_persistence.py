"""
tests/test_model_persistence.py
===============================
Unit tests for model saving, loading, metadata schema, prediction equivalence,
and error handling for pickle and joblib artifacts.
"""

import os
import tempfile
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from ml.model_persistence import (
    save_model_joblib,
    load_model_joblib,
    save_model_pickle,
    load_model_pickle,
    verify_prediction_equivalence
)
from ml.neural_network_numpy import (
    NeuralNetwork,
    save_numpy_network_weights,
    load_numpy_network_weights
)


def test_joblib_persistence_and_equivalence():
    X = pd.DataFrame({"feat1": [1.0, 2.0, 3.0], "feat2": [4.0, 5.0, 6.0]})
    y = np.array([5.0, 7.0, 9.0])
    model = LinearRegression().fit(X, y)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model.joblib")
        save_model_joblib(model, path, metadata={"model_name": "TestLR"})

        loaded_model, meta = load_model_joblib(path)
        assert meta["model_name"] == "TestLR"
        assert "sklearn_version" in meta

        is_equiv, max_diff = verify_prediction_equivalence(model, loaded_model, X)
        assert is_equiv
        assert max_diff < 1e-6


def test_pickle_persistence_and_equivalence():
    X = pd.DataFrame({"feat1": [1.0, 2.0], "feat2": [3.0, 4.0]})
    y = np.array([10.0, 20.0])
    model = LinearRegression().fit(X, y)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model.pkl")
        save_model_pickle(model, path, metadata={"test_run": True})

        loaded_model, meta = load_model_pickle(path)
        assert meta["test_run"] is True

        is_equiv, max_diff = verify_prediction_equivalence(model, loaded_model, X)
        assert is_equiv
        assert max_diff < 1e-6


def test_numpy_weights_save_and_load():
    net = NeuralNetwork(layer_sizes=[2, 4, 1], activations=["ReLU", "Linear"], seed=42)
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    preds_before = net.predict(X)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "numpy_net.pkl")
        save_numpy_network_weights(net, path)

        loaded_net = load_numpy_network_weights(path)
        preds_after = loaded_net.predict(X)

        np.testing.assert_allclose(preds_before, preds_after, atol=1e-6)


def test_missing_artifact_raises_error():
    caught_joblib = False
    try:
        load_model_joblib("non_existent_file.joblib")
    except FileNotFoundError:
        caught_joblib = True
    assert caught_joblib, "Expected FileNotFoundError for missing joblib file"

    caught_pickle = False
    try:
        load_model_pickle("non_existent_file.pkl")
    except FileNotFoundError:
        caught_pickle = True
    assert caught_pickle, "Expected FileNotFoundError for missing pickle file"
