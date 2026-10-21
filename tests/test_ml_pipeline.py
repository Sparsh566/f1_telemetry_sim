"""
tests/test_ml_pipeline.py
=========================
Unit tests for data leakage prevention (group-based race splitting),
feature schema validation, baseline comparison, and training pipeline.
"""

import numpy as np
import pandas as pd

from ml.dataset import (
    ALL_FEATURES,
    build_leak_free_race_dataset,
    split_by_race_group,
    validate_feature_schema
)
from ml.model_evaluation import BaselineMeanLapRegressor, calculate_regression_metrics
from ml.model_training import train_f1_lap_predictor
from ml.predict import predict_lap_time


def test_race_group_split_prevents_leakage():
    df = build_leak_free_race_dataset(num_races=10, samples_per_race=5, random_state=42)
    train_df, val_df, test_df = split_by_race_group(df, group_col="race_id", random_state=42)

    train_races = set(train_df["race_id"].unique())
    val_races = set(val_df["race_id"].unique())
    test_races = set(test_df["race_id"].unique())

    # Crucial check: all sets of race_id must be completely disjoint!
    assert train_races.isdisjoint(val_races), "Data leakage: train and val share race_ids!"
    assert train_races.isdisjoint(test_races), "Data leakage: train and test share race_ids!"
    assert val_races.isdisjoint(test_races), "Data leakage: val and test share race_ids!"


def test_schema_validation_detects_missing_and_empty():
    # Empty DataFrame
    caught_empty = False
    try:
        validate_feature_schema(pd.DataFrame())
    except ValueError as e:
        caught_empty = "empty" in str(e).lower()
    assert caught_empty, "Expected ValueError for empty DataFrame"

    # Missing required column
    caught_missing = False
    bad_df = pd.DataFrame({"speed": [300.0], "tyre": ["Soft"]})
    try:
        validate_feature_schema(bad_df, expected_features=ALL_FEATURES)
    except ValueError as e:
        caught_missing = "missing required" in str(e).lower()
    assert caught_missing, "Expected ValueError for missing columns"

    # Missing target when require_target=True
    caught_no_target = False
    valid_no_target = pd.DataFrame({col: [1.0] for col in ALL_FEATURES})
    valid_no_target["tyre"] = ["Soft"]
    try:
        validate_feature_schema(valid_no_target, require_target=True)
    except ValueError as e:
        caught_no_target = "missing target" in str(e).lower()
    assert caught_no_target, "Expected ValueError for missing target"


def test_baseline_mean_regressor():
    y_train = np.array([80.0, 82.0, 84.0])
    baseline = BaselineMeanLapRegressor().fit(None, y_train)

    X_test = pd.DataFrame({"dummy": [1, 2, 3, 4]})
    preds = baseline.predict(X_test)

    # Expected mean is 82.0 for all predictions
    np.testing.assert_allclose(preds, [82.0, 82.0, 82.0, 82.0])


def test_training_and_inference_feature_consistency():
    # Small test training run
    model, df_compare, meta, (X_test, y_test) = train_f1_lap_predictor(
        num_races=6,
        n_estimators=10,
        random_state=42
    )

    assert "RandomForestRegressor" in df_compare["Model"].values
    assert "Baseline (Training Set Mean)" in df_compare["Model"].values

    # Test standalone prediction using same schema
    preds_df = predict_lap_time(model, X_test.head(5))
    assert "predicted_lap_time" in preds_df.columns
    assert not preds_df["predicted_lap_time"].isnull().any()
