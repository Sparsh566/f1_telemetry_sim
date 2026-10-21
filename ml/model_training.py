"""
ml/model_training.py
====================
Reproducible Model Training Pipeline for F1 Telemetry Regression.

Trains Scikit-Learn RandomForest pipelines on leak-free race-grouped datasets,
computes R², MAE, RMSE, and benchmark lifts against baseline estimators.
"""

from typing import Dict, Any, Tuple, Optional, List
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from ml.dataset import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
    build_leak_free_race_dataset,
    split_by_race_group,
    validate_feature_schema
)
from ml.model_evaluation import (
    calculate_regression_metrics,
    evaluate_model_against_baseline,
    BaselineMeanLapRegressor
)


def create_f1_regression_pipeline(
    n_estimators: int = 150,
    max_depth: Optional[int] = 15,
    random_state: int = 42
) -> Pipeline:
    """
    Constructs a version-safe Scikit-Learn preprocessing and RandomForest pipeline.
    """
    ohe_kwargs = {"handle_unknown": "ignore"}
    try:
        OneHotEncoder(sparse_output=False)
        ohe_kwargs["sparse_output"] = False
    except TypeError:
        ohe_kwargs["sparse"] = False

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(**ohe_kwargs), CATEGORICAL_FEATURES),
            ("num", StandardScaler(), NUMERICAL_FEATURES)
        ],
        remainder="drop"
    )

    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=1
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", rf)
    ])


def train_f1_lap_predictor(
    df: Optional[pd.DataFrame] = None,
    num_races: int = 40,
    n_estimators: int = 150,
    random_state: int = 42
) -> Tuple[Pipeline, pd.DataFrame, Dict[str, Any], Tuple[pd.DataFrame, pd.Series]]:
    """
    Executes an end-to-end reproducible training workflow:
      1. Generates or validates dataset
      2. Splits strictly by race_id group (no data leakage)
      3. Fits pipeline
      4. Evaluates on held-out test races against BaselineMeanLapRegressor
      5. Extracts feature importances and metadata
    """
    if df is None or df.empty:
        df = build_leak_free_race_dataset(num_races=num_races, random_state=random_state)
    else:
        df = validate_feature_schema(df, expected_features=ALL_FEATURES, require_target=True)

    # Leak-free split by race_id
    train_df, val_df, test_df = split_by_race_group(df, group_col="race_id", random_state=random_state)

    X_train = train_df[ALL_FEATURES]
    y_train = train_df[TARGET_COLUMN].values

    X_val = val_df[ALL_FEATURES]
    y_val = val_df[TARGET_COLUMN].values

    X_test = test_df[ALL_FEATURES]
    y_test = test_df[TARGET_COLUMN].values

    pipeline = create_f1_regression_pipeline(n_estimators=n_estimators, random_state=random_state)
    t0 = time.perf_counter()
    pipeline.fit(X_train, y_train)
    train_duration = time.perf_counter() - t0

    # Evaluate against baseline
    df_compare, eval_summary = evaluate_model_against_baseline(
        model=pipeline,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        model_name="RandomForestRegressor"
    )

    # Validation metrics
    val_pred = pipeline.predict(X_val)
    val_metrics = calculate_regression_metrics(y_val, val_pred, prefix="val_")

    # Feature importances
    rf_step = pipeline.named_steps["regressor"]
    pre_step = pipeline.named_steps["preprocessor"]

    feature_names = []
    try:
        feature_names = list(pre_step.get_feature_names_out())
    except Exception:
        # Fallback names
        feature_names = [f"cat_{c}" for c in CATEGORICAL_FEATURES] + NUMERICAL_FEATURES

    importances = rf_step.feature_importances_
    df_importances = pd.DataFrame({
        "Feature": feature_names[:len(importances)],
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

    metadata: Dict[str, Any] = {
        "model_type": "Scikit-Learn Pipeline (OneHotEncoder + StandardScaler + RandomForestRegressor)",
        "train_races_count": int(train_df["race_id"].nunique()),
        "val_races_count": int(val_df["race_id"].nunique()),
        "test_races_count": int(test_df["race_id"].nunique()),
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "train_duration_sec": round(train_duration, 3),
        "features": ALL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "numerical_features": NUMERICAL_FEATURES,
        "target": TARGET_COLUMN,
        "metrics": {
            **eval_summary["train_metrics"],
            **val_metrics,
            **eval_summary["test_metrics"],
            "baseline_test_mae": eval_summary["baseline_metrics"]["test_mae"],
            "lift_mae_seconds": eval_summary["lift_mae_seconds"]
        },
        "feature_importances": df_importances.to_dict(orient="records"),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    return pipeline, df_compare, metadata, (X_test, pd.Series(y_test, name=TARGET_COLUMN))
