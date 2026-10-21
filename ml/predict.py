"""
ml/predict.py
=============
Standalone Inference Script for F1 Telemetry Predictions.

Allows loading a saved model artifact (.joblib or .pkl) and running
predictions on telemetry features independently of Streamlit.

Usage (CLI):
    python ml/predict.py --model models/f1_rf_pipeline.joblib
"""

import sys
import os
import argparse
from typing import Dict, Any, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd

from ml.dataset import ALL_FEATURES, validate_feature_schema
from ml.model_persistence import load_model_joblib, load_model_pickle


def predict_lap_time(
    model: Any,
    telemetry_df: pd.DataFrame,
    metadata: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Validates input features and returns predictions alongside input telemetry.
    """
    valid_df = validate_feature_schema(telemetry_df, expected_features=ALL_FEATURES, require_target=False)
    preds = model.predict(valid_df[ALL_FEATURES])

    res = telemetry_df.copy()
    res["predicted_lap_time"] = np.round(preds, 3)

    return res


def main():
    parser = argparse.ArgumentParser(description="Run F1 Lap Time Predictions from Serialized Model.")
    parser.add_argument("--model", type=str, required=True, help="Path to saved .joblib or .pkl model artifact.")
    parser.add_argument("--format", type=str, default="auto", choices=["auto", "joblib", "pickle"], help="Serialization format.")
    args = parser.parse_args()

    fmt = args.format
    if fmt == "auto":
        fmt = "pickle" if args.model.endswith(".pkl") else "joblib"

    print(f"Loading model from: {args.model} (format={fmt})...")
    if fmt == "joblib":
        model, metadata = load_model_joblib(args.model)
    else:
        model, metadata = load_model_pickle(args.model)

    print("Model loaded successfully!")
    if metadata:
        print(f"Model Type: {metadata.get('model_type', 'Unknown')}")
        print(f"Created At: {metadata.get('created_at_utc', 'Unknown')}")
        if "metrics" in metadata:
            metrics = metadata["metrics"]
            print(f"Trained Performance -> Test R²: {metrics.get('test_r2', 'N/A')}, Test MAE: {metrics.get('test_mae', 'N/A')}s")

    # Sample input for 2025 grid
    sample_data = pd.DataFrame([
        {
            "car": "VER",
            "tyre": "Soft",
            "speed": 315.4,
            "throttle": 98.0,
            "brake": 0.0,
            "tyre_wear": 18.5,
            "fuel": 45.0,
            "brake_temp": 480.0,
            "aggression": 1.15,
            "pit_stops": 1,
            "incident_count": 0
        },
        {
            "car": "HAM",
            "tyre": "Medium",
            "speed": 298.2,
            "throttle": 92.0,
            "brake": 0.0,
            "tyre_wear": 34.0,
            "fuel": 52.0,
            "brake_temp": 420.0,
            "aggression": 1.05,
            "pit_stops": 1,
            "incident_count": 0
        },
        {
            "car": "NOR",
            "tyre": "Hard",
            "speed": 290.0,
            "throttle": 88.0,
            "brake": 0.0,
            "tyre_wear": 52.0,
            "fuel": 68.0,
            "brake_temp": 405.0,
            "aggression": 1.00,
            "pit_stops": 0,
            "incident_count": 0
        }
    ])

    print("\nRunning sample predictions on telemetry input:")
    results = predict_lap_time(model, sample_data, metadata)
    print(results[["car", "tyre", "speed", "tyre_wear", "fuel", "predicted_lap_time"]].to_string(index=False))


if __name__ == "__main__":
    main()
