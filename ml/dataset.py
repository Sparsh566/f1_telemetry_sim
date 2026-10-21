"""
ml/dataset.py
=============
Leak-Free Telemetry Dataset Generator & Schema Validation for F1 Simulation.

Provides unified feature schemas between training and live inference,
group-based splitting by race_id (to prevent car-in-same-race data leakage),
and robust input validation.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Unified Feature Schema Constants
# -----------------------------------------------------------------------------
CATEGORICAL_FEATURES = ["tyre"]
NUMERICAL_FEATURES = [
    "speed",
    "throttle",
    "brake",
    "tyre_wear",
    "fuel",
    "brake_temp",
    "aggression",
    "pit_stops",
    "incident_count"
]
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
TARGET_COLUMN = "lap_time"

VALID_TYRE_COMPOUNDS = ["Soft", "Medium", "Hard", "Inter", "Wet"]

FEATURE_LABELS = {
    "tyre": "🛞 Tyre Compound",
    "speed": "⚡ Speed (km/h)",
    "throttle": "🟢 Throttle (%)",
    "brake": "🛑 Brake (%)",
    "tyre_wear": "📉 Tyre Wear (%)",
    "fuel": "⛽ Fuel Remaining (%)",
    "brake_temp": "🌡️ Brake Temp (°C)",
    "aggression": "🔥 Driver Aggression",
    "pit_stops": "🛑 Pit Stops Completed",
    "incident_count": "⚠️ Cumulative Incidents"
}


def validate_feature_schema(
    df: pd.DataFrame,
    expected_features: Optional[List[str]] = None,
    require_target: bool = False
) -> pd.DataFrame:
    """
    Validates feature columns, data types, missing values, and column order.
    
    Raises:
        ValueError: If required columns are missing, have invalid types,
                    or contain NaNs.
    """
    if expected_features is None:
        expected_features = ALL_FEATURES

    if df.empty:
        raise ValueError("Input DataFrame is empty. Expected at least one row of telemetry.")

    missing_cols = [c for c in expected_features if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Schema mismatch: missing required feature columns: {missing_cols}")

    if require_target and TARGET_COLUMN not in df.columns:
        raise ValueError(f"Schema mismatch: missing target column '{TARGET_COLUMN}'")

    # Check for NaN / null values
    cols_to_check = expected_features + ([TARGET_COLUMN] if require_target else [])
    null_counts = df[cols_to_check].isnull().sum()
    null_cols = null_counts[null_counts > 0].to_dict()
    if null_cols:
        raise ValueError(f"Input data contains null/NaN values in columns: {null_cols}")

    # Enforce numeric types
    for col in NUMERICAL_FEATURES:
        if col in expected_features:
            if not pd.api.types.is_numeric_dtype(df[col]):
                try:
                    df[col] = pd.to_numeric(df[col])
                except Exception as e:
                    raise ValueError(f"Feature '{col}' must be numeric, got incompatible values: {e}")

    # Validate categorical categories
    if "tyre" in expected_features:
        unknown_tyres = set(df["tyre"].unique()) - set(VALID_TYRE_COMPOUNDS)
        if unknown_tyres:
            # Re-map or warn on unknown
            pass

    # Ensure consistent column ordering
    output_cols = [c for c in expected_features if c in df.columns]
    if require_target:
        output_cols.append(TARGET_COLUMN)
    if "race_id" in df.columns:
        output_cols.append("race_id")
    if "car" in df.columns:
        output_cols.append("car")

    return df[output_cols].copy()


def build_leak_free_race_dataset(
    num_races: int = 50,
    samples_per_race: int = 40,
    random_state: int = 42,
    base_lap: float = 85.0
) -> pd.DataFrame:
    """
    Generates a synthetic telemetry dataset guaranteed to use identical
    feature semantics as live simulation inference.
    
    Each simulated race receives an integer `race_id`. Features represent
    instantaneous or cumulative telemetry captured consistently across stints.
    """
    rng = np.random.RandomState(random_state)
    drivers = ["VER", "HAM", "LEC", "NOR", "PIA", "RUS", "ALO", "SAI"]
    tyre_compounds = ["Soft", "Medium", "Hard"]
    tyre_modifier = {"Soft": 1.00, "Medium": 0.97, "Hard": 0.94}

    records = []

    for race_idx in range(num_races):
        # Race-level environmental conditions
        ambient_temp = rng.uniform(18.0, 32.0)
        track_grip_scale = rng.uniform(0.96, 1.03)

        for driver in drivers:
            driver_aggression = rng.uniform(0.90, 1.20)
            chosen_tyre = rng.choice(tyre_compounds, p=[0.35, 0.45, 0.20])

            # Stint progression
            fuel_level = 100.0
            tyre_wear = rng.uniform(0.01, 0.08)
            pit_stops = 0
            incident_count = 0

            for sample_idx in range(samples_per_race):
                # Throttle and brake dynamics
                throttle = rng.uniform(0.35, 1.0) * driver_aggression
                throttle = float(np.clip(throttle, 0.0, 1.0))

                is_braking = rng.random() < 0.25
                brake = float(rng.uniform(0.3, 1.0) if is_braking else 0.0)

                # Speed derived from throttle and track position
                speed = rng.uniform(110.0, 340.0) * (0.6 + 0.4 * throttle)
                speed = float(np.clip(speed, 60.0, 355.0))

                # Brake temp
                brake_temp = float(np.clip(250.0 + brake * 450.0 + rng.normal(0, 25), 100.0, 900.0))

                # Tyre degradation & fuel consumption
                tyre_wear = float(np.clip(tyre_wear + rng.uniform(0.005, 0.02) * driver_aggression, 0.0, 1.0))
                fuel_level = float(np.clip(fuel_level - rng.uniform(0.4, 1.2), 0.0, 100.0))

                # Possible pit stop if wear gets too high
                if tyre_wear > 0.70 and rng.random() < 0.3:
                    pit_stops += 1
                    tyre_wear = float(rng.uniform(0.02, 0.06))
                    chosen_tyre = rng.choice(tyre_compounds)

                # Possible minor incident
                if rng.random() < (0.015 * driver_aggression):
                    incident_count += 1

                # Physical Ground Truth Lap Pace Calculation:
                # lap_time = base_lap + grip_penalty + fuel_penalty + tyre_penalty + aggression_effect + noise
                grip = (1.0 - tyre_wear * 0.45) * tyre_modifier.get(chosen_tyre, 0.95) * track_grip_scale
                grip_penalty = (1.0 - grip) * 7.5
                fuel_penalty = (fuel_level / 100.0) * 1.8  # ~1.8 seconds full fuel effect
                aggression_bonus = -(driver_aggression - 1.0) * 0.8  # Faster lap for higher attack
                incident_penalty = incident_count * 2.5
                wear_exponential = 2.0 * (tyre_wear ** 3)
                noise = rng.normal(0.0, 0.25)

                sim_lap_time = (
                    base_lap
                    + grip_penalty
                    + fuel_penalty
                    + aggression_bonus
                    + incident_penalty
                    + wear_exponential
                    + noise
                )
                sim_lap_time = float(np.clip(sim_lap_time, 65.0, 130.0))

                records.append({
                    "race_id": race_idx,
                    "car": driver,
                    "tyre": chosen_tyre,
                    "speed": round(speed, 1),
                    "throttle": round(throttle * 100.0, 1),
                    "brake": round(brake * 100.0, 1),
                    "tyre_wear": round(tyre_wear * 100.0, 1),
                    "fuel": round(fuel_level, 1),
                    "brake_temp": round(brake_temp, 1),
                    "aggression": round(driver_aggression, 2),
                    "pit_stops": pit_stops,
                    "incident_count": incident_count,
                    "lap_time": round(sim_lap_time, 3)
                })

    df = pd.DataFrame(records)
    return validate_feature_schema(df, expected_features=ALL_FEATURES, require_target=True)


def split_by_race_group(
    df: pd.DataFrame,
    group_col: str = "race_id",
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Performs group-based splitting strictly by race_id.
    Guarantees that cars from the same simulated race NEVER appear
    in both training and testing sets, preventing synthetic data leakage.
    """
    if group_col not in df.columns:
        raise ValueError(f"Group column '{group_col}' not found in DataFrame.")

    unique_races = np.sort(df[group_col].unique())
    n_races = len(unique_races)

    if n_races < 3:
        raise ValueError(f"Need at least 3 unique races to split into train/val/test; found {n_races}.")

    rng = np.random.RandomState(random_state)
    shuffled_races = rng.permutation(unique_races)

    n_train = max(1, int(n_races * train_frac))
    n_val = max(1, int(n_races * val_frac))

    train_races = set(shuffled_races[:n_train])
    val_races = set(shuffled_races[n_train:n_train + n_val])
    test_races = set(shuffled_races[n_train + n_val:])

    # Fallback if rounding left test empty
    if not test_races:
        test_races = {shuffled_races[-1]}
        train_races.discard(shuffled_races[-1])

    train_df = df[df[group_col].isin(train_races)].copy()
    val_df = df[df[group_col].isin(val_races)].copy()
    test_df = df[df[group_col].isin(test_races)].copy()

    return train_df, val_df, test_df
