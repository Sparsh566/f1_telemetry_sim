"""
f1_ensemble_analytics.py
========================
High-Performance Ensemble Machine Learning & Model Diagnostics Engine for Formula 1 Telemetry.

Comprehensive implementation for:
  1. Ensemble Methods:
     - Bagging (Random Forest, Bagging Regressor)
     - Boosting (XGBoost, LightGBM, Gradient Boosting)
  2. Bias-Variance Decomposition & Trade-off Analysis.
  3. Overfitting vs. Underfitting Detection & Diagnostics.
  4. L1 (Lasso) and L2 (Ridge) Regularization Dynamics.
"""

from typing import Dict, Any, List, Tuple, Optional
import time
import numpy as np
import pandas as pd

# Lazy loader for machine learning dependencies
def get_ensemble_modules():
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import (
        RandomForestRegressor,
        BaggingRegressor,
        GradientBoostingRegressor,
        HistGradientBoostingRegressor
    )
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    from sklearn.preprocessing import StandardScaler
    
    # Try importing XGBoost
    xgb_available = False
    XGBRegressor = None
    try:
        import xgboost as xgb
        XGBRegressor = xgb.XGBRegressor
        xgb_available = True
    except Exception:
        pass

    # Try importing LightGBM
    lgb_available = False
    LGBMRegressor = None
    try:
        import lightgbm as lgb
        LGBMRegressor = lgb.LGBMRegressor
        lgb_available = True
    except Exception:
        pass

    return {
        "train_test_split": train_test_split,
        "RandomForestRegressor": RandomForestRegressor,
        "BaggingRegressor": BaggingRegressor,
        "GradientBoostingRegressor": GradientBoostingRegressor,
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor,
        "DecisionTreeRegressor": DecisionTreeRegressor,
        "r2_score": r2_score,
        "mae": mean_absolute_error,
        "mse": mean_squared_error,
        "StandardScaler": StandardScaler,
        "xgb_available": xgb_available,
        "XGBRegressor": XGBRegressor,
        "lgb_available": lgb_available,
        "LGBMRegressor": LGBMRegressor
    }


# -----------------------------------------------------------------------------
# 1. Telemetry Dataset Generation & Preprocessing
# -----------------------------------------------------------------------------
TELEMETRY_FEATURES = [
    "speed",
    "throttle",
    "brake",
    "tyre_wear",
    "fuel",
    "brake_temp",
    "aggression"
]

FEATURE_LABELS = {
    "speed": "⚡ Speed (km/h)",
    "throttle": "🟢 Throttle (%)",
    "brake": "🛑 Brake (%)",
    "tyre_wear": "📉 Tyre Wear (%)",
    "fuel": "⛽ Fuel Level (%)",
    "brake_temp": "🌡️ Brake Temp (°C)",
    "aggression": "🔥 Driver Aggression"
}

def generate_telemetry_regression_dataset(n_samples: int = 1000, noise_level: float = 0.8, random_state: int = 42) -> pd.DataFrame:
    """
    Generates a realistic Formula 1 telemetry dataset modeling lap time
    as a non-linear function of tyre degradation, vehicle weight (fuel),
    speed, throttle, brake thermals, and driver aggression.
    """
    rng = np.random.RandomState(random_state)
    
    speed = rng.uniform(160.0, 340.0, size=n_samples)
    throttle = np.clip(speed / 340.0 + rng.normal(0, 0.08, size=n_samples), 0.1, 1.0)
    brake = np.clip((340.0 - speed) / 340.0 * 0.8 + rng.normal(0, 0.05, size=n_samples), 0.0, 1.0)
    tyre_wear = rng.uniform(0.05, 0.85, size=n_samples)
    fuel = rng.uniform(10.0, 100.0, size=n_samples)
    brake_temp = 300.0 + brake * 550.0 + rng.normal(0, 25.0, size=n_samples)
    aggression = rng.uniform(0.85, 1.25, size=n_samples)
    
    # Physics-based lap pace computation
    base_lap = 85.0
    speed_factor = - (speed - 240.0) * 0.06
    fuel_penalty = (fuel / 100.0) * 2.8
    tyre_penalty = (tyre_wear ** 1.7) * 4.5
    thermal_penalty = np.where(brake_temp > 750.0, (brake_temp - 750.0) * 0.015, 0.0)
    aggression_bonus = - (aggression - 1.0) * 2.2
    
    noise = rng.normal(0, noise_level, size=n_samples)
    lap_time = base_lap + speed_factor + fuel_penalty + tyre_penalty + thermal_penalty + aggression_bonus + noise
    
    compounds = ["Soft", "Medium", "Hard"]
    tyre_compound = rng.choice(compounds, size=n_samples, p=[0.4, 0.4, 0.2])
    
    df = pd.DataFrame({
        "speed": np.round(speed, 2),
        "throttle": np.round(throttle * 100.0, 1),
        "brake": np.round(brake * 100.0, 1),
        "tyre_wear": np.round(tyre_wear * 100.0, 1),
        "fuel": np.round(fuel, 1),
        "brake_temp": np.round(brake_temp, 1),
        "aggression": np.round(aggression, 3),
        "tyre_compound": tyre_compound,
        "lap_time": np.round(lap_time, 3)
    })
    return df


def extract_live_telemetry_for_ensemble(state: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Extracts live telemetry from current simulator state and history.
    """
    rows = []
    for driver_code, car in state.items():
        base_info = {
            "driver": driver_code,
            "tyre_compound": car.get("tyre", "Medium"),
            "aggression": float(car.get("aggression", 1.0))
        }
        for point in car.get("history", []):
            speed = float(point.get("speed", 250.0))
            throttle = float(point.get("throttle", 0.8)) * 100.0
            brake = float(point.get("brake", 0.1)) * 100.0
            brake_temp = float(point.get("brake_temp", 450.0))
            tyre_wear = float(point.get("tyre_wear", 0.2)) * 100.0
            fuel = float(point.get("fuel", 50.0))
            lap_time = float(point.get("lap_time", 85.0))
            
            rows.append({
                **base_info,
                "speed": round(speed, 2),
                "throttle": round(throttle, 1),
                "brake": round(brake, 1),
                "tyre_wear": round(tyre_wear, 1),
                "fuel": round(fuel, 1),
                "brake_temp": round(brake_temp, 1),
                "lap_time": round(lap_time, 3)
            })
            
    if not rows:
        return generate_telemetry_regression_dataset(600)
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# 2. Ensemble Benchmark Suite: Bagging vs. Boosting
# -----------------------------------------------------------------------------
def benchmark_ensemble_models(df: pd.DataFrame, test_size: float = 0.25, random_state: int = 42) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Trains and benchmarks Bagging (Random Forest, Bagging Regressor) and
    Boosting (XGBoost, LightGBM, Gradient Boosting) on F1 telemetry.
    Returns metrics table and fitted model objects.
    """
    mods = get_ensemble_modules()
    X = df[TELEMETRY_FEATURES].copy()
    y = df["lap_time"].values
    
    X_train, X_test, y_train, y_test = mods["train_test_split"](X, y, test_size=test_size, random_state=random_state)
    
    models: Dict[str, Any] = {}
    
    # 1. Base Single Decision Tree (Baseline)
    models["Single Decision Tree (Baseline)"] = mods["DecisionTreeRegressor"](max_depth=6, random_state=random_state)
    
    # 2. Bagging Regressor
    models["Bagging (Bootstrap Aggregation)"] = mods["BaggingRegressor"](
        estimator=mods["DecisionTreeRegressor"](max_depth=6),
        n_estimators=35,
        random_state=random_state
    )
    
    # 3. Random Forest (Bagging + Feature Subspace)
    models["Random Forest (Bagging)"] = mods["RandomForestRegressor"](
        n_estimators=50,
        max_depth=8,
        random_state=random_state
    )
    
    # 4. XGBoost (Boosting)
    if mods["xgb_available"]:
        models["XGBoost (Gradient Boosting)"] = mods["XGBRegressor"](
            n_estimators=60,
            max_depth=4,
            learning_rate=0.08,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=random_state,
            n_jobs=1
        )
    else:
        models["Gradient Boosting (XGBoost Alt)"] = mods["GradientBoostingRegressor"](
            n_estimators=60,
            max_depth=4,
            learning_rate=0.08,
            random_state=random_state
        )
        
    # 5. LightGBM (Histogram Boosting)
    if mods["lgb_available"]:
        models["LightGBM (Histogram Boosting)"] = mods["LGBMRegressor"](
            n_estimators=60,
            max_depth=5,
            learning_rate=0.08,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=random_state,
            verbose=-1,
            n_jobs=1
        )
    else:
        models["HistGradientBoosting (LightGBM Alt)"] = mods["HistGradientBoostingRegressor"](
            max_iter=60,
            max_depth=5,
            learning_rate=0.08,
            l2_regularization=1.0,
            random_state=random_state
        )
        
    results = []
    fitted_models = {}
    
    for name, model in models.items():
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        fit_time_ms = (time.perf_counter() - t0) * 1000.0
        
        t0_pred = time.perf_counter()
        y_pred = model.predict(X_test)
        pred_time_ms = (time.perf_counter() - t0_pred) * 1000.0
        
        r2 = mods["r2_score"](y_test, y_pred)
        mae_val = mods["mae"](y_test, y_pred)
        rmse_val = np.sqrt(mods["mse"](y_test, y_pred))
        
        if "Baseline" in name:
            paradigm = "Single Tree"
            mechanism = "Single greedy split (High Variance risk)"
        elif "Bagging" in name or "Random Forest" in name:
            paradigm = "Bagging (Variance Reducer)"
            mechanism = "Parallel bootstrap trees + feature sub-sampling"
        else:
            paradigm = "Boosting (Bias Reducer)"
            mechanism = "Sequential residual correction + learning rate"
            
        results.append({
            "Model Name": name,
            "Paradigm": paradigm,
            "Mechanism": mechanism,
            "R² Score": round(float(r2), 4),
            "MAE (seconds)": round(float(mae_val), 3),
            "RMSE (seconds)": round(float(rmse_val), 3),
            "Train Latency (ms)": round(float(fit_time_ms), 1),
            "Predict Latency (ms)": round(float(pred_time_ms), 2)
        })
        fitted_models[name] = model
        
    results_df = pd.DataFrame(results).sort_values(by="R² Score", ascending=False).reset_index(drop=True)
    return results_df, fitted_models


# -----------------------------------------------------------------------------
# 3. Bias-Variance Decomposition Engine
# -----------------------------------------------------------------------------
def compute_bias_variance_tradeoff(
    df: pd.DataFrame,
    model_family: str = "Random Forest",
    depth_range: Optional[List[int]] = None,
    n_bootstraps: int = 12,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Computes empirical Bias^2, Variance, and Expected Mean Squared Error
    across model complexity (tree depth) using bootstrap re-sampling.
    
    Formula:
        Expected Error = Bias^2 + Variance + Irreducible Noise
    """
    if depth_range is None:
        depth_range = [1, 2, 3, 5, 7, 10, 14]

    mods = get_ensemble_modules()
    X = df[TELEMETRY_FEATURES].values
    y = df["lap_time"].values
    
    X_train_full, X_test, y_train_full, y_test = mods["train_test_split"](
        X, y, test_size=0.3, random_state=random_state
    )
    
    n_test = len(y_test)
    records = []
    
    for depth in depth_range:
        predictions = np.zeros((n_bootstraps, n_test))
        train_errors = []
        
        for b in range(n_bootstraps):
            boot_idx = np.random.RandomState(random_state + b * 13).choice(
                len(X_train_full), size=len(X_train_full), replace=True
            )
            X_boot = X_train_full[boot_idx]
            y_boot = y_train_full[boot_idx]
            
            if model_family == "Single Decision Tree":
                model = mods["DecisionTreeRegressor"](max_depth=depth, random_state=b)
            elif model_family == "Bagging (Random Forest)":
                model = mods["RandomForestRegressor"](n_estimators=20, max_depth=depth, random_state=b)
            elif model_family == "Boosting (XGBoost)":
                if mods["xgb_available"]:
                    model = mods["XGBRegressor"](n_estimators=25, max_depth=depth, learning_rate=0.1, random_state=b, n_jobs=1)
                else:
                    model = mods["GradientBoostingRegressor"](n_estimators=25, max_depth=depth, learning_rate=0.1, random_state=b)
            else: # LightGBM
                if mods["lgb_available"]:
                    model = mods["LGBMRegressor"](n_estimators=25, max_depth=depth, learning_rate=0.1, random_state=b, verbose=-1, n_jobs=1)
                else:
                    model = mods["HistGradientBoostingRegressor"](max_iter=25, max_depth=depth, learning_rate=0.1, random_state=b)
                    
            model.fit(X_boot, y_boot)
            train_preds = model.predict(X_boot)
            train_errors.append(mods["mse"](y_boot, train_preds))
            predictions[b, :] = model.predict(X_test)
            
        # Expected prediction: E[f_hat(x)]
        expected_pred = np.mean(predictions, axis=0)
        
        # Bias^2 = mean((E[f_hat(x)] - y_true)^2)
        bias_squared = float(np.mean((expected_pred - y_test) ** 2))
        
        # Variance = mean(var(predictions across models))
        variance = float(np.mean(np.var(predictions, axis=0)))
        
        total_test_mse = float(np.mean([mods["mse"](y_test, predictions[b]) for b in range(n_bootstraps)]))
        avg_train_mse = float(np.mean(train_errors))
        
        records.append({
            "Tree Depth": depth,
            "Train MSE": round(avg_train_mse, 3),
            "Test MSE": round(total_test_mse, 3),
            "Bias² (Underfitting Error)": round(bias_squared, 3),
            "Variance (Overfitting Error)": round(variance, 3),
            "Bias² + Variance": round(bias_squared + variance, 3)
        })
        
    return pd.DataFrame(records)


# -----------------------------------------------------------------------------
# 4. Overfitting vs Underfitting Diagnostics & Generalization Gap
# -----------------------------------------------------------------------------
def analyze_overfitting_underfitting(
    df: pd.DataFrame,
    max_depth: int = 12,
    n_estimators: int = 50,
    min_samples_split: int = 2,
    learning_rate: float = 0.1,
    reg_alpha: float = 0.0,
    reg_lambda: float = 0.0,
    model_type: str = "Random Forest"
) -> Dict[str, Any]:
    """
    Evaluates train vs. test metrics to diagnose Underfitting,
    Overfitting, or Optimal Generalization.
    """
    mods = get_ensemble_modules()
    X = df[TELEMETRY_FEATURES].values
    y = df["lap_time"].values
    
    X_train, X_test, y_train, y_test = mods["train_test_split"](X, y, test_size=0.25, random_state=42)
    
    if model_type == "Decision Tree":
        model = mods["DecisionTreeRegressor"](
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42
        )
    elif model_type == "Random Forest":
        model = mods["RandomForestRegressor"](
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42
        )
    elif model_type == "XGBoost":
        if mods["xgb_available"]:
            model = mods["XGBRegressor"](
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                random_state=42,
                n_jobs=1
            )
        else:
            model = mods["GradientBoostingRegressor"](
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=42
            )
    else:  # LightGBM
        if mods["lgb_available"]:
            model = mods["LGBMRegressor"](
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                random_state=42,
                verbose=-1,
                n_jobs=1
            )
        else:
            model = mods["HistGradientBoostingRegressor"](
                max_iter=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                l2_regularization=reg_lambda,
                random_state=42
            )

    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    train_r2 = float(mods["r2_score"](y_train, y_pred_train))
    test_r2 = float(mods["r2_score"](y_test, y_pred_test))
    
    train_mse = float(mods["mse"](y_train, y_pred_train))
    test_mse = float(mods["mse"](y_test, y_pred_test))
    
    train_mae = float(mods["mae"](y_train, y_pred_train))
    test_mae = float(mods["mae"](y_test, y_pred_test))
    
    gen_gap_r2 = train_r2 - test_r2
    gen_gap_mse = test_mse - train_mse
    
    # State classification
    if train_r2 < 0.65 and test_r2 < 0.65:
        diagnosis = "⚠️ High Bias (Underfitting)"
        diagnosis_color = "warning"
        explanation = "The model is too simplistic to capture telemetry dynamics (speed, aero, tyre degradation). Both train and test errors are high."
    elif gen_gap_r2 > 0.12 or (test_mse / (train_mse + 1e-6) > 2.0):
        diagnosis = "🚨 High Variance (Overfitting)"
        diagnosis_color = "error"
        explanation = f"The model has memorized idiosyncrasies of training laps. Notice the large train-test gap (Δ R² = {gen_gap_r2:.3f}). It fails to generalize."
    else:
        diagnosis = "✅ Optimal Generalization (Balanced)"
        diagnosis_color = "success"
        explanation = f"Balanced trade-off: Model achieves strong performance on unseen test laps (Test R²: {test_r2:.3f}) with low generalization gap."
        
    return {
        "train_r2": round(train_r2, 4),
        "test_r2": round(test_r2, 4),
        "train_mse": round(train_mse, 3),
        "test_mse": round(test_mse, 3),
        "train_mae": round(train_mae, 3),
        "test_mae": round(test_mae, 3),
        "gen_gap_r2": round(gen_gap_r2, 4),
        "gen_gap_mse": round(gen_gap_mse, 3),
        "diagnosis": diagnosis,
        "diagnosis_color": diagnosis_color,
        "explanation": explanation,
        "model": model
    }


# -----------------------------------------------------------------------------
# 5. Regularization Dynamics: L1 (Lasso) vs. L2 (Ridge)
# -----------------------------------------------------------------------------
def analyze_regularization_impact(
    df: pd.DataFrame,
    reg_type: str = "L2 (Ridge)",
    penalty_values: Optional[List[float]] = None
) -> pd.DataFrame:
    """
    Simulates the impact of L1 (sparsity/feature selection) and
    L2 (weight shrinkage) regularization on XGBoost / LightGBM.
    """
    if penalty_values is None:
        penalty_values = [0.0, 0.05, 0.5, 2.0, 10.0, 30.0, 75.0]

    mods = get_ensemble_modules()
    X = df[TELEMETRY_FEATURES].values
    y = df["lap_time"].values
    
    X_train, X_test, y_train, y_test = mods["train_test_split"](X, y, test_size=0.25, random_state=42)
    
    rows = []
    for alpha in penalty_values:
        if mods["xgb_available"]:
            if reg_type == "L1 (Lasso / reg_alpha)":
                model = mods["XGBRegressor"](
                    n_estimators=45, max_depth=6, learning_rate=0.1,
                    reg_alpha=alpha, reg_lambda=0.0, random_state=42, n_jobs=1
                )
            else:
                model = mods["XGBRegressor"](
                    n_estimators=45, max_depth=6, learning_rate=0.1,
                    reg_alpha=0.0, reg_lambda=alpha, random_state=42, n_jobs=1
                )
        elif mods["lgb_available"]:
            if reg_type == "L1 (Lasso / reg_alpha)":
                model = mods["LGBMRegressor"](
                    n_estimators=45, max_depth=6, learning_rate=0.1,
                    reg_alpha=alpha, reg_lambda=0.0, random_state=42, verbose=-1, n_jobs=1
                )
            else:
                model = mods["LGBMRegressor"](
                    n_estimators=45, max_depth=6, learning_rate=0.1,
                    reg_alpha=0.0, reg_lambda=alpha, random_state=42, verbose=-1, n_jobs=1
                )
        else:
            model = mods["HistGradientBoostingRegressor"](
                max_iter=45, max_depth=6, learning_rate=0.1,
                l2_regularization=alpha, random_state=42
            )

        model.fit(X_train, y_train)
        
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)
        
        train_r2 = float(mods["r2_score"](y_train, train_preds))
        test_r2 = float(mods["r2_score"](y_test, test_preds))
        train_mse = float(mods["mse"](y_train, train_preds))
        test_mse = float(mods["mse"](y_test, test_preds))
        
        rows.append({
            "Penalty Strength (λ / α)": alpha,
            "Train R²": round(train_r2, 4),
            "Test R²": round(test_r2, 4),
            "Train MSE": round(train_mse, 3),
            "Test MSE": round(test_mse, 3),
            "Generalization Gap (Δ R²)": round(train_r2 - test_r2, 4)
        })
        
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# 6. Pedagogical Study Notes & Cheat Sheet
# -----------------------------------------------------------------------------
STUDENT_STUDY_GUIDE = {
    "bagging_vs_boosting": r"""
### 🏁 Bagging vs. Boosting: The Core Comparison

| Dimension | 🎒 Bagging (e.g., Random Forest) | 🚀 Boosting (e.g., XGBoost, LightGBM) |
| :--- | :--- | :--- |
| **Primary Goal** | **Reduce Variance** (combats overfitting) | **Reduce Bias** (combats underfitting) |
| **Model Formation** | **Independent parallel** base learners | **Sequential** base learners (each fixes prior errors) |
| **Data Sampling** | Bootstrap sampling (sampling with replacement) | Weighted samples / gradient residuals ($y - \hat{y}$) |
| **Feature Randomness**| Random subset of features considered at each split | Tree splitting guided by exact greedy or histogram binning |
| **Aggregation** | Simple average (regression) or majority vote (classification)| Weighted sum with learning rate shrinkage ($\eta$) |
| **Speed / Scaling** | Easily parallelizable across CPU cores | Histograms enable ultra-fast leaf-wise binning (LightGBM/XGB) |
    """,
    "bias_variance": r"""
### ⚖️ The Bias-Variance Trade-Off Explained

The total expected prediction error decomposes mathematically as:
$$\text{Expected Error}(x) = \underbrace{\text{Bias}[\hat{f}(x)]^2}_{\text{Underfitting}} + \underbrace{\text{Var}[\hat{f}(x)]}_{\text{Overfitting}} + \underbrace{\sigma^2}_{\text{Irreducible Noise}}$$

* **Bias**: Error introduced by approximating a complicated real-world physics problem by a simpler model. High bias $\rightarrow$ model misses key telemetry relations.
* **Variance**: Sensitivity of the model to small fluctuations in training laps. High variance $\rightarrow$ model fits noise/curb bumps rather than true track dynamics.
* **Irreducible Error ($\sigma^2$)**: Inherent stochastic noise in sensor telemetry (wind gusts, sensor thermal drift).
    """,
    "overfitting_remedies": r"""
### 🛡️ Overfitting & Underfitting: Causes, Identification, & Cures

#### 1. Underfitting (High Bias)
* **Symptom**: High training error **AND** high testing error. Low $R^2$ across both sets.
* **Root Cause**: Model capacity too low (e.g., single tree with depth = 1).
* **Remedies**:
  - Increase model complexity (higher `max_depth`, add polynomial/interaction telemetry features).
  - Transition from a single tree to an ensemble booster (XGBoost/LightGBM).
  - Decrease regularization penalties.

#### 2. Overfitting (High Variance)
* **Symptom**: Near-zero training error ($R^2 \approx 1.0$), but test error degrades sharply (large generalization gap).
* **Root Cause**: Model memorized transient noise or outliers in the training set.
* **Remedies**:
  - **Pruning**: Constrain `max_depth`, increase `min_samples_split` or `min_samples_leaf`.
  - **Ensemble Averaging**: Use Random Forest (Bagging) to average out variance.
  - **Regularization**: Add L1/L2 penalties (`reg_alpha`, `reg_lambda`) to penalize large leaf weights.
  - **Early Stopping**: Stop boosting iterations when validation loss ceases to improve.
    """,
    "regularization_l1_l2": r"""
### 🎯 Regularization: L1 (Lasso) vs. L2 (Ridge)

In Gradient Boosting (XGBoost & LightGBM), the objective function includes a tree complexity regularization term:
$$\mathcal{L}^{(t)} = \sum_{i=1}^n l(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)) + \underbrace{\gamma T + \alpha \sum_{j=1}^T |w_j|}_{\text{L1 Lasso Penalty}} + \underbrace{\frac{1}{2} \lambda \sum_{j=1}^T w_j^2}_{\text{L2 Ridge Penalty}}$$

* **L1 Regularization (`reg_alpha` / Lasso)**:
  - Adds penalty proportional to the absolute values of leaf weights $|w_j|$.
  - **Feature Sparsity**: Drives uninformative sensor coefficients to zero, performing automatic telemetry feature selection.
* **L2 Regularization (`reg_lambda` / Ridge)**:
  - Adds penalty proportional to the squared values of leaf weights $w_j^2$.
  - **Weight Shrinkage**: Smooths out extreme prediction spikes, stabilizing leaf outputs against telemetry noise.
    """
}
