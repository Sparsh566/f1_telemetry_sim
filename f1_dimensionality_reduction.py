"""
f1_dimensionality_reduction.py
==============================
High-Performance Dimensionality Reduction Engine for Formula 1 Telemetry.
Implements:
  1. Principal Component Analysis (PCA) - Linear variance maximization,
     eigenvalue decomposition, scree analysis, and feature loadings.
  2. t-Distributed Stochastic Neighbor Embedding (t-SNE) - Non-linear
     probabilistic manifold learning, resolving the crowding problem
     with Student-t distribution and tunable perplexity.
"""

from typing import Dict, Any, List, Tuple, Optional

def get_dim_red_modules():
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    return np, pd, StandardScaler, PCA, TSNE


# -----------------------------------------------------------------------------
# 1. Feature Extraction & Standardization
# -----------------------------------------------------------------------------
FEATURE_COLUMNS = [
    "speed",
    "throttle",
    "brake",
    "lap_time",
    "tyre_wear",
    "fuel",
    "brake_temp",
    "aggression"
]

FEATURE_DISPLAY_NAMES = {
    "speed": "⚡ Speed (km/h)",
    "throttle": "🟢 Throttle (%)",
    "brake": "🛑 Brake (%)",
    "lap_time": "⏱️ Lap Pace (s)",
    "tyre_wear": "📉 Tyre Wear (%)",
    "fuel": "⛽ Fuel Level (%)",
    "brake_temp": "🌡️ Brake Temp (°C)",
    "aggression": "🔥 Driver Aggression"
}


def extract_telemetry_samples(
    state: Dict[str, Dict[str, Any]],
    include_history: bool = True
) -> Any:
    """
    Extracts multi-channel telemetry feature rows from the simulator state.
    Includes both the instantaneous state and time-series history samples
    to build a robust sample set for dimensionality reduction.
    """
    _, pd, *_ = get_dim_red_modules()
    rows: List[Dict[str, Any]] = []

    for driver, car in state.items():
        base_aggression = float(car.get("aggression", 1.0))
        compound = car.get("tyre", "Medium")

        # Extract historical telemetry points
        history = list(car.get("history", []))
        if include_history and history:
            for idx, h in enumerate(history):
                # Fuel burns down over time; estimate historical fuel
                fuel_est = max(5.0, float(car.get("fuel", 50.0)) + (len(history) - idx) * 0.08)
                # Tyre wear increases over time; estimate historical wear
                wear_est = max(0.01, float(car.get("tyre_wear", 0.1)) - (len(history) - idx) * 0.001)

                speed_val = float(h.get("speed", car.get("speed", 250.0)))
                # Throttle and brake derived from speed dynamics
                throttle_val = min(100.0, max(0.0, (speed_val / 330.0) * 100.0))
                brake_val = max(0.0, 100.0 - throttle_val) if speed_val < 200.0 else 0.0

                rows.append({
                    "driver": driver,
                    "tyre_compound": compound,
                    "speed": speed_val,
                    "throttle": throttle_val,
                    "brake": brake_val,
                    "lap_time": float(h.get("lap_time", car.get("lap_time", 85.0))),
                    "tyre_wear": wear_est * 100.0,
                    "fuel": min(100.0, fuel_est),
                    "brake_temp": float(h.get("brake_temp", car.get("brake_temp", 450.0))),
                    "aggression": base_aggression,
                    "sample_type": "Historical Snapshot"
                })

        # Instantaneous current tick point
        curr_speed = float(car.get("speed", 250.0))
        curr_throttle = min(100.0, max(0.0, (curr_speed / 330.0) * 100.0))
        curr_brake = max(0.0, 100.0 - curr_throttle) if curr_speed < 200.0 else 0.0

        rows.append({
            "driver": driver,
            "tyre_compound": compound,
            "speed": curr_speed,
            "throttle": curr_throttle,
            "brake": curr_brake,
            "lap_time": float(car.get("lap_time", 85.0)),
            "tyre_wear": float(car.get("tyre_wear", 0.1)) * 100.0,
            "fuel": float(car.get("fuel", 50.0)),
            "brake_temp": float(car.get("brake_temp", 450.0)),
            "aggression": base_aggression,
            "sample_type": "Live Current State"
        })

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# 2. Principal Component Analysis (PCA) Engine
# -----------------------------------------------------------------------------
class PCAResult:
    def __init__(
        self,
        transformed_df: pd.DataFrame,
        explained_variance_ratio: np.ndarray,
        cumulative_variance: np.ndarray,
        eigenvalues: np.ndarray,
        loadings_df: pd.DataFrame,
        feature_names: List[str],
        n_components: int
    ):
        self.transformed_df = transformed_df
        self.explained_variance_ratio = explained_variance_ratio
        self.cumulative_variance = cumulative_variance
        self.eigenvalues = eigenvalues
        self.loadings_df = loadings_df
        self.feature_names = feature_names
        self.n_components = n_components

    def get_variance_summary_text(self) -> str:
        var_pc1 = self.explained_variance_ratio[0] * 100.0
        var_pc2 = self.explained_variance_ratio[1] * 100.0 if self.n_components > 1 else 0.0
        total_2d = var_pc1 + var_pc2
        return (
            f"PC1 captures {var_pc1:.1f}% of variance, PC2 captures {var_pc2:.1f}% of variance. "
            f"Together, just 2 dimensions preserve {total_2d:.1f}% of the multi-sensor race dynamics."
        )

    def interpret_component(self, component_idx: int = 0) -> str:
        """Provides an engineering interpretation of the component based on top loadings."""
        col_name = f"PC{component_idx + 1}"
        loadings = self.loadings_df[col_name].abs().sort_values(ascending=False)
        top_features = loadings.head(2).index.tolist()
        
        # Check signs of top features
        top_desc = []
        for feat in top_features:
            weight = self.loadings_df.loc[feat, col_name]
            direction = "High" if weight > 0 else "Low"
            top_desc.append(f"{direction} {FEATURE_DISPLAY_NAMES.get(feat, feat)}")
            
        return f"{col_name} is primarily driven by: {' & '.join(top_desc)}"


def compute_pca(
    df: Any,
    features: List[str] = FEATURE_COLUMNS,
    n_components: int = 4
) -> Optional[PCAResult]:
    """
    Performs standard-scaler normalization and PCA decomposition on telemetry.
    Computes coordinates, explained variance, cumulative variance, and loadings.
    """
    if df is None or len(df) < 3:
        return None

    try:
        np, pd, StandardScaler, PCA, _ = get_dim_red_modules()
        X = df[features].copy()
        
        # Standardize features: zero-mean and unit variance
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Adjust components if fewer samples than features
        max_components = min(len(features), len(df), n_components)
        pca = PCA(n_components=max_components, random_state=42)
        Z = pca.fit_transform(X_scaled)

        # Transformed coordinates dataframe
        transformed_df = df.copy()
        for i in range(max_components):
            transformed_df[f"PC{i+1}"] = Z[:, i]

        # Explained variance calculations
        exp_var_ratio = pca.explained_variance_ratio_
        cum_var = np.cumsum(exp_var_ratio)
        eigenvalues = pca.explained_variance_

        # Component loadings = eigenvectors * sqrt(eigenvalues)
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        loadings_df = pd.DataFrame(
            loadings,
            index=features,
            columns=[f"PC{i+1}" for i in range(max_components)]
        )

        return PCAResult(
            transformed_df=transformed_df,
            explained_variance_ratio=exp_var_ratio,
            cumulative_variance=cum_var,
            eigenvalues=eigenvalues,
            loadings_df=loadings_df,
            feature_names=features,
            n_components=max_components
        )
    except Exception as e:
        print(f"PCA computation warning: {e}")
        return None


# -----------------------------------------------------------------------------
# 3. t-Distributed Stochastic Neighbor Embedding (t-SNE) Engine
# -----------------------------------------------------------------------------
class TSNEResult:
    def __init__(
        self,
        embedded_df: Any,
        perplexity: float,
        n_iter: int,
        kl_divergence: float
    ):
        self.embedded_df = embedded_df
        self.perplexity = perplexity
        self.n_iter = n_iter
        self.kl_divergence = kl_divergence


def compute_tsne(
    df: Any,
    features: List[str] = FEATURE_COLUMNS,
    perplexity: float = 20.0,
    n_iter: int = 750,
    random_state: int = 42
) -> Optional[TSNEResult]:
    """
    Standardizes telemetry and fits a 2D t-SNE manifold embedding.
    Preserves local neighbor affinities using Gaussian (high-D) and
    Student-t (low-D) probability distributions.
    """
    if df is None or len(df) < 5:
        return None

    try:
        _, _, StandardScaler, _, TSNE = get_dim_red_modules()
        X = df[features].copy()
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Perplexity must be strictly less than n_samples
        n_samples = len(df)
        effective_perp = min(float(perplexity), max(1.0, float(n_samples - 1) / 3.0))
        if effective_perp >= n_samples:
            effective_perp = max(1.0, float(n_samples - 1))

        tsne_kwargs = {
            "n_components": 2,
            "perplexity": effective_perp,
            "random_state": random_state,
            "learning_rate": "auto",
        }
        if n_samples >= 10:
            tsne_kwargs["method"] = "barnes_hut"
            tsne_kwargs["init"] = "pca"
        else:
            tsne_kwargs["method"] = "exact"
            tsne_kwargs["init"] = "random"

        # Handle scikit-learn version differences: max_iter (new) vs n_iter (old)
        try:
            tsne = TSNE(**tsne_kwargs, max_iter=int(n_iter))
        except TypeError:
            try:
                tsne = TSNE(**tsne_kwargs, n_iter=int(n_iter))
            except TypeError:
                tsne = TSNE(**tsne_kwargs)

        embedding = tsne.fit_transform(X_scaled)

        embedded_df = df.copy()
        embedded_df["tSNE1"] = embedding[:, 0]
        embedded_df["tSNE2"] = embedding[:, 1]

        kl_div = float(getattr(tsne, "kl_divergence_", 0.0))

        return TSNEResult(
            embedded_df=embedded_df,
            perplexity=effective_perp,
            n_iter=n_iter,
            kl_divergence=kl_div
        )
    except Exception as e:
        print(f"t-SNE computation warning: {e}")
        return None

