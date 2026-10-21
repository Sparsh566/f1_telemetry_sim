"""
ml/model_persistence.py
=======================
Model Serialization, Deserialization, Validation, and Serving Engine.

Provides unified deployment functions using Scikit-Learn's `joblib` and
Python's standard library `pickle`, verifying numerical equivalence
of predictions before and after loading, recording comprehensive metadata,
and highlighting security implications.
"""

from typing import Dict, Any, Tuple, Optional
import os
import sys
import time
import io
import pickle
import numpy as np
import pandas as pd
import sklearn
import joblib

from ml.dataset import ALL_FEATURES, TARGET_COLUMN


SECURITY_WARNING = (
    "SECURITY NOTICE: Python 'pickle' and 'joblib' files can execute arbitrary "
    "bytecode during deserialization. Never load untrusted model files downloaded "
    "from untrusted third parties."
)


def extract_system_metadata(custom_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Generates standard provenance and environment metadata for model deployment."""
    meta = {
        "format_version": "1.0",
        "python_version": sys.version.split()[0],
        "sklearn_version": sklearn.__version__,
        "joblib_version": joblib.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "feature_schema": ALL_FEATURES,
        "target": TARGET_COLUMN,
        "created_at_utc": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
        "created_timestamp": time.time(),
        "security_notice": SECURITY_WARNING
    }
    if custom_metadata:
        meta.update(custom_metadata)
    return meta


def save_model_joblib(
    model: Any,
    filepath: str,
    metadata: Optional[Dict[str, Any]] = None,
    compress: int = 3
) -> str:
    """
    Saves a trained model or pipeline bundle using Joblib with memory-mapping support.
    """
    full_meta = extract_system_metadata(metadata)
    full_meta["serialization_tool"] = "joblib"

    bundle = {
        "model": model,
        "metadata": full_meta
    }

    dir_path = os.path.dirname(filepath)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    joblib.dump(bundle, filepath, compress=compress)
    return filepath


def load_model_joblib(filepath: str) -> Tuple[Any, Dict[str, Any]]:
    """
    Loads and validates a model bundle saved via Joblib.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model artifact not found at path: {filepath}")

    bundle = joblib.load(filepath)
    if not isinstance(bundle, dict) or "model" not in bundle:
        # Legacy fallback if just model was dumped
        return bundle, {}

    model = bundle["model"]
    metadata = bundle.get("metadata", {})
    return model, metadata


def save_model_pickle(
    model: Any,
    filepath: str,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Saves a trained model bundle using Python standard library's pickle.
    """
    full_meta = extract_system_metadata(metadata)
    full_meta["serialization_tool"] = "pickle"

    bundle = {
        "model": model,
        "metadata": full_meta
    }

    dir_path = os.path.dirname(filepath)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    with open(filepath, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

    return filepath


def load_model_pickle(filepath: str) -> Tuple[Any, Dict[str, Any]]:
    """
    Loads a model bundle saved via Python's pickle.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model artifact not found at path: {filepath}")

    with open(filepath, "rb") as f:
        bundle = pickle.load(f)

    if not isinstance(bundle, dict) or "model" not in bundle:
        return bundle, {}

    model = bundle["model"]
    metadata = bundle.get("metadata", {})
    return model, metadata


def verify_prediction_equivalence(
    model_a: Any,
    model_b: Any,
    X_sample: pd.DataFrame,
    atol: float = 1e-5
) -> Tuple[bool, float]:
    """
    Verifies that predictions from two models (e.g. before saving vs after loading)
    are numerically equivalent within an absolute tolerance.
    """
    preds_a = np.asarray(model_a.predict(X_sample)).ravel()
    preds_b = np.asarray(model_b.predict(X_sample)).ravel()

    max_abs_diff = float(np.max(np.abs(preds_a - preds_b)))
    is_equivalent = bool(max_abs_diff <= atol)

    return is_equivalent, max_abs_diff


def benchmark_serialization_formats(
    model: Any,
    metadata: Optional[Dict[str, Any]] = None,
    X_sample: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Direct empirical head-to-head comparison between pickle and joblib:
    dump latency, load latency, file size, and prediction verification.
    """
    meta = extract_system_metadata(metadata)

    # 1. Benchmark Pickle
    t0 = time.perf_counter()
    pkl_buf = io.BytesIO()
    pickle.dump({"model": model, "metadata": meta}, pkl_buf, protocol=pickle.HIGHEST_PROTOCOL)
    t_pkl_dump = (time.perf_counter() - t0) * 1000.0
    pkl_size = pkl_buf.tell()

    pkl_buf.seek(0)
    t0 = time.perf_counter()
    loaded_pkl = pickle.load(pkl_buf)
    t_pkl_load = (time.perf_counter() - t0) * 1000.0

    # 2. Benchmark Joblib
    t0 = time.perf_counter()
    joblib_buf = io.BytesIO()
    joblib.dump({"model": model, "metadata": meta}, joblib_buf, compress=3)
    t_joblib_dump = (time.perf_counter() - t0) * 1000.0
    joblib_size = joblib_buf.tell()

    joblib_buf.seek(0)
    t0 = time.perf_counter()
    loaded_joblib = joblib.load(joblib_buf)
    t_joblib_load = (time.perf_counter() - t0) * 1000.0

    # Prediction verification if sample provided
    pkl_verified = "N/A"
    joblib_verified = "N/A"
    if X_sample is not None and not X_sample.empty:
        ok_pkl, diff_pkl = verify_prediction_equivalence(model, loaded_pkl["model"], X_sample)
        ok_jb, diff_jb = verify_prediction_equivalence(model, loaded_joblib["model"], X_sample)
        pkl_verified = f"✅ Match (Δ={diff_pkl:.2e})" if ok_pkl else f"❌ Mismatch (Δ={diff_pkl:.2e})"
        joblib_verified = f"✅ Match (Δ={diff_jb:.2e})" if ok_jb else f"❌ Mismatch (Δ={diff_jb:.2e})"

    df_bench = pd.DataFrame([
        {
            "Format": "Joblib (`.joblib`)",
            "File Size (Bytes)": joblib_size,
            "Dump Time (ms)": round(t_joblib_dump, 2),
            "Load Time (ms)": round(t_joblib_load, 2),
            "Memory Mapping": "✅ Supported (zero-copy numpy)",
            "Prediction Match": joblib_verified,
            "Best For": "Heavy numeric arrays & Scikit-Learn pipelines"
        },
        {
            "Format": "Pickle (`.pkl`)",
            "File Size (Bytes)": pkl_size,
            "Dump Time (ms)": round(t_pkl_dump, 2),
            "Load Time (ms)": round(t_pkl_load, 2),
            "Memory Mapping": "❌ No (Full in-memory copy)",
            "Prediction Match": pkl_verified,
            "Best For": "Standard Python primitives & simple objects"
        }
    ])

    return df_bench
