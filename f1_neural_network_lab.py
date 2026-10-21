"""
f1_neural_network_lab.py
========================
Interactive Neural Network Learning Lab & Model Deployment Studio for Streamlit.

Implements the 7 core educational sections:
  1. Perceptron Playground (Weights, bias, weighted sum, step & sigmoid, linear boundary, XOR limitation)
  2. Activation Function Explorer (Curves, derivatives, vanishing gradients, dying ReLUs)
  3. Forward Propagation Visualizer (Input -> Hidden -> Output layer, weights, intermediate activations, dimensions)
  4. Loss Function Explorer (MSE vs. MAE for regression, Binary Cross-Entropy for classification)
  5. Backpropagation & Optimizer Lab (Analytical chain rule, finite-difference verification, SGD vs. Adam tournament)
  6. F1 Neural Network Training (NumPy MLP vs. Random Forest vs. Mean Baseline on leak-free telemetry)
  7. Model Deployment Lab (Pickle vs. Joblib benchmark, metadata, prediction equivalence verification, security notice)
  8. Student Deep Learning Master Guide (Theory notes, formulas, and Markdown export)
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import io
import time
import numpy as np
import pandas as pd
import streamlit as st

from ml.dataset import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
    FEATURE_LABELS,
    build_leak_free_race_dataset,
    split_by_race_group,
    validate_feature_schema
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
from ml.predict import predict_lap_time
from ml.activations import ACTIVATIONS
from ml.losses import LossFunction
from ml.optimizers import SGDOptimizer, SGDMomentumOptimizer, AdamOptimizer
from ml.neural_network_numpy import (
    SinglePerceptron,
    DenseLayer,
    NeuralNetwork,
    NumpyStandardScaler,
    check_network_gradients,
    save_numpy_network_weights,
    load_numpy_network_weights
)


def render_neural_network_lab(
    state: Dict[str, Dict[str, Any]],
    current_track_info: Dict[str, Any],
    driver_labels: Optional[Dict[str, str]] = None
) -> None:
    """Renders the comprehensive 7-section learning lab and deployment hub."""
    if driver_labels is None:
        driver_labels = {}

    st.info("💡 **In Plain English**: This is a virtual racing brain built from scratch using pure math (NumPy). It learns through trial-and-error by predicting lap times and comparing them to real telemetry. Explore perceptrons, activation curves, forward/backward propagation, loss functions, SGD vs. Adam, and model deployment (Pickle vs. Joblib)!")
    st.caption("🧠 *Deep Learning from Scratch (Pure NumPy): Perceptrons, Activation Functions & Derivatives, Forward & Backpropagation Chain Rule, Loss Functions (MSE/MAE/BCE), Optimizers (SGD, Adam), and Model Deployment (Pickle vs. Joblib).*")

    # 8-Tab Navigation for the Lab
    (
        tab_sec_perc,
        tab_sec_act,
        tab_sec_fwd,
        tab_sec_loss,
        tab_sec_backprop,
        tab_sec_train,
        tab_sec_deploy,
        tab_sec_guide
    ) = st.tabs([
        "🎯 1. Perceptron Playground",
        "🧬 2. Activation Functions",
        "🔢 3. Forward Propagation",
        "📉 4. Loss Functions",
        "🔄 5. Backprop & Optimizers",
        "🏎️ 6. F1 Neural Net Training",
        "📦 7. Model Deployment Lab",
        "📚 8. Theory Guide"
    ])

    # =========================================================================
    # SECTION 1: PERCEPTRON PLAYGROUND
    # =========================================================================
    with tab_sec_perc:
        st.subheader("🎯 Section 1: Single Perceptron & The Linearity Limit")
        st.markdown("Frank Rosenblatt's 1958 Perceptron computes a linear combination of inputs: $z = \\mathbf{w}^T \\mathbf{x} + b$.")

        p_col1, p_col2 = st.columns([1.2, 1.0])
        with p_col1:
            st.markdown("#### ⚙️ Interactive Perceptron Parameters")
            w1_val = st.slider("Weight $w_1$ (Tyre Wear weight):", min_value=-3.0, max_value=3.0, value=1.2, step=0.1, key="perc_w1")
            w2_val = st.slider("Weight $w_2$ (Pace Drop weight):", min_value=-3.0, max_value=3.0, value=1.8, step=0.1, key="perc_w2")
            b_val = st.slider("Bias $b$ (Threshold bias):", min_value=-5.0, max_value=5.0, value=-1.5, step=0.1, key="perc_b")

            st.markdown("#### 🧪 Test Single Telemetry Point")
            x1_val = st.slider("Input $x_1$ (Normalized Tyre Wear):", min_value=-2.0, max_value=2.0, value=0.5, step=0.1, key="perc_x1")
            x2_val = st.slider("Input $x_2$ (Normalized Pace Drop):", min_value=-2.0, max_value=2.0, value=0.8, step=0.1, key="perc_x2")

            z_calc = (w1_val * x1_val) + (w2_val * x2_val) + b_val
            step_out = 1 if z_calc >= 0.0 else 0
            sig_out = 1.0 / (1.0 + np.exp(-np.clip(z_calc, -30.0, 30.0)))

            m1, m2, m3 = st.columns(3)
            m1.metric("Weighted Sum $z$", f"{z_calc:.3f}")
            m2.metric("Step Output", f"{step_out} ({'🛑 PIT' if step_out == 1 else '🟢 STAY'})")
            m3.metric("Sigmoid Prob $\\sigma(z)$", f"{sig_out:.1%}")

        with p_col2:
            st.markdown("#### 🛑 2D Decision Boundary (Pit Window)")
            st.caption("Boundary equation where $w_1 x_1 + w_2 x_2 + b = 0 \\implies x_2 = -\\frac{w_1 x_1 + b}{w_2}$")

            w2_safe = w2_val if abs(w2_val) > 1e-4 else 1e-4
            x1_line = np.linspace(-2.0, 2.0, 50)
            x2_line = -(w1_val * x1_line + b_val) / w2_safe

            df_boundary = pd.DataFrame({
                "x1 (Tyre Wear)": x1_line,
                "Decision Boundary Line (x2)": x2_line
            }).set_index("x1 (Tyre Wear)")
            st.line_chart(df_boundary)

            st.write("---")
            st.markdown("#### 🧩 The XOR Limitation Demonstration")
            if st.button("Run XOR Limitation Check", key="btn_run_xor"):
                xor_res = SinglePerceptron.demonstrate_xor_limitation()
                st.warning(f"**Single Perceptron XOR Accuracy**: {xor_res['accuracy'] * 100:.0f}%")
                st.info(xor_res["explanation"])
                df_xor = pd.DataFrame({
                    "Input x1": [x[0] for x in xor_res["inputs"]],
                    "Input x2": [x[1] for x in xor_res["inputs"]],
                    "Target y": xor_res["targets"],
                    "Perceptron Pred": xor_res["predictions"]
                })
                st.dataframe(df_xor, use_container_width=True, hide_index=True)

    # =========================================================================
    # SECTION 2: ACTIVATION FUNCTION EXPLORER
    # =========================================================================
    with tab_sec_act:
        st.subheader("🧬 Section 2: Activation Functions & Analytical Derivatives")
        st.markdown("Activation functions introduce non-linearity so networks can model curved physics dynamics.")

        act_choice = st.selectbox("Select Activation Function to Explore:", list(ACTIVATIONS.keys()), key="act_choice_sel")
        act_obj = ACTIVATIONS[act_choice]

        a_c1, a_c2 = st.columns([1.2, 1.0])
        with a_c1:
            z_vals = np.linspace(-5.0, 5.0, 200)
            f_vals = act_obj(z_vals)
            df_vals = act_obj.derivative(z_vals)

            df_act_chart = pd.DataFrame({
                "z (Input Score)": z_vals,
                f"f(z) [{act_choice}]": f_vals,
                f"f'(z) Analytical Derivative": df_vals
            }).set_index("z (Input Score)")

            st.line_chart(df_act_chart)

        with a_c2:
            st.markdown(f"**Mathematical Formula**: ${act_obj.formula}$")
            st.markdown(f"**Analytical Derivative**: ${act_obj.derivative_formula}$")
            st.markdown(f"**Output Range**: `[{act_obj.output_range[0]}, {act_obj.output_range[1]}]`")
            st.info(f"💡 **Description & Racing Telemetry Use Case**:\n\n{act_obj.description}")
            st.warning(f"⚠️ **Known Limitations**:\n\n{act_obj.limitation}")

    # =========================================================================
    # SECTION 3: FORWARD PROPAGATION VISUALIZER
    # =========================================================================
    with tab_sec_fwd:
        st.subheader("🔢 Section 3: Forward Propagation Step-by-Step")
        st.markdown("Trace how input telemetry flows through weights, biases, and non-linear squashing to produce a continuous lap time prediction.")

        fwd_c1, fwd_c2 = st.columns([1.1, 1.1])
        with fwd_c1:
            st.markdown("#### 1. Input Features ($X$)")
            in_wear = st.slider("Simulated Tyre Wear (%):", 0.0, 100.0, 45.0, 5.0, key="fwd_in_wear")
            in_fuel = st.slider("Simulated Fuel Level (%):", 0.0, 100.0, 60.0, 5.0, key="fwd_in_fuel")

            # Normalized input vector (1 x 2)
            X_small = np.array([[(in_wear - 50.0) / 25.0, (in_fuel - 50.0) / 25.0]])

            st.markdown("#### 2. Network Weights & Biases (Pre-configured)")
            W1_small = np.array([[0.8, -0.6], [0.4, 0.9]])
            b1_small = np.array([[0.1, -0.2]])
            W2_small = np.array([[1.5], [-1.2]])
            b2_small = np.array([[85.0]])

            st.write("**Hidden Layer Weights $W^{[1]}$ (2×2):**")
            st.write(W1_small)
            st.write("**Hidden Layer Bias $b^{[1]}$ (1×2):**", b1_small)

        with fwd_c2:
            st.markdown("#### 3. Mathematical Intermediate Activations")
            # Step 1: Z1 = X * W1 + b1
            Z1_small = np.dot(X_small, W1_small) + b1_small
            st.write("**Step A: Pre-activation $Z^{[1]} = X W^{[1]} + b^{[1]}$**:")
            st.write(np.round(Z1_small, 4))

            # Step 2: A1 = ReLU(Z1)
            A1_small = np.maximum(0.0, Z1_small)
            st.write("**Step B: Activated $A^{[1]} = \\text{ReLU}(Z^{[1]})$**:")
            st.write(np.round(A1_small, 4))

            # Step 3: Z2 = A1 * W2 + b2
            Z2_small = np.dot(A1_small, W2_small) + b2_small
            y_pred_small = float(Z2_small[0, 0])

            st.write("**Step C: Output Lap Time $\\hat{y} = A^{[1]} W^{[2]} + b^{[2]}$**:")
            st.metric("🏁 Predicted Lap Pace", f"{y_pred_small:.3f} s")

            st.caption("""
            **Dimensional Consistency Check**:
            * $X$: $(1 \\times 2)$
            * $W^{[1]}$: $(2 \\times 2) \\implies Z^{[1]}$: $(1 \\times 2)$
            * $W^{[2]}$: $(2 \\times 1) \\implies \\hat{y}$: $(1 \\times 1)$
            """)

    # =========================================================================
    # SECTION 4: LOSS FUNCTION EXPLORER
    # =========================================================================
    with tab_sec_loss:
        st.subheader("📉 Section 4: Loss Functions (MSE vs. MAE vs. Binary Cross-Entropy)")
        st.markdown("Loss functions measure prediction penalty, providing the error signal backpropagated into the weights.")

        loss_c1, loss_c2 = st.columns(2)
        with loss_c1:
            st.markdown("#### ⏱️ Regression Losses: Lap Pace Prediction")
            true_lap_slider = st.slider("Actual Ground Truth Lap Time ($y$):", 75.0, 95.0, 85.0, 0.5, key="loss_true_lap")
            pred_lap_slider = st.slider("Model Predicted Lap Time ($\\hat{y}$):", 75.0, 95.0, 86.5, 0.5, key="loss_pred_lap")

            err_diff = pred_lap_slider - true_lap_slider
            mse_val = LossFunction.mse(np.array([pred_lap_slider]), np.array([true_lap_slider]))
            mae_val = LossFunction.mae(np.array([pred_lap_slider]), np.array([true_lap_slider]))

            lm1, lm2, lm3 = st.columns(3)
            lm1.metric("Delta $(\\hat{y} - y)$", f"{err_diff:+.2f} s")
            lm2.metric("MSE Loss $(\\Delta^2)$", f"{mse_val:.3f} s²")
            lm3.metric("MAE Loss $(|\\Delta|)$", f"{mae_val:.3f} s")

            # Loss landscape curve
            delta_range = np.linspace(-5.0, 5.0, 100)
            df_loss_curve = pd.DataFrame({
                "Prediction Error Delta (s)": delta_range,
                "Mean Squared Error (MSE)": delta_range ** 2,
                "Mean Absolute Error (MAE)": np.abs(delta_range)
            }).set_index("Prediction Error Delta (s)")
            st.line_chart(df_loss_curve)

        with loss_c2:
            st.markdown("#### 🛑 Classification Loss: Binary Cross-Entropy (Pit Call)")
            st.caption("Measures penalty when predicting probabilities ($p \\in [0, 1]$) for a binary event ($y \\in \\{0, 1\\}$).")

            bce_true = st.radio("True Strategy Outcome ($y$):", [1, 0], format_func=lambda x: "🛑 1 (Box/Pit Stop)" if x == 1 else "🟢 0 (Stay Out)", horizontal=True, key="bce_true_y")
            bce_prob = st.slider("Model Confidence Probability ($p$):", 0.01, 0.99, 0.85, 0.01, key="bce_prob_p")

            bce_loss_val = LossFunction.binary_cross_entropy(np.array([bce_prob]), np.array([bce_true]))
            st.metric("Binary Cross-Entropy Loss", f"{bce_loss_val:.4f}")

            # Plot BCE curves
            p_axis = np.linspace(0.01, 0.99, 100)
            df_bce_curve = pd.DataFrame({
                "Predicted Probability (p)": p_axis,
                "BCE Loss if True=1 (Pit)": -np.log(p_axis),
                "BCE Loss if True=0 (Stay)": -np.log(1.0 - p_axis)
            }).set_index("Predicted Probability (p)")
            st.line_chart(df_bce_curve)

    # =========================================================================
    # SECTION 5: BACKPROPAGATION & OPTIMIZER LAB
    # =========================================================================
    with tab_sec_backprop:
        st.subheader("🔄 Section 5: Analytical Backpropagation & Optimizer Arena")
        st.markdown("Backpropagation computes analytical gradients via the calculus chain rule: $\\frac{\\partial L}{\\partial W^{[l]}} = \\frac{1}{N} (A^{[l-1]})^T \\delta^{[l]}$.")

        bp_c1, bp_c2 = st.columns([1.2, 1.0])
        with bp_c1:
            st.markdown("#### 🔬 Numerical Finite-Difference Gradient Check")
            st.caption("Verifies that the analytical chain-rule backpropagation code matches $(L(W+\\epsilon) - L(W-\\epsilon)) / (2\\epsilon)$ within $10^{-4}$ tolerance.")

            if st.button("🧪 Run Finite-Difference Gradient Verification", key="btn_run_grad_check"):
                test_net = NeuralNetwork(layer_sizes=[3, 4, 1], activations=["Tanh", "Linear"], seed=101)
                X_check = np.random.randn(8, 3)
                y_check = np.random.randn(8, 1)

                ok_grad, max_rel_err = check_network_gradients(test_net, X_check, y_check, eps=1e-5)
                if ok_grad:
                    st.success(f"✅ **GRADIENT CHECK PASSED**: Max relative discrepancy = `{max_rel_err:.2e}` (< 1e-4). Analytical backprop is mathematically verified!")
                else:
                    st.error(f"❌ Discrepancy observed: `{max_rel_err:.2e}`")

            st.write("---")
            st.markdown("#### 🏁 3-Way Optimizer Convergence Tournament (SGD vs. Momentum vs. Adam)")
            st.caption("Trains 3 identical neural networks starting with identical initial weights on the same telemetry data.")

            opt_lr_sel = st.slider("Learning Rate (η):", 0.001, 0.05, 0.01, 0.002, key="bp_opt_lr")
            opt_epochs_sel = st.slider("Training Epochs:", 20, 100, 50, 10, key="bp_opt_epochs")

            if st.button("⚡ Run Optimizer Tournament", key="btn_run_opt_tourney"):
                with st.spinner("Training SGD, Momentum, and Adam models..."):
                    df_synth = build_leak_free_race_dataset(num_races=15, random_state=42)
                    num_cols = ["speed", "throttle", "brake", "tyre_wear", "fuel", "brake_temp", "aggression"]
                    X_tour = df_synth[num_cols].values
                    y_tour = df_synth["lap_time"].values.reshape(-1, 1)

                    scaler_t = NumpyStandardScaler().fit(X_tour)
                    X_tour_norm = scaler_t.transform(X_tour)

                    in_d = X_tour_norm.shape[1]
                    l_sizes = [in_d, 24, 12, 1]
                    l_acts = ["ReLU", "ReLU", "Linear"]

                    opts = [
                        ("Vanilla SGD", SGDOptimizer(lr=opt_lr_sel)),
                        ("SGD + Momentum", SGDMomentumOptimizer(lr=opt_lr_sel, momentum=0.9)),
                        ("Adam", AdamOptimizer(lr=opt_lr_sel))
                    ]

                    opt_hist = {}
                    for opt_label, opt_obj in opts:
                        net_t = NeuralNetwork(layer_sizes=l_sizes, activations=l_acts, seed=42)
                        h = net_t.fit(X_tour_norm, y_tour, epochs=opt_epochs_sel, batch_size=32, optimizer=opt_obj)
                        opt_hist[opt_label] = h["train_loss"]

                    df_opt_chart = pd.DataFrame(opt_hist, index=[f"Epoch {e}" for e in range(1, opt_epochs_sel + 1)])
                    st.session_state["df_opt_tournament"] = df_opt_chart

            if "df_opt_tournament" in st.session_state:
                st.line_chart(st.session_state["df_opt_tournament"])

        with bp_c2:
            st.info("""
            💡 **Why Adam Beats Vanilla SGD on Racing Telemetry**:
            * **Noisy Sensors**: Telemetry features oscillate rapidly (e.g., brake lock-ups, sudden kerb bounces).
            * **Vanilla SGD**: Updates all parameters using the exact same scalar step, causing chaotic oscillations across steep ravines.
            * **Momentum**: Simulates inertia (a heavy rolling ball), dampening sideways oscillations and accelerating downhill.
            * **Adam**: Tracks running 1st moments ($m_t$, mean) and 2nd moments ($v_t$, variance), adapting individual learning rates for every single parameter!
            """)

    # =========================================================================
    # SECTION 6: F1 NEURAL NETWORK TRAINING
    # =========================================================================
    with tab_sec_train:
        st.subheader("🏎️ Section 6: From-Scratch F1 Neural Net Training & Benchmarking")
        st.markdown("Train the pure NumPy Multi-Layer Perceptron on leak-free telemetry and evaluate against **Random Forest** and a **Mean Baseline**.")

        if "df_f1_nn_leakfree" not in st.session_state:
            st.session_state["df_f1_nn_leakfree"] = build_leak_free_race_dataset(num_races=30, random_state=42)

        df_f1_data = st.session_state["df_f1_nn_leakfree"]

        tc1, tc2, tc3 = st.columns(3)
        tc1.metric("Simulated Races", f"{df_f1_data['race_id'].nunique()} Races")
        tc2.metric("Total Samples", f"{len(df_f1_data):,} Laps")
        tc3.metric("Data Leakage Protection", "✅ Split by race_id group")

        if st.button("🚀 Train All Models & Benchmark (NumPy Net vs RF vs Baseline)", key="btn_train_bench_all"):
            with st.spinner("Training models on disjoint race groups..."):
                train_df, val_df, test_df = split_by_race_group(df_f1_data, group_col="race_id", random_state=42)

                num_feats = ["speed", "throttle", "brake", "tyre_wear", "fuel", "brake_temp", "aggression", "pit_stops", "incident_count"]
                X_tr_num = train_df[num_feats].values
                y_tr_num = train_df["lap_time"].values
                X_te_num = test_df[num_feats].values
                y_te_num = test_df["lap_time"].values

                # Standardize using training statistics only
                scaler_nn = NumpyStandardScaler().fit(X_tr_num, feature_names=num_feats)
                X_tr_scaled = scaler_nn.transform(X_tr_num)
                X_te_scaled = scaler_nn.transform(X_te_num)

                # 1. Train NumPy Neural Net
                nn_model = NeuralNetwork(layer_sizes=[len(num_feats), 24, 12, 1], activations=["ReLU", "ReLU", "Linear"], seed=42)
                nn_hist = nn_model.fit(X_tr_scaled, y_tr_num, epochs=60, batch_size=32, optimizer=AdamOptimizer(lr=0.01))
                y_pred_nn = nn_model.predict(X_te_scaled)
                m_nn = calculate_regression_metrics(y_te_num, y_pred_nn)

                # 2. Train Random Forest (Full pipeline)
                rf_pipeline, rf_metrics, (X_te_rf, y_te_rf, y_pred_rf), rf_meta = train_model_from_df(df_f1_data, random_state=42)

                # 3. Baseline Mean Lap Regressor
                baseline_mod = BaselineMeanLapRegressor().fit(X_tr_num, y_tr_num)
                y_pred_base = baseline_mod.predict(X_te_num)
                m_base = calculate_regression_metrics(y_te_num, y_pred_base)

                df_results = pd.DataFrame([
                    {
                        "Model Architecture": "Baseline (Training Set Mean)",
                        "Test R²": m_base["r2"],
                        "Test MAE (s)": m_base["mae"],
                        "Test RMSE (s)": m_base["rmse"],
                        "Core Engine": "Naive Reference"
                    },
                    {
                        "Model Architecture": "Pure NumPy Multi-Layer Perceptron (MLP)",
                        "Test R²": m_nn["r2"],
                        "Test MAE (s)": m_nn["mae"],
                        "Test RMSE (s)": m_nn["rmse"],
                        "Core Engine": "First-Principles NumPy (No PyTorch/TF)"
                    },
                    {
                        "Model Architecture": "Scikit-Learn RandomForestRegressor",
                        "Test R²": rf_metrics["r2"],
                        "Test MAE (s)": rf_metrics["mae"],
                        "Test RMSE (s)": rf_metrics["rmse"],
                        "Core Engine": "Ensemble Decision Trees (150 Trees)"
                    }
                ])

                st.session_state["df_bench_results"] = df_results
                st.session_state["active_nn_model"] = nn_model
                st.session_state["active_nn_scaler"] = scaler_nn
                st.session_state["active_rf_pipeline"] = rf_pipeline

        if "df_bench_results" in st.session_state:
            st.dataframe(st.session_state["df_bench_results"], use_container_width=True, hide_index=True)

    # =========================================================================
    # SECTION 7: MODEL DEPLOYMENT LAB (PICKLE VS. JOBLIB)
    # =========================================================================
    with tab_sec_deploy:
        st.subheader("📦 Section 7: Model Deployment, Serialization & Serving")
        st.markdown("Learn how production race engineering pipelines serialize, version, and serve model artifacts.")

        st.warning(f"🛡️ **{SECURITY_WARNING}**")

        # Benchmark Pickle vs Joblib
        st.markdown("#### ⚖️ Serialization Benchmark: Python `pickle` vs. Scikit-Learn `joblib`")
        bench_rf = st.session_state.get("active_rf_pipeline", create_f1_regression_pipeline(n_estimators=20))
        if "active_rf_pipeline" not in st.session_state:
            sample_train_df = build_leak_free_race_dataset(num_races=5, random_state=42)
            bench_rf.fit(sample_train_df[ALL_FEATURES], sample_train_df[TARGET_COLUMN].values)

        from ml.dataset import ALL_FEATURES
        # Build clean sample for benchmark
        sample_eval_df = build_leak_free_race_dataset(num_races=2, random_state=42)[ALL_FEATURES].head(8)
        df_ser_bench = benchmark_serialization_formats(bench_rf, X_sample=sample_eval_df)
        st.dataframe(df_ser_bench, use_container_width=True, hide_index=True)

        st.write("---")
        dep_c1, dep_c2 = st.columns(2)
        with dep_c1:
            st.markdown("#### 📥 Export Trained Model Bundle")
            fmt_sel = st.radio("Choose Serialization Format:", ["joblib (.joblib)", "pickle (.pkl)"], horizontal=True, key="deploy_fmt_radio")
            is_joblib = "joblib" in fmt_sel

            model_meta = {
                "model_type": "F1_LapTime_Regressor",
                "features": ALL_FEATURES,
                "target": TARGET_COLUMN,
                "exported_from": "F1_Telemetry_Simulator_Studio"
            }

            buf = io.BytesIO()
            if is_joblib:
                import joblib
                joblib.dump({"model": bench_rf, "metadata": model_meta}, buf, compress=3)
                ext = "joblib"
            else:
                import pickle
                pickle.dump({"model": bench_rf, "metadata": model_meta}, buf, protocol=pickle.HIGHEST_PROTOCOL)
                ext = "pkl"

            buf.seek(0)
            st.download_button(
                label=f"💾 Download Serialized Model ({ext.upper()})",
                data=buf.getvalue(),
                file_name=f"f1_lap_predictor_{current_track_info['name'].lower()}.{ext}",
                mime="application/octet-stream",
                help=f"Download production deployment artifact using {ext}"
            )

        with dep_c2:
            st.markdown("#### 📤 Standalone Prediction Workflow (`ml/predict.py`)")
            st.caption("How to run predictions without Streamlit using the standalone prediction script:")
            st.code("""
# Run inference from command line:
python ml/predict.py --model models/f1_rf_pipeline.joblib

# In Python scripts:
from ml.model_persistence import load_model_joblib
from ml.predict import predict_lap_time

model, meta = load_model_joblib("models/f1_rf_pipeline.joblib")
predictions = predict_lap_time(model, live_telemetry_df)
            """, language="python")

    # =========================================================================
    # SECTION 8: THEORY & INTERVIEW STUDY GUIDE
    # =========================================================================
    with tab_sec_guide:
        st.subheader("📚 F1 Deep Learning & Model Deployment Student Master Guide")
        st.markdown("Curated mathematical formulas, chain rule derivations, and interview-ready concepts.")

        import f1_neural_network
        with st.expander("⚡ 1. The Perceptron & The Linearity Limit", expanded=True):
            st.markdown(f1_neural_network.NN_STUDY_GUIDE["perceptron_basics"])

        with st.expander("🧬 2. Activation Functions: Non-Linearity & Vanishing Gradients", expanded=False):
            st.markdown(f1_neural_network.NN_STUDY_GUIDE["activations_derivatives"])

        with st.expander("🔄 3. Forward & Backward Propagation (Analytical Chain Rule)", expanded=False):
            st.markdown(f1_neural_network.NN_STUDY_GUIDE["backpropagation_chain_rule"])

        with st.expander("🚀 4. Loss Functions & Optimizers: SGD vs. Adam", expanded=False):
            st.markdown(f1_neural_network.NN_STUDY_GUIDE["optimizers_sgd_adam"])

        with st.expander("📦 5. Model Deployment Fundamentals: Pickle vs. Joblib", expanded=False):
            st.markdown(f1_neural_network.NN_STUDY_GUIDE["deployment_pickle_joblib"])
