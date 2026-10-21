# 🏎️ Formula 1 Telemetry Simulator & Predictive Analytics

An advanced, real-time **Formula 1 telemetry simulator and predictive machine learning dashboard** built with **Python**, **Streamlit**, and **Scikit-Learn**. 

The system simulates 8 Formula 1 cars across 7 real-world Grand Prix circuits, modeling vehicle dynamics (velocity, braking, tyre degradation, fuel burn, carbon-ceramic brake disc temperatures, DRS aerodynamic boost, and pit stop strategies) with live visual telemetry, time acceleration, and an integrated Machine Learning prediction engine.

---

## 📌 Project Overview: What It Is & How It Works

### 🎯 Objective
To create an authentic, interactive Formula 1 race engineering environment that combines **stochastic vehicle physics**, **real-world circuit profiles**, **real-time dashboard analytics**, and **predictive machine learning**.

### ⚙️ How the Simulator Works
1. **Circuit Physics Modeling**: Tracks 8 real F1 drivers on the **official 2025 World Championship grid** (`VER`, `HAM`, `LEC`, `NOR`, `PIA`, `RUS`, `ALO`, `SAI`) around selected world-championship tracks with authentic circuit lengths (e.g., 5.793 km Monza, 7.004 km Spa, 3.337 km Monaco).
2. **Dynamic Physics Loop**:
   - **Speed & Throttle**: Computed each tick using driver aggression, throttle percentage, and aerodynamic downforce levels.
   - **Tyre Model**: Soft, Medium, Hard, Inter, and Wet compounds with unique grip coefficients and progressive degradation curves.
   - **Brake Thermodynamics**: Heats up to 900°C under heavy chicane braking and cools on straights.
   - **Fuel Penalty**: Burning fuel gradually lightens the vehicle, generating faster lap times toward the end of a stint.
   - **DRS (Drag Reduction System)**: Aerodynamic rear-wing opening on straights when exceeding threshold speeds.
   - **Pit Stop Mechanics**: Strategic tyre compound swaps with realistic pit lane transit times and variance.
   - **Safety Car System**: Neutralizes field speed upon simulated track incidents.
3. **Dual Telemetry Visualization**:
   - **Live Leaderboard**: Race position, official lap count (`Lap X/Total`), in-cell track progress bars (`0%–100%`), 3-sector timing (`S1`, `S2`, `S3`), tyre degradation, and fuel tank levels.
   - **Graphical & Tabular Views**: Real-time line graphs (speed, lap pace, brake temps) and structured data tables with full metric explanations and tooltips.
4. **Predictive Machine Learning Engine**:
   - Generates synthetic race datasets across varying weather, aggression, tyre, and track conditions.
   - Trains a **Random Forest Regressor** to predict expected average lap times.
   - Outputs evaluation metrics ($R^2$, MAE), feature importances, and downloadable serialized models (`.joblib`).

---

## 👥 Team Members & Contributions (4-Member Team)

| Member | Role | Key Contributions & Responsibilities |
| :--- | :--- | :--- |
| **Member 1** | **Lead System Architect & Frontend UI/UX** | • Designed and developed the Streamlit user interface and application layout.<br>• Implemented real-time dashboard controls (Start, Stop, Reset, Safety Car).<br>• Designed the interactive Grand Prix Circuit selector header and profile cards.<br>• Integrated in-cell visual progress bars and hover tooltips for all telemetry headers.<br>• Built the beginner-friendly telemetry glossary and column guide modal. |
| **Member 2** | **Vehicle Dynamics & Physics Simulation Engineer** | • Designed mathematical models for speed, throttle, and braking forces.<br>• Implemented multi-compound tyre degradation and grip decay algorithms.<br>• Developed carbon-ceramic brake disc thermodynamic heating and cooling formulas.<br>• Built the Drag Reduction System (DRS) state machine (timer, cooldown, speed check).<br>• Programmed driver aggression modeling with interactive randomization popup alerts. |
| **Member 3** | **Circuit Modeling & Telemetry Data Engineer** | • Researched and integrated 7 real-world F1 circuits (Monza, Silverstone, Spa, Monaco, etc.).<br>• Implemented 3-sector timing (`S1`, `S2`, `S3`) and distance-based lap progression.<br>• Engineered the simulation speed multiplier (`1x` to `30x`) for fluid race progression.<br>• Built dual-tab view architecture (`Graphical Charts View` vs `Telemetry Data Tables View`).<br>• Developed instantaneous CSV telemetry snapshot exporter with circuit tagging. |
| **Member 4** | **Machine Learning & Predictive Analytics Engineer** | • Designed synthetic race generation pipeline for training dataset construction.<br>• Built Scikit-Learn `ColumnTransformer` and `RandomForestRegressor` ML pipeline.<br>• Implemented cross-version compatible feature encoding (`OneHotEncoder` / `StandardScaler`).<br>• Developed feature importance analysis and model evaluation metrics ($R^2$ and MAE).<br>• Implemented `.joblib` model serialization and live session prediction inference. |

---

## 🚀 Key Features

### 🎮 1. Easy Spectator Mode (Beginner & Non-Tech Friendly)
- **🏎️ Live 2D Grand Prix Track Map**: Watch all 8 cars battle and overtake in real time on high-accuracy 2D track layouts.
- **🎙️ Live F1 TV Commentary Feed**: Generates real-time broadcast commentary in plain English (leader gaps, wheel-to-wheel battles, tyre cliff alerts, and DRS activations).
- **📞 1-Click Pitstop Radio**: Take the pitwall headset and order tyre changes for any driver with a single click.
- **📺 Broadcast TV Tower Leaderboard**: Simplified standings showing positions P1-P8, car speeds, and colour-coded tyre health (🟢 Fresh, 🟡 Used, 🔴 Critical/Box Now).
- **📖 F1 Beginner's Rules & Strategy Guide**: Plain-English explanations of the 5 tyre compounds, the tyre cliff, fuel burn weight, DRS overtaking, and pit strategy.

### 🏁 2. Grand Prix Circuit Selection
Choose from 7 real-world Formula 1 circuits with dynamic physics calibration:
- **🇮🇹 Monza (Autodromo Nazionale Monza)**: 5.793 km, 53 laps — ultra-high top speed (up to 355 km/h), heavy chicane braking.
- **🇬🇧 Silverstone (Silverstone Circuit)**: 5.891 km, 52 laps — high-speed sweeping corners, intense lateral tyre stress.
- **🇧🇪 Spa-Francorchamps (Circuit de Spa)**: 7.004 km, 44 laps — longest track, Eau Rouge & Kemmel Straight.
- **🇲🇨 Circuit de Monaco (Monte Carlo)**: 3.337 km, 78 laps — tight street walls, low speeds, elevated incident risk.
- **🇯🇵 Suzuka (Suzuka International Circuit)**: 5.807 km, 53 laps — figure-8 layout, continuous high-G S-Curves.
- **🇧🇭 Bahrain (Bahrain International Circuit)**: 5.412 km, 57 laps — abrasive desert tarmac, high tyre wear, 3 DRS zones.
- **🇧🇷 Interlagos (Autódromo José Carlos Pace)**: 4.309 km, 71 laps — anti-clockwise bowl with steep elevation changes.

### ⏱️ 2. Time Acceleration & Live Controls
- **⚡ Sim Speed Slider**: Accelerate simulation time (`1x`, `2x`, `5x`, `8x`, `15x`, `30x`). At `8x`, full race laps complete in ~10 seconds.
- **🎲 Randomized Driver Aggression**: Assigns randomized attack multipliers (`0.85x` to `1.25x`) with instant breakdown cards classifying each driver as **🔥 High**, **⚡ Balanced**, or **🛡️ Safe**.
- **Per-Car Pit Strategy**: Queue tyre changes (`Soft`, `Medium`, `Hard`, `Inter`, `Wet`) from the sidebar.
- **Manual DRS Override**: Force immediate DRS activation for testing speed boosts.

### 📊 3. Dual-View Telemetry Analysis
- **📈 Graphical Charts View**: Live trend lines for Speed (km/h), Lap Times (s), and Brake Disc Heat (°C) with clean driver labels.
- **📋 Telemetry Data Tables View**: Self-explanatory tables with clear columns:
  - **Speed & Velocity**: Current Speed, Max Recorded Speed, Average Speed, DRS Status.
  - **Lap Timing & Pace**: Position, Current Pace, Previous Lap, Personal Best, Gap to Leader.
  - **Brakes & Tyres**: Disc Temp, Thermal Window (`Optimal`, `Cold`, `Overheated`), Compound, Tyre Wear %, Fuel Remaining %, Driver Aggression.
- **🗺️ 2D Circuit GPS Map & Battle Visualizer**: High-precision 2D parametric track layouts for all 7 Grand Prix circuits with live car markers, team-colored dots, sector splits, and real-time gap tracking.

### 🤖 4. Machine Learning Module
- Generates synthetic multi-race training datasets.
- Fits a Scikit-Learn `RandomForestRegressor` with one-hot encoded tyre compounds and scaled numeric parameters.
- Evaluates model performance using $R^2$ score and Mean Absolute Error (MAE).
- Predicts expected average lap times for every car in the current live session.
- Allows exporting trained `.joblib` models and `.csv` prediction sheets.

### 🌳 5. Ensemble Studio & Model Diagnostics
- **🏆 Ensemble Arena**: Head-to-head benchmarking of Single Decision Tree, Bagging Regressor, Random Forest, XGBoost, and LightGBM.
- **📊 Feature Importance & Explainability**: Compares Gini impurity vs Gain vs Split frequencies across sensors.
- **🔮 Pitwall 'What-If' Strategy Calculator**: Live sandbox to simulate tyre compounds, wear, fuel, and aggression with real-time pitstop advice.
- **⚖️ Bias–Variance Trade-Off Lab**: Empirical decomposition of $\text{Error} = \text{Bias}^2 + \text{Variance} + \sigma^2$ across tree depths.
- **🛡️ Overfitting & Regularization Simulator**: Real-time diagnostic badge (High Bias vs Balanced vs High Variance), Generalization Gap ($\Delta R^2$), and interactive sliders for L1 Lasso (`reg_alpha`) and L2 Ridge (`reg_lambda`).
- **📚 Student Reference Guide**: Integrated mathematical formulations, causes/remedies of overfitting, and downloadable markdown study notes.

### 🧠 6. From-Scratch Neural Networks & Model Deployment Hub
- **⚡ Pure-NumPy Multi-Layer Perceptron (MLP)**: Transparent feedforward neural network engineered from first principles with He/Xavier Glorot initialization, forward caching, and analytical backpropagation matrix calculus.
- **🏁 3-Way Optimizer Convergence Race**: Side-by-side loss convergence tournament comparing **Vanilla SGD**, **SGD with Polyak Momentum (0.9)**, and **Adam (Adaptive Moment Estimation)** on identical initial weights.
- **🎯 Single Perceptron Pit Window Classifier**: Visualizes Frank Rosenblatt's 1958 linear separator for binary pit decisions based on tyre degradation and pace drop.
- **🧬 Activation Functions & Exact Derivatives**: Interactive mathematical playground plotting curves and gradients for **ReLU**, **Leaky ReLU**, **Sigmoid**, **Tanh**, and **Linear** identity.
- **📦 Model Deployment Hub (Pickle vs. Joblib)**: Empirical serialization benchmark comparing byte payload size, serialization latency, load time, and zero-copy `memmap` memory mapping. Supports exporting, uploading, and deploying serialized models to generate live pitwall inferences across all 8 active drivers.
- **📚 Student Deep Learning Master Guide**: Step-by-step mathematical explanations, chain rule derivations, optimizer formulations, and one-click `.md` notes export.

---

## 🛠️ Tech Stack

- **Language**: Python 3.10+ (tested on Python 3.14)
- **UI & Dashboard Framework**: Streamlit (v1.40+)
- **Data Manipulation**: NumPy, Pandas
- **Machine Learning & Deep Learning**: Scikit-Learn, Pure NumPy (From-Scratch MLP, Activations, SGD/Adam, Backprop), XGBoost, LightGBM
- **Model Deployment & Serialization**: Joblib, Python Pickle (`memmap` zero-copy vs object stream)
- **Data Export**: Standard CSV & Markdown Notes

---

## 💻 Installation & How to Run

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/username/F1-Telemetry-Simulator.git
cd F1-Telemetry-Simulator
```

### 2️⃣ Set Up a Virtual Environment (Optional but Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Simulator
```bash
python -m streamlit run F1_Telemetry_Simulator.py
```
*The app will automatically open in your default browser at `http://localhost:8501`.*

### 5️⃣ Run Automated Unit Tests (26 Tests)
```bash
python run_tests.py
```
*Executes unit tests covering activations, loss gradients, optimizers, finite-difference backprop checks, model persistence, and leak-free race splitting.*

### 6️⃣ Run Standalone Model Inference (CLI)
```bash
python ml/predict.py --model models/f1_rf_pipeline.joblib
```
*Loads serialized artifacts and executes predictions on telemetry features independently of Streamlit.*

---

## 🔬 Scientific & Simulation Disclaimer
This project is an **approximate educational simulation** built for learning and demonstration. Vehicle dynamics, thermodynamics, tyre wear, and telemetry distributions are mathematical approximations designed to illustrate race engineering concepts and machine learning workflows; they do not reproduce proprietary, track-measured telemetry from official Formula 1 teams. All machine learning metrics reflect synthetic simulator data.

---

## 🛡️ Model Deployment Security Notice
Python `pickle` and `joblib` artifacts use bytecode serialization. When loaded via `joblib.load()` or `pickle.load()`, arbitrary Python code can be executed. **Never load model files from untrusted or unverified sources.**

---

## 📖 Telemetry Metrics Glossary

| Heading | Name | Meaning & Impact |
| :--- | :--- | :--- |
| **🏆 Pos** | Race Position | Driver standing (`P1` to `P8`) based on total race distance covered. |
| **🏎️ Driver** | Driver Code | 3-letter abbreviation (`VER`, `HAM`, `LEC`, `NOR`, `PIA`, `RUS`, `ALO`, `SAI`). |
| **🔄 Lap** | Lap Count | Current lap number out of total scheduled Grand Prix laps (e.g. `4/53`). |
| **📍 Lap Progress** | Circuit Completion | Track distance completed in the active lap (`0%` to `100%`). |
| **⏱️ Sector** | Timing Sector | Current circuit sector (`S1` start, `S2` middle technical, `S3` final straight). |
| **⚡ Speed** | Velocity | Real-time velocity in km/h. |
| **⏱️ Lap Time** | Ongoing Lap Pace | Estimated pace for the ongoing lap in seconds. |
| **🏁 Last Lap** | Completed Lap Time | Official time of the last completed lap in seconds. |
| **🔥 Aggression** | Driving Intensity | Multiplier (`>1.10x` aggressive attack, `<0.95x` conservative saving). |
| **🛞 Tyre** | Compound | Current rubber fitted (`Soft`, `Medium`, `Hard`, `Inter`, `Wet`). |
| **📉 Tyre Wear** | Degradation | Wear percentage. Above 60–70%, grip drops and lap times worsen. |
| **⛽ Fuel** | Tank Level | Fuel percentage remaining. Lighter fuel loads generate faster pace. |
| **🌡️ Brake Temp** | Disc Temperature | Carbon-ceramic disc heat in °C. Optimum window is 300°C–650°C. |
| **💨 DRS** | Drag Reduction | Rear wing flap open (`YES`) for top speed boost on straights. |
| **🛑 Status** | Race Status | `TRACK` (racing), `PIT REQ` (queued), `IN PIT` (pit lane), `FIN` (finished). |

---

## 📄 License
This project is licensed under the **MIT License** — free to use, modify, and distribute.
