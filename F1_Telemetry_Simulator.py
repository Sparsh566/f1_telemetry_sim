import streamlit as st
import random
import time
from datetime import datetime
from collections import deque
from pathlib import Path
import io
import csv
import math
import importlib.util
from typing import Dict, Any

# Fast check for ML dependencies without loading heavy libraries at startup
def check_ml_available() -> bool:
    try:
        return all(importlib.util.find_spec(pkg) is not None for pkg in ["pandas", "numpy", "sklearn", "joblib"])
    except Exception:
        return False

# -----------------------------------------------------------------------------
# F1 Circuit Database
# -----------------------------------------------------------------------------
TRACKS: Dict[str, Dict[str, Any]] = {
    "🇮🇹 Monza (Autodromo Nazionale Monza)": {
        "name": "Monza",
        "country": "Italy",
        "length_m": 5793.0,
        "base_lap": 81.0,
        "total_laps": 53,
        "speed_scale": 1.08,
        "max_speed": 355.0,
        "wear_factor": 0.85,
        "brake_factor": 1.25,
        "incident_factor": 0.8,
        "drs_zones": 2,
        "tyre_stress": "Low - Medium",
        "description": "The Temple of Speed — long straights, heavy braking chicanes & lowest aerodynamic downforce."
    },
    "🇬🇧 Silverstone (Silverstone Circuit)": {
        "name": "Silverstone",
        "country": "United Kingdom",
        "length_m": 5891.0,
        "base_lap": 87.5,
        "total_laps": 52,
        "speed_scale": 1.02,
        "max_speed": 335.0,
        "wear_factor": 1.30,
        "brake_factor": 0.95,
        "incident_factor": 0.9,
        "drs_zones": 2,
        "tyre_stress": "High (Maggotts/Becketts)",
        "description": "Home of British Motor Racing — legendary high-speed sweeping corners putting severe lateral load on tyres."
    },
    "🇧🇪 Spa-Francorchamps (Circuit de Spa)": {
        "name": "Spa-Francorchamps",
        "country": "Belgium",
        "length_m": 7004.0,
        "base_lap": 104.5,
        "total_laps": 44,
        "speed_scale": 1.04,
        "max_speed": 342.0,
        "wear_factor": 1.10,
        "brake_factor": 1.05,
        "incident_factor": 1.1,
        "drs_zones": 2,
        "tyre_stress": "Medium - High",
        "description": "The Ardennes masterpiece — the longest circuit on the calendar featuring the iconic Eau Rouge & Raidillon."
    },
    "🇲🇨 Circuit de Monaco (Monte Carlo)": {
        "name": "Monaco",
        "country": "Monaco",
        "length_m": 3337.0,
        "base_lap": 72.0,
        "total_laps": 78,
        "speed_scale": 0.82,
        "max_speed": 288.0,
        "wear_factor": 0.65,
        "brake_factor": 1.20,
        "incident_factor": 1.6,
        "drs_zones": 1,
        "tyre_stress": "Very Low",
        "description": "The Crown Jewel — unforgiving armco barriers, tight hairpins, zero margin for error and immense driver precision."
    },
    "🇯🇵 Suzuka (Suzuka International Racing Course)": {
        "name": "Suzuka",
        "country": "Japan",
        "length_m": 5807.0,
        "base_lap": 89.0,
        "total_laps": 53,
        "speed_scale": 1.00,
        "max_speed": 332.0,
        "wear_factor": 1.25,
        "brake_factor": 0.90,
        "incident_factor": 1.0,
        "drs_zones": 1,
        "tyre_stress": "High (S-Curves)",
        "description": "Historic figure-8 track — famous technical Esses, Degner curves and the blistering flat-out 130R corner."
    },
    "🇧🇭 Bahrain (Bahrain International Circuit)": {
        "name": "Bahrain",
        "country": "Bahrain",
        "length_m": 5412.0,
        "base_lap": 89.5,
        "total_laps": 57,
        "speed_scale": 1.01,
        "max_speed": 338.0,
        "wear_factor": 1.35,
        "brake_factor": 1.25,
        "incident_factor": 0.9,
        "drs_zones": 3,
        "tyre_stress": "Very High (Abrasive)",
        "description": "Desert Sakhir spectacle — abrasive asphalt causing heavy tyre degradation with multiple heavy braking overtaking zones."
    },
    "🇧🇷 Interlagos (Autódromo José Carlos Pace)": {
        "name": "Interlagos",
        "country": "Brazil",
        "length_m": 4309.0,
        "base_lap": 70.8,
        "total_laps": 71,
        "speed_scale": 1.00,
        "max_speed": 334.0,
        "wear_factor": 1.05,
        "brake_factor": 1.00,
        "incident_factor": 1.2,
        "drs_zones": 2,
        "tyre_stress": "Medium",
        "description": "Natural amphitheatre — undulating anti-clockwise layout, Senna S sequence and dramatic race overtakes."
    }
}

DEFAULT_TRACK_KEY = "🇮🇹 Monza (Autodromo Nazionale Monza)"

# -----------------------------------------------------------------------------
# Constants & tuning
# -----------------------------------------------------------------------------
NUM_CARS = 8
CAR_NAMES = ["VER", "HAM", "LEC", "NOR", "RUS", "ALO", "SAI", "PER"][:NUM_CARS]
TYRE_COMPOUNDS = ["Soft", "Medium", "Hard", "Inter", "Wet"]

# Simulation tuning
REFRESH_MS_DEFAULT = 800
MAX_HISTORY = 120

# DRS
DRS_SPEED_THRESHOLD = 210.0
DRS_BOOST_MIN = 7.0
DRS_BOOST_MAX = 14.0
DRS_DURATION_SEC = 5.0
DRS_COOLDOWN_SEC = 15.0

# Brake temps
BRAKE_HEAT_PER_BRAKE = 40.0
BRAKE_COOL_PER_TICK = 2.0
BRAKE_MIN_TEMP = 20.0
BRAKE_MAX_TEMP = 900.0

# Tyre / pit / fuel
TYRE_MAX_GRIP = 1.0
TYRE_MIN_GRIP = 0.6
TYRE_WEAR_PER_LAP_BASE = 0.015
FUEL_EFFECT_PER_PERCENT = 0.02
PIT_LANE_TIME = 20.0
PIT_VARIATION = 2.0

# Safety car
SAFETY_CAR_SPEED = 110.0
INCIDENT_PROB_PER_TICK = 0.0009

# -----------------------------------------------------------------------------
# Core simulator functions
# -----------------------------------------------------------------------------

def make_initial_state(track_info: Dict[str, Any] = None) -> Dict[str, Dict[str, Any]]:
    if track_info is None:
        track_info = TRACKS[DEFAULT_TRACK_KEY]

    base_track_lap = track_info.get("base_lap", 80.0)
    speed_scale = track_info.get("speed_scale", 1.0)

    state: Dict[str, Dict[str, Any]] = {}
    for i, name in enumerate(CAR_NAMES):
        base_lap = random.uniform(base_track_lap - 1.5, base_track_lap + 2.5)
        tyre = random.choice(TYRE_COMPOUNDS[:3])
        aggression = round(random.uniform(0.9, 1.10), 2)
        grid_stagger = float((NUM_CARS - i) * 20.0)
        state[name] = {
            "position": i + 1,
            "lap": 1,
            "lap_distance": grid_stagger,
            "total_distance": grid_stagger,
            "lap_time": base_lap,
            "last_lap_time": None,
            "best_lap_time": round(base_lap, 3),
            "sector": 1,
            "speed": random.uniform(60, 210) * speed_scale,
            "throttle": random.uniform(0.2, 1.0),
            "brake": 0.0,
            "tyre": tyre,
            "tyre_wear": random.uniform(0.0, 0.08),
            "fuel": random.uniform(65, 100),  # percent
            "history": deque(maxlen=MAX_HISTORY),
            "brake_temp": random.uniform(80.0, 250.0),
            "drs_available": True,
            "drs_active": False,
            "drs_timer": 0.0,
            "drs_cooldown": 0.0,
            "pit_request": False,
            "in_pit": False,
            "pit_timer": 0.0,
            "pit_tyre_choice": None,
            "aggression": aggression,
            "incident": False,
            "finished": False,
        }
        # Populate small initial history
        for _ in range(8):
            t = time.time() - random.uniform(0, 60)
            state[name]["history"].append({
                "t": t,
                "speed": random.uniform(60, 210) * speed_scale,
                "lap_time": base_lap + random.uniform(-1.0, 1.0),
                "brake_temp": state[name]["brake_temp"],
                "drs_active": False,
                "tyre_wear": state[name]["tyre_wear"],
                "in_pit": False,
            })
    return state


def simulate_step(state: Dict[str, Dict[str, Any]], dt: float = 1.0, safety_car_active: bool = False, track_info: Dict[str, Any] = None, speed_mult: float = 1.0) -> Dict[str, Dict[str, Any]]:
    if track_info is None:
        track_info = TRACKS[DEFAULT_TRACK_KEY]

    track_length = track_info.get("length_m", 5000.0)
    total_race_laps = track_info.get("total_laps", 50)
    speed_scale = track_info.get("speed_scale", 1.0)
    max_track_speed = track_info.get("max_speed", 335.0)
    wear_factor = track_info.get("wear_factor", 1.0)
    brake_factor = track_info.get("brake_factor", 1.0)
    incident_factor = track_info.get("incident_factor", 1.0)
    drs_zones = track_info.get("drs_zones", 2)

    # Simulation effective elapsed time
    dt_effective = dt * speed_mult
    timestamp = time.time()

    for name, car in state.items():
        if car.get("finished", False):
            continue

        if safety_car_active:
            target_speed = SAFETY_CAR_SPEED
            speed = car["speed"] + (target_speed - car["speed"]) * 0.5
            throttle = 0.18
        else:
            speed_delta = random.uniform(-6, 6) * speed_scale
            speed = car["speed"] + speed_delta
            throttle = max(0.0, min(1.0, car["throttle"] + random.uniform(-0.12, 0.12)))

        aggression = car.get("aggression", 1.0)
        throttle = min(1.0, throttle * aggression)

        # occasional braking events (corners)
        brake = 0.0
        if random.random() < 0.02 * aggression:
            brake = random.uniform(0.4, 1.0)
            throttle = max(0.0, throttle - brake * 0.5)

        # fuel burn
        fuel_burn_rate = 0.06
        fuel = max(0.0, car["fuel"] - fuel_burn_rate * throttle * dt_effective)

        # tyre wear adjusted for track abrasiveness
        wear = TYRE_WEAR_PER_LAP_BASE * (0.5 + throttle) * aggression * wear_factor
        wear *= dt_effective / 90.0
        wear += (brake * 0.002) * (dt_effective / 1.0)
        tyre_wear = min(1.0, car.get("tyre_wear", 0.0) + wear)

        # grip
        compound_modifier = {"Soft": 1.00, "Medium": 0.97, "Hard": 0.94, "Inter": 0.9, "Wet": 0.88}
        grip = max(TYRE_MIN_GRIP, TYRE_MAX_GRIP - tyre_wear * 0.5) * compound_modifier.get(car["tyre"], 1.0)

        # lap_time influenced by base lap, grip, fuel weight, and track characteristics
        base_lap = car.get("lap_time", track_info.get("base_lap", 80.0))
        fuel_penalty = (fuel / 100.0) * FUEL_EFFECT_PER_PERCENT * 100.0
        grip_penalty = (1.0 - grip) * 6.0
        lap_noise = random.uniform(-0.4, 0.6)
        lap_time = max(50.0, base_lap + lap_noise + grip_penalty + fuel_penalty * 0.05)

        # incidents chance scaled by track difficulty
        if not safety_car_active and random.random() < (INCIDENT_PROB_PER_TICK * aggression * incident_factor):
            speed = max(0, speed * 0.5)
            lap_time += random.uniform(1.5, 5.0)
            car["incident"] = True
        else:
            car["incident"] = False

        # brake temp updates scaled by track braking factor
        heat_gain = brake * BRAKE_HEAT_PER_BRAKE * brake_factor
        brake_temp = car.get("brake_temp", BRAKE_MIN_TEMP) + heat_gain
        if brake < 0.01:
            brake_temp -= BRAKE_COOL_PER_TICK * (dt_effective / 1.0)
        brake_temp = max(BRAKE_MIN_TEMP, min(BRAKE_MAX_TEMP, brake_temp))

        # pit stop handling
        if car.get("pit_request") and not car.get("in_pit"):
            car["in_pit"] = True
            car["pit_timer"] = PIT_LANE_TIME + random.uniform(-PIT_VARIATION, PIT_VARIATION)
            car["pit_request"] = False

        if car.get("in_pit"):
            car["pit_timer"] -= dt_effective
            speed = min(speed, 60.0)
            if car["pit_timer"] <= 0:
                new_ty = car.get("pit_tyre_choice", car["tyre"])
                car["tyre"] = new_ty
                car["tyre_wear"] = min(0.20, car.get("tyre_wear", 1.0) * 0.10)
                car["in_pit"] = False
                car["pit_timer"] = 0.0

        # DRS handling (activation chance influenced by DRS zone count)
        drs_chance = 0.03 * drs_zones
        if car["drs_active"]:
            boost = random.uniform(DRS_BOOST_MIN, DRS_BOOST_MAX) * (0.9 + 0.2 * (1 - tyre_wear))
            speed = min(max_track_speed, speed + boost)
            car["drs_timer"] -= dt_effective
            if car["drs_timer"] <= 0:
                car["drs_active"] = False
                car["drs_cooldown"] = DRS_COOLDOWN_SEC
                car["drs_available"] = False
        else:
            if car["drs_available"] and speed >= (DRS_SPEED_THRESHOLD * speed_scale * 0.95) and random.random() < drs_chance:
                car["drs_active"] = True
                car["drs_timer"] = DRS_DURATION_SEC

        if car["drs_cooldown"] > 0:
            car["drs_cooldown"] = max(0.0, car["drs_cooldown"] - dt_effective)
            if car["drs_cooldown"] == 0.0:
                car["drs_available"] = True

        # update values back
        car["speed"] = max(0.0, min(max_track_speed, speed))
        car["throttle"] = throttle
        car["brake"] = brake
        car["fuel"] = fuel
        car["lap_time"] = lap_time
        car["brake_temp"] = brake_temp
        car["tyre_wear"] = tyre_wear

        # Distance, lap and sector progression using effective simulation time
        step_distance = (car["speed"] / 3.6) * dt_effective
        car["lap_distance"] = car.get("lap_distance", 0.0) + step_distance
        car["total_distance"] = car.get("total_distance", 0.0) + step_distance

        # Lap completion
        while car["lap_distance"] >= track_length:
            car["lap"] = car.get("lap", 1) + 1
            car["lap_distance"] -= track_length
            car["last_lap_time"] = round(lap_time, 3)
            if car.get("best_lap_time") is None or lap_time < car["best_lap_time"]:
                car["best_lap_time"] = round(lap_time, 3)
            
            # Check race completion
            if car["lap"] > total_race_laps:
                car["finished"] = True
                car["lap"] = total_race_laps
                break

        car["sector"] = min(3, int(car["lap_distance"] / (track_length / 3.0)) + 1)

        # append to history
        car["history"].append({
            "t": timestamp,
            "speed": car["speed"],
            "lap_time": lap_time,
            "throttle": throttle,
            "brake": brake,
            "brake_temp": brake_temp,
            "drs_active": car["drs_active"],
            "tyre_wear": tyre_wear,
            "in_pit": car.get("in_pit", False),
            "fuel": round(fuel, 3),
        })

    # Update positions using race progress (lap and total_distance)
    sorted_cars = sorted(
        state.items(),
        key=lambda kv: (kv[1].get("lap", 1), kv[1].get("lap_distance", 0.0)),
        reverse=True
    )
    for pos, (name, car) in enumerate(sorted_cars, start=1):
        car["position"] = pos

    return state


def export_csv_bytes(state_arg: Dict[str, Dict[str, Any]], track_name: str = "Monza") -> bytes:
    output = io.StringIO()
    writer = csv.writer(output)
    header = [
        "circuit", "timestamp", "car", "position", "lap", "sector", "lap_time", "speed", "throttle",
        "brake", "brake_temp", "drs_active", "tyre", "tyre_wear", "fuel", "in_pit", "aggression", "incident"
    ]
    writer.writerow(header)
    t0 = time.time()
    for name, car in state_arg.items():
        last = car["history"][-1] if car["history"] else {"t": t0, "speed": car["speed"], "lap_time": car["lap_time"], "brake_temp": car.get("brake_temp", BRAKE_MIN_TEMP)}
        writer.writerow([
            track_name,
            datetime.fromtimestamp(last.get("t", t0)).isoformat(),
            name,
            car["position"],
            car["lap"],
            car.get("sector", 1),
            round(last.get("lap_time", car["lap_time"]), 3),
            round(last.get("speed", car["speed"]), 3),
            round(last.get("throttle", 0.0), 3),
            round(last.get("brake", 0.0), 3),
            round(last.get("brake_temp", car.get("brake_temp", BRAKE_MIN_TEMP)), 2),
            bool(last.get("drs_active", car.get("drs_active", False))),
            car["tyre"],
            round(car.get("tyre_wear", 0.0), 4),
            round(car.get("fuel", 0.0), 3),
            bool(car.get("in_pit", False)),
            round(car.get("aggression", 1.0), 3),
            bool(car.get("incident", False))
        ])
    return output.getvalue().encode("utf-8")


# -----------------------------------------------------------------------------
# ML Helper Functions (Lazy loaded on demand)
# -----------------------------------------------------------------------------
def get_ml_modules():
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_absolute_error
    import joblib
    return pd, np, RandomForestRegressor, Pipeline, OneHotEncoder, StandardScaler, ColumnTransformer, train_test_split, r2_score, mean_absolute_error, joblib

def run_simulated_race(ticks: int = 300, dt: float = 1.0, safety_car_prob: float = 0.0002, track_info: Dict[str, Any] = None):
    pd, np, *_ = get_ml_modules()
    if track_info is None:
        track_info = TRACKS[DEFAULT_TRACK_KEY]
    state_local = make_initial_state(track_info)
    safety_car_active_local = False
    sc_timer_local = 0.0
    for _ in range(int(ticks)):
        if random.random() < float(safety_car_prob):
            safety_car_active_local = not safety_car_active_local
            sc_timer_local = float(random.uniform(30, 90)) if safety_car_active_local else 0.0
        if sc_timer_local > 0:
            sc_timer_local -= dt
            if sc_timer_local <= 0:
                safety_car_active_local = False
        state_local = simulate_step(state_local, dt=dt, safety_car_active=safety_car_active_local, track_info=track_info, speed_mult=1.0)
    rows_local = []
    for name, car in state_local.items():
        history = list(car.get("history", []))
        if not history:
            continue
        avg_speed = float(np.mean([h.get("speed", 0.0) for h in history]))
        avg_lap = float(np.mean([h.get("lap_time", car.get("lap_time", 0.0)) for h in history]))
        avg_brake = float(np.mean([h.get("brake_temp", BRAKE_MIN_TEMP) for h in history]))
        pit_count = int(any(h.get("in_pit", False) for h in history))
        incidents = int(any(h.get("incident", False) for h in history) or bool(car.get("incident", False)))
        rows_local.append({
            "car": name,
            "tyre": car.get("tyre", "Unknown"),
            "start_aggression": float(car.get("aggression", 1.0)),
            "start_fuel": float(car.get("fuel", 0.0)),
            "start_tyre_wear": float(car.get("tyre_wear", 0.0)),
            "start_brake_temp": float(car.get("brake_temp", BRAKE_MIN_TEMP)),
            "avg_speed": avg_speed,
            "avg_brake": avg_brake,
            "pit_count": pit_count,
            "incidents": incidents,
            "target_avg_lap": avg_lap
        })
    return pd.DataFrame(rows_local)

def build_training_dataset(num_races: int = 100, ticks_per_race: int = 300, show_progress: bool = False, track_info: Dict[str, Any] = None):
    pd, *_ = get_ml_modules()
    dfs_local = []
    for i in range(int(num_races)):
        df_local = run_simulated_race(ticks=int(ticks_per_race), track_info=track_info)
        df_local["race_id"] = int(i)
        dfs_local.append(df_local)
        if show_progress and ((i + 1) % max(1, int(num_races // 10)) == 0):
            st.sidebar.write(f"Generated {i+1}/{num_races} simulated races")
    if not dfs_local:
        return pd.DataFrame()
    return pd.concat(dfs_local, ignore_index=True)

def train_model_from_df(df_local, random_state: int = 42):
    pd, np, RandomForestRegressor, Pipeline, OneHotEncoder, StandardScaler, ColumnTransformer, train_test_split, r2_score, mean_absolute_error, _ = get_ml_modules()
    features = ["tyre", "start_aggression", "start_fuel", "start_tyre_wear", "start_brake_temp", "avg_speed", "avg_brake", "pit_count", "incidents"]
    X = df_local[features].copy()
    y = df_local["target_avg_lap"].copy()
    cat_cols = ["tyre"]
    num_cols = [c for c in features if c not in cat_cols]

    ohe_kwargs = {"handle_unknown": "ignore"}
    try:
        OneHotEncoder(sparse_output=False)
        ohe_kwargs["sparse_output"] = False
    except TypeError:
        ohe_kwargs["sparse"] = False

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(**ohe_kwargs), cat_cols),
        ("num", StandardScaler(), num_cols)
    ], remainder="drop")
    rf = RandomForestRegressor(n_estimators=150, random_state=random_state, n_jobs=-1)
    model_local = Pipeline([("pre", pre), ("rf", rf)])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.18, random_state=random_state)
    model_local.fit(X_train, y_train)
    y_pred = model_local.predict(X_test)
    metrics_local = {"r2": float(r2_score(y_test, y_pred)), "mae": float(mean_absolute_error(y_test, y_pred))}
    return model_local, metrics_local, (X_test, y_test, y_pred)

def prepare_features_from_state(state_arg: Dict[str, Dict[str, Any]]):
    pd, *_ = get_ml_modules()
    rows_p = []
    for name, car in state_arg.items():
        last = car.get("history", [])[-1] if car.get("history") else {}
        rows_p.append({
            "car": name,
            "tyre": car.get("tyre", "Unknown"),
            "start_aggression": float(car.get("aggression", 1.0)),
            "start_fuel": float(car.get("fuel", 0.0)),
            "start_tyre_wear": float(car.get("tyre_wear", 0.0)),
            "start_brake_temp": float(car.get("brake_temp", BRAKE_MIN_TEMP)),
            "avg_speed": float(last.get("speed", car.get("speed", 0.0))),
            "avg_brake": float(last.get("brake_temp", car.get("brake_temp", BRAKE_MIN_TEMP))),
            "pit_count": int(car.get("pit_request", False) or car.get("in_pit", False)),
            "incidents": int(car.get("incident", False))
        })
    return pd.DataFrame(rows_p)


# -----------------------------------------------------------------------------
# Streamlit App Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    st.set_page_config(page_title="F1 Telemetry Simulator — Clean Build", layout="wide")

    # Initial circuit selection
    if "selected_track" not in st.session_state:
        st.session_state.selected_track = DEFAULT_TRACK_KEY

    current_track_info = TRACKS[st.session_state.selected_track]

    # Simulation speed multiplier
    if "sim_speed" not in st.session_state:
        st.session_state.sim_speed = 8.0  # default 8x speed so laps advance smoothly in ~10s

    # Session state initialization
    if "sim_state" not in st.session_state:
        st.session_state.sim_state = make_initial_state(current_track_info)
        st.session_state.running = False
        st.session_state.last_update = time.time()
        st.session_state.refresh_ms = REFRESH_MS_DEFAULT
        st.session_state.safety_car = False

    # Advance simulation state if running
    now = time.time()
    if st.session_state.running:
        dt = max(0.1, min(2.0, now - st.session_state.last_update)) if st.session_state.last_update else 1.0
        st.session_state.sim_state = simulate_step(
            st.session_state.sim_state,
            dt=dt,
            safety_car_active=st.session_state.safety_car,
            track_info=current_track_info,
            speed_mult=float(st.session_state.sim_speed)
        )
        st.session_state.last_update = now

    state = st.session_state.sim_state

    # -------------------------------------------------------------------------
    # Header & Track Selector
    # -------------------------------------------------------------------------
    st.title("🏎️ F1 Telemetry Simulator")

    track_col, stat_col1, stat_col2, stat_col3, stat_col4 = st.columns([2.5, 1, 1, 1, 1.2])

    with track_col:
        new_track = st.selectbox(
            "🏁 Choose Grand Prix Circuit",
            options=list(TRACKS.keys()),
            index=list(TRACKS.keys()).index(st.session_state.selected_track),
            key="circuit_dropdown"
        )
        # If user selected a new track, reinitialize state
        if new_track != st.session_state.selected_track:
            st.session_state.selected_track = new_track
            current_track_info = TRACKS[new_track]
            st.session_state.sim_state = make_initial_state(current_track_info)
            st.session_state.running = False
            st.session_state.safety_car = False
            st.session_state.last_update = time.time()
            st.rerun()

    with stat_col1:
        st.metric("Lap Distance", f"{current_track_info['length_m']/1000.0:.3f} km")
    with stat_col2:
        st.metric("Race Laps", f"{current_track_info['total_laps']} Laps")
    with stat_col3:
        st.metric("DRS Zones", f"{current_track_info['drs_zones']} Zones")
    with stat_col4:
        st.metric("Tyre Stress", current_track_info['tyre_stress'])

    st.caption(f"ℹ️ **Circuit Profile**: {current_track_info['description']}")
    st.write("---")

    # -------------------------------------------------------------------------
    # Aggression Randomization Popup / Notification
    # -------------------------------------------------------------------------
    if "aggr_popup_data" in st.session_state:
        popup_data = st.session_state["aggr_popup_data"]
        
        # Streamlit toast alert
        st.toast("🎲 Aggression parameters successfully randomized!", icon="🏎️")
        
        # Prominent popup banner showing all car assignments
        with st.container():
            st.success("### 🎲 Aggression Values Randomized & Applied!")
            p_cols = st.columns(len(popup_data))
            for i, (cname, val) in enumerate(popup_data.items()):
                style_label = "🔥 High" if val >= 1.12 else ("⚡ Balanced" if val >= 0.95 else "🛡️ Safe")
                p_cols[i].metric(label=f"🏎️ {cname}", value=f"{val:.2f}x", delta=style_label)
            st.write("---")
        
        # Clear once displayed
        del st.session_state["aggr_popup_data"]

    # -------------------------------------------------------------------------
    # Controls
    # -------------------------------------------------------------------------
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1.2, 1, 1.2])
    with col1:
        if st.button("Start Simulation", use_container_width=True):
            st.session_state.running = True
            st.session_state.last_update = time.time()
            st.rerun()
        if st.button("Stop Simulation", use_container_width=True):
            st.session_state.running = False
            st.rerun()

    with col2:
        if st.button("Reset Race", use_container_width=True):
            st.session_state.sim_state = make_initial_state(current_track_info)
            st.session_state.running = False
            st.session_state.safety_car = False
            st.session_state.last_update = time.time()
            for cname, c in st.session_state.sim_state.items():
                st.session_state[f"aggr_{cname}"] = c["aggression"]
            st.rerun()

    with col3:
        if st.button("🎲 Randomize aggression", use_container_width=True):
            applied_summary = {}
            for cname, c in st.session_state.sim_state.items():
                val = round(random.uniform(0.85, 1.25), 2)
                c["aggression"] = val
                st.session_state[f"aggr_{cname}"] = val
                applied_summary[cname] = val
            st.session_state["aggr_popup_data"] = applied_summary
            st.rerun()

    with col4:
        sc = st.checkbox("Safety Car active", value=st.session_state.safety_car, key="safety_car_box")
        st.session_state.safety_car = sc

    with col5:
        speed_choice = st.select_slider(
            "⚡ Sim Speed",
            options=[1, 2, 5, 8, 15, 30],
            value=int(st.session_state.sim_speed),
            format_func=lambda x: f"{x}x",
            key="sim_speed_slider"
        )
        st.session_state.sim_speed = speed_choice

    st.write("---")

    # Sidebar controls
    with st.sidebar:
        st.header("Simulation Settings")
        st.session_state.refresh_ms = st.slider("Refresh (ms)", 200, 2000, int(st.session_state.refresh_ms), step=100)
        st.write(f"Active Track: **{current_track_info['name']} ({current_track_info['country']})**")
        st.write(f"Sim Speed Multiplier: **{st.session_state.sim_speed}x**")
        st.write("---")
        st.subheader("Per-car controls")
        selected = st.selectbox("Select car", CAR_NAMES)
        pit_choice_key = f"pit_choice_{selected}"
        pit_tyre_choice = st.selectbox("Pit tyre choice", TYRE_COMPOUNDS, index=0, key=pit_choice_key)
        
        if st.button("Request Pit (selected)"):
            st.session_state.sim_state[selected]["pit_request"] = True
            st.session_state.sim_state[selected]["pit_tyre_choice"] = pit_tyre_choice
            st.success(f"Pit requested for {selected} ({pit_tyre_choice})")

        if st.button("Force DRS (selected)"):
            car = st.session_state.sim_state[selected]
            car["drs_active"] = True
            car["drs_timer"] = DRS_DURATION_SEC
            car["drs_cooldown"] = 0.0
            car["drs_available"] = True
            st.success(f"DRS forced for {selected}!")

        aggr_key = f"aggr_{selected}"
        current_aggr = float(st.session_state.sim_state[selected]["aggression"])
        if aggr_key not in st.session_state:
            st.session_state[aggr_key] = current_aggr
        new_aggr = st.slider("Aggression (selected)", 0.7, 1.4, current_aggr, 0.01, key=aggr_key)
        st.session_state.sim_state[selected]["aggression"] = float(new_aggr)

        st.write("---")
        st.write("Status:", "🟢 Running" if st.session_state.running else "🔴 Stopped")

    # -------------------------------------------------------------------------
    # Leaderboard & Metrics
    # -------------------------------------------------------------------------
    track_len = current_track_info["length_m"]
    rows = []
    for name, car in sorted(state.items(), key=lambda kv: kv[1]["position"]):
        pit_status = "🏁 FIN" if car.get("finished") else ("IN PIT" if car.get("in_pit") else ("PIT REQ" if car.get("pit_request") else "TRACK"))
        progress_pct = int(min(1.0, max(0.0, car.get("lap_distance", 0.0) / track_len)) * 100)
        rows.append({
            "Position": f"P{car['position']}",
            "Driver": name,
            "Lap": f"{car['lap']}/{current_track_info['total_laps']}",
            "Progress": progress_pct,
            "Sector": f"S{car.get('sector', 1)}",
            "Speed (km/h)": round(car["speed"], 1),
            "Lap Time (s)": round(car["lap_time"], 3),
            "Last Lap": f"{car['last_lap_time']:.3f}s" if car.get("last_lap_time") else "—",
            "Aggression": f"{car.get('aggression', 1.0):.2f}x",
            "Tyre": car["tyre"],
            "Wear": round(car['tyre_wear'] * 100, 1),
            "Fuel (%)": round(car["fuel"], 1),
            "Brake Temp (°C)": round(car.get("brake_temp", 0), 1),
            "DRS": "YES" if car.get("drs_active") else "—",
            "Status": pit_status
        })

    st.subheader(f"🏁 Session Leaderboard — {current_track_info['name']} Grand Prix")

    # Column Explanations & Beginner Guide Expander
    with st.expander("📖 Telemetry Column Guide (Click to see what each column means)", expanded=False):
        st.markdown("""
        | Column Heading | Full Meaning | Description & Significance |
        | :--- | :--- | :--- |
        | **🏆 Pos** | **Race Position** | Current standing (`P1` to `P8`) based on total race distance covered. |
        | **🏎️ Driver** | **F1 Driver** | Driver code (`VER` = Verstappen, `HAM` = Hamilton, `LEC` = Leclerc, etc.). |
        | **🔄 Lap** | **Lap Count** | Current completed lap out of total scheduled Grand Prix laps (e.g. `3/53`). |
        | **📍 Lap Progress** | **Circuit Completion %** | How far the car has traveled through the current lap (`0%` = start line, `100%` = finish line). |
        | **⏱️ Sector** | **Track Timing Sector** | The track is split into 3 sectors: `S1` (opening turns), `S2` (middle technical), `S3` (final straight/corners). |
        | **⚡ Speed (km/h)** | **Car Velocity** | Current real-time speed in kilometers per hour. |
        | **⏱️ Lap Time (s)** | **Current Lap Pace** | Estimated pace for the ongoing lap in seconds. |
        | **🏁 Last Lap** | **Previous Lap Time** | The official recorded time of the driver's last completed lap. |
        | **🔥 Aggression** | **Driver Attack Style** | Driving aggression multiplier (`>1.10x` = aggressive/overtaking, `<0.95x` = tyre/fuel saving). |
        | **🛞 Tyre** | **Compound Type** | **Soft** (red, fastest, quick wear), **Medium** (yellow, balanced), **Hard** (white, durable), **Inter/Wet** (rain). |
        | **📉 Tyre Wear** | **Tyre Degradation** | Percentage of tyre tread worn away. When wear passes 60-70%, grip drops and car loses lap time! |
        | **⛽ Fuel Remaining** | **Fuel Tank Level** | Percentage of fuel remaining. Burning fuel makes the car lighter and faster as the race progresses. |
        | **🌡️ Brake Temp** | **Carbon Brake Heat** | Temperature of carbon-ceramic brakes in °C (heats in heavy braking chicanes, cools on straights). |
        | **💨 DRS** | **Drag Reduction Wing** | Rear wing opens on straights when within range for an aerodynamic top-speed boost (`YES` = active). |
        | **🛑 Race Status** | **Pit & Track Status** | `TRACK` = racing, `PIT REQ` = scheduled for tyre change, `IN PIT` = in pit lane, `FIN` = finished race. |
        """)

    # Interactive DataFrame with tooltips and progress bars
    st.dataframe(
        rows,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Position": st.column_config.TextColumn("🏆 Pos", help="Current position in race (P1 to P8) based on total distance covered"),
            "Driver": st.column_config.TextColumn("🏎️ Driver", help="F1 Driver code (VER, HAM, LEC, NOR, RUS, ALO, SAI, PER)"),
            "Lap": st.column_config.TextColumn("🔄 Lap", help="Current Lap / Total Grand Prix Laps"),
            "Progress": st.column_config.ProgressColumn("📍 Lap Progress", help="Track completion percentage for current lap (0% start to 100% finish line)", min_value=0, max_value=100, format="%d%%"),
            "Sector": st.column_config.TextColumn("⏱️ Sector", help="Current circuit timing sector: S1, S2, or S3"),
            "Speed (km/h)": st.column_config.NumberColumn("⚡ Speed (km/h)", help="Instantaneous car velocity in km/h", format="%.1f"),
            "Lap Time (s)": st.column_config.NumberColumn("⏱️ Lap Time (s)", help="Pace estimation for current lap in seconds", format="%.3f"),
            "Last Lap": st.column_config.TextColumn("🏁 Last Lap (s)", help="Official time of last completed lap in seconds"),
            "Aggression": st.column_config.TextColumn("🔥 Aggression", help="Driver aggression level multiplier (>1.10x aggressive attack, <0.95x conservative saving)"),
            "Tyre": st.column_config.TextColumn("🛞 Tyre", help="Current tyre compound (Soft, Medium, Hard, Inter, Wet)"),
            "Wear": st.column_config.ProgressColumn("📉 Tyre Wear", help="Tyre degradation percentage. Worn tyres lose grip and slow down the car", min_value=0, max_value=100, format="%.1f%%"),
            "Fuel (%)": st.column_config.ProgressColumn("⛽ Fuel Remaining", help="Fuel percentage in tank. As fuel burns, car becomes lighter and faster", min_value=0, max_value=100, format="%.1f%%"),
            "Brake Temp (°C)": st.column_config.NumberColumn("🌡️ Brake Temp (°C)", help="Carbon-ceramic brake disc temperature. Optimum range: 300°C–700°C", format="%.0f"),
            "DRS": st.column_config.TextColumn("💨 DRS", help="Drag Reduction System: YES indicates rear wing is open for speed boost"),
            "Status": st.column_config.TextColumn("🛑 Race Status", help="Car status: TRACK (racing), PIT REQ (pit stop queued), IN PIT (in pit box), or FIN (race completed)")
        }
    )

    # Metrics
    fastest = min(state.items(), key=lambda kv: kv[1]["lap_time"])
    st.metric(f"⚡ Fastest Lap Pace ({current_track_info['name']})", f"{round(fastest[1]['lap_time'],3)}s ({fastest[0]})")

    # -------------------------------------------------------------------------
    # Telemetry Trends: Dual View (Interactive Graphs vs Clean Data Tables)
    # -------------------------------------------------------------------------
    st.write("### 📊 Live Telemetry Analysis")
    
    DRIVER_LABELS = {
        "VER": "VER (Verstappen)",
        "HAM": "HAM (Hamilton)",
        "LEC": "LEC (Leclerc)",
        "NOR": "NOR (Norris)",
        "RUS": "RUS (Russell)",
        "ALO": "ALO (Alonso)",
        "SAI": "SAI (Sainz)",
        "PER": "PER (Perez)"
    }

    # Prepare clean DataFrames with explicit headers and index names
    import pandas as pd
    history_len = min(len(car["history"]) for car in state.values()) if state else 0
    ticks_index = [f"T-{history_len - 1 - i}" for i in range(history_len)] if history_len > 0 else []

    speed_dict = {DRIVER_LABELS.get(name, name): [round(p["speed"], 1) for p in car["history"]] for name, car in state.items()}
    df_speed = pd.DataFrame(speed_dict, index=ticks_index)
    df_speed.index.name = "⏱️ Time Sample"

    lap_dict = {DRIVER_LABELS.get(name, name): [round(p.get("lap_time", 0), 3) for p in car["history"]] for name, car in state.items()}
    df_lap = pd.DataFrame(lap_dict, index=ticks_index)
    df_lap.index.name = "⏱️ Time Sample"

    brake_dict = {DRIVER_LABELS.get(name, name): [round(p.get("brake_temp", BRAKE_MIN_TEMP), 1) for p in car["history"]] for name, car in state.items()}
    df_brake = pd.DataFrame(brake_dict, index=ticks_index)
    df_brake.index.name = "⏱️ Time Sample"

    tab_graphs, tab_tables = st.tabs(["📈 Graphical Charts View", "📋 Telemetry Data Tables View"])

    with tab_graphs:
        st.caption("💡 *Hover over charts to see exact figures. Click top-right of any chart to view as table.*")
        col_g1, col_g2, col_g3 = st.columns([1.2, 1.2, 1.1])
        with col_g1:
            st.subheader("⚡ Live Speed Telemetry (km/h)")
            st.line_chart(df_speed)
        with col_g2:
            st.subheader("⏱️ Lap Time History (seconds)")
            st.line_chart(df_lap)
        with col_g3:
            st.subheader("🌡️ Brake Disc Temperatures (°C)")
            st.line_chart(df_brake)

            st.write("---")
            st.subheader("🛞 Tyre Condition Overview")
            wear_table = [{
                "🏎️ Driver": DRIVER_LABELS.get(n, n),
                "🛞 Compound": c["tyre"],
                "📉 Tyre Wear": f"{round(c['tyre_wear']*100, 1)}%",
                "🛡️ Grip Status": "🟢 Optimal Grip" if c["tyre_wear"] < 0.35 else ("🟡 Medium Grip" if c["tyre_wear"] < 0.65 else "🔴 Degraded / Box Now")
            } for n, c in state.items()]
            st.dataframe(wear_table, use_container_width=True, hide_index=True)

    with tab_tables:
        st.caption("📋 *Clear, self-explanatory tabular breakdowns of all car dynamics.*")
        t_col1, t_col2 = st.columns(2)
        with t_col1:
            st.subheader("⚡ Speed & Velocity Breakdown")
            speed_table_rows = []
            for name, car in sorted(state.items(), key=lambda kv: kv[1]["position"]):
                speeds = [p["speed"] for p in car["history"]] if car["history"] else [car["speed"]]
                speed_table_rows.append({
                    "Driver": DRIVER_LABELS.get(name, name),
                    "Current Velocity": f"{car['speed']:.1f} km/h",
                    "Top Speed (Session)": f"{max(speeds):.1f} km/h",
                    "Avg Speed (Session)": f"{sum(speeds)/len(speeds):.1f} km/h",
                    "DRS Wing": "🟢 Open (+Boost)" if car.get("drs_active") else "⚪ Closed"
                })
            st.dataframe(
                speed_table_rows,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Driver": st.column_config.TextColumn("🏎️ Driver (Full Name)", help="Formula 1 driver abbreviation & name"),
                    "Current Velocity": st.column_config.TextColumn("⚡ Current Speed", help="Real-time speed right now"),
                    "Top Speed (Session)": st.column_config.TextColumn("🚀 Max Speed Recorded", help="Fastest velocity recorded this session"),
                    "Avg Speed (Session)": st.column_config.TextColumn("📊 Average Speed", help="Mean speed across recorded telemetry"),
                    "DRS Wing": st.column_config.TextColumn("💨 DRS Status", help="Rear wing state")
                }
            )

        with t_col2:
            st.subheader("⏱️ Lap Timing & Pace Breakdown")
            pace_table_rows = []
            leader_dist = max(c.get("total_distance", 0.0) for c in state.values())
            for name, car in sorted(state.items(), key=lambda kv: kv[1]["position"]):
                gap_m = leader_dist - car.get("total_distance", 0.0)
                gap_str = "LEADER" if car["position"] == 1 else f"+{gap_m:.0f}m"
                pace_table_rows.append({
                    "Pos": f"P{car['position']}",
                    "Driver": DRIVER_LABELS.get(name, name),
                    "Current Pace": f"{car['lap_time']:.3f}s",
                    "Last Lap Time": f"{car['last_lap_time']:.3f}s" if car.get("last_lap_time") else "—",
                    "Best Lap Time": f"{car.get('best_lap_time', car['lap_time']):.3f}s",
                    "Gap to P1": gap_str
                })
            st.dataframe(
                pace_table_rows,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Pos": st.column_config.TextColumn("🏆 Position", help="Current standing in race"),
                    "Driver": st.column_config.TextColumn("🏎️ Driver", help="Driver code and name"),
                    "Current Pace": st.column_config.TextColumn("⏱️ Current Lap Time", help="Estimated pace of ongoing lap"),
                    "Last Lap Time": st.column_config.TextColumn("🏁 Previous Lap", help="Time achieved on last completed lap"),
                    "Best Lap Time": st.column_config.TextColumn("🥇 Personal Best Lap", help="Driver's fastest lap so far"),
                    "Gap to P1": st.column_config.TextColumn("🎯 Distance to Leader", help="Distance behind the race leader")
                }
            )

        st.write("---")
        st.subheader("🌡️ Brakes & Tyre Degradation Breakdown")
        tech_rows = []
        for name, car in sorted(state.items(), key=lambda kv: kv[1]["position"]):
            b_temp = car.get("brake_temp", BRAKE_MIN_TEMP)
            b_status = "🟢 Optimum (Biting)" if (300 <= b_temp <= 650) else ("🟡 Cold (Below Temp)" if b_temp < 300 else "🔴 Overheated (>650°C)")
            tech_rows.append({
                "Driver": DRIVER_LABELS.get(name, name),
                "Brake Temp (°C)": round(b_temp, 0),
                "Brake Health": b_status,
                "Compound": car["tyre"],
                "Tyre Degradation (%)": round(car["tyre_wear"] * 100, 1),
                "Remaining Fuel (%)": round(car["fuel"], 1),
                "Aggression": f"{car.get('aggression', 1.0):.2f}x"
            })
        st.dataframe(
            tech_rows,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Driver": st.column_config.TextColumn("🏎️ Driver", help="F1 Driver"),
                "Brake Temp (°C)": st.column_config.NumberColumn("🌡️ Brake Temp (°C)", help="Carbon disc temperature in °C", format="%.0f"),
                "Brake Health": st.column_config.TextColumn("🩺 Brake Thermal Window", help="Optimum operating temperature is 300°C–650°C"),
                "Compound": st.column_config.TextColumn("🛞 Tyre Compound", help="Soft, Medium, Hard, Inter, Wet"),
                "Tyre Degradation (%)": st.column_config.ProgressColumn("📉 Tyre Wear", help="Degradation level (higher = less grip)", min_value=0, max_value=100, format="%.1f%%"),
                "Remaining Fuel (%)": st.column_config.ProgressColumn("⛽ Fuel Level", help="Percentage of fuel remaining", min_value=0, max_value=100, format="%.1f%%"),
                "Aggression": st.column_config.TextColumn("🔥 Driving Aggression", help="Current aggression multiplier")
            }
        )

    st.markdown("---")

    # Show active DRS cars
    active_drs = [DRIVER_LABELS.get(name, name) for name, car in state.items() if car.get("drs_active")]
    st.write(f"**💨 DRS Active Wings Open:** {', '.join(active_drs) if active_drs else 'None (All closed)'}")

    csv_bytes = export_csv_bytes(state, track_name=current_track_info["name"])
    st.download_button("Download current telemetry (CSV)", csv_bytes, file_name=f"telemetry_{current_track_info['name'].lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")

    st.markdown("---")
    st.caption("Built-in models are approximate and tuned for demo realism with real-world circuit parameters.")

    # -------------------------------------------------------------------------
    # ML Extension UI
    # -------------------------------------------------------------------------
    ml_installed = check_ml_available()
    if not ml_installed:
        with st.sidebar.expander("📈 Machine Learning: Predict next race", expanded=False):
            st.write("Machine Learning tools require additional Python packages: pandas, numpy, scikit-learn, joblib.")
            st.write("Install them in your environment with: `pip install pandas numpy scikit-learn joblib` and then restart Streamlit.")
    else:
        with st.sidebar.expander("📈 Machine Learning: Predict next race", expanded=False):
            st.write(f"Train model on synthetic races at **{current_track_info['name']}**.")
            num_races = st.number_input("Number of simulated races to generate (training)", min_value=20, max_value=1000, value=100, step=20)
            ticks_per_race = st.number_input("Ticks per simulated race (higher = more realistic)", min_value=50, max_value=1000, value=300, step=50)
            if st.button("Generate dataset & Train model"):
                with st.spinner("Generating dataset and training model — this may take a while..."):
                    *_, joblib = get_ml_modules()
                    df_train = build_training_dataset(num_races=int(num_races), ticks_per_race=int(ticks_per_race), show_progress=True, track_info=current_track_info)
                    st.session_state["ml_dataset"] = df_train
                    model_tr, metrics_tr, _ = train_model_from_df(df_train)
                    st.session_state["ml_model"] = model_tr
                    st.session_state["ml_metrics"] = metrics_tr
                    st.success(f"Training done — R²: {metrics_tr['r2']:.3f}, MAE: {metrics_tr['mae']:.3f}s")
                    buf = io.BytesIO()
                    joblib.dump(model_tr, buf)
                    buf.seek(0)
                    st.session_state["ml_model_blob"] = buf.read()

            if "ml_metrics" in st.session_state:
                m_local = st.session_state["ml_metrics"]
                st.write(f"Last training R²: **{m_local['r2']:.3f}**, MAE: **{m_local['mae']:.3f}s**")
                if st.button("Show feature importances (approx)"):
                    try:
                        rf_local = st.session_state["ml_model"].named_steps["rf"]
                        pre_local = st.session_state["ml_model"].named_steps["pre"]
                        try:
                            feature_names_local = list(pre_local.get_feature_names_out())
                        except Exception:
                            ohe_local = pre_local.named_transformers_["cat"]
                            cat_names = list(ohe_local.get_feature_names_out(["tyre"]))
                            num_names = pre_local.transformers_[1][2]
                            feature_names_local = list(cat_names) + list(num_names)
                        importances_local = rf_local.feature_importances_
                        fi_local = sorted(zip(feature_names_local, importances_local), key=lambda x: x[1], reverse=True)
                        for name_f, imp_f in fi_local:
                            st.write(f"{name_f}: {imp_f:.3f}")
                    except Exception as e:
                        st.write("Could not extract importances:", e)

            if "ml_model_blob" in st.session_state:
                st.download_button("Download trained model (.joblib)", st.session_state["ml_model_blob"], file_name=f"f1_rf_model_{current_track_info['name'].lower()}.joblib", mime="application/octet-stream")

        # Predictions
        if "ml_model" in st.session_state:
            _, np, *_, joblib = get_ml_modules()
            model_local = st.session_state["ml_model"]
            df_feat_local = prepare_features_from_state(state)
            if not df_feat_local.empty:
                preds_local = model_local.predict(df_feat_local.drop(columns=["car"]))
                df_feat_local["predicted_avg_lap"] = np.round(preds_local, 3)
                st.subheader(f"ML Predictions for {current_track_info['name']} (predicted avg lap time)")
                st.table(df_feat_local[["car", "tyre", "start_aggression", "start_fuel", "start_tyre_wear", "predicted_avg_lap"]])
                csv_pred = df_feat_local.to_csv(index=False).encode("utf-8")
                st.download_button("Download ML predictions (CSV)", csv_pred, file_name=f"ml_predictions_{current_track_info['name'].lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")
        else:
            with st.sidebar.expander("ML status"):
                st.info("Train an ML model using the sidebar to enable predictions for the current session.")

    # Auto-rerun loop for smooth live simulation
    if st.session_state.running:
        time.sleep(max(0.1, st.session_state.refresh_ms / 1000.0))
        st.rerun()
