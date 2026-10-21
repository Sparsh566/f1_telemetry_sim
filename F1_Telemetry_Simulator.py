import streamlit as st
import pandas as pd
import numpy as np
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
CAR_NAMES = ["VER", "HAM", "LEC", "NOR", "PIA", "RUS", "ALO", "SAI"][:NUM_CARS]
DRIVER_LABELS = {
    "VER": "VER (Verstappen - Red Bull)",
    "HAM": "HAM (Hamilton - Ferrari)",
    "LEC": "LEC (Leclerc - Ferrari)",
    "NOR": "NOR (Norris - McLaren)",
    "PIA": "PIA (Piastri - McLaren)",
    "RUS": "RUS (Russell - Mercedes)",
    "ALO": "ALO (Alonso - Aston Martin)",
    "SAI": "SAI (Sainz - Williams)"
}
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
            "pit_stops": 0,
            "incident_count": 0,
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
            car["incident_count"] = car.get("incident_count", 0) + 1
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
                car["pit_stops"] = car.get("pit_stops", 0) + 1

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


def fast_forward_race_to_finish(state: Dict[str, Dict[str, Any]], track_info: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Instantly fast-forwards remaining race laps to produce the final classification,
    podium standings, and 100% completed telemetry state for all cars.
    """
    total_laps = track_info.get("total_laps", 53)
    track_len = track_info.get("length_m", 5793.0)
    base_lap = track_info.get("base_lap", 80.0)

    for name, car in state.items():
        curr_lap = car.get("lap", 1)
        remaining = max(0, total_laps - curr_lap)
        aggr = car.get("aggression", 1.0)

        # Simulate final lap times and fastest lap
        for _ in range(curr_lap, total_laps + 1):
            sim_lap = round(base_lap * (random.uniform(0.97, 1.02) / (aggr ** 0.3)), 3)
            if car.get("best_lap_time") is None or sim_lap < car["best_lap_time"]:
                car["best_lap_time"] = sim_lap
            car["last_lap_time"] = sim_lap

        car["lap"] = total_laps
        car["lap_distance"] = track_len
        car["total_distance"] = (track_len * total_laps) + (aggr * 150.0) + random.uniform(0.0, 100.0)
        car["finished"] = True
        car["sector"] = 3
        car["fuel"] = max(2.0, round(car.get("fuel", 50.0) - remaining * 1.5, 1))
        car["tyre_wear"] = min(0.92, round(car.get("tyre_wear", 0.15) + remaining * 0.014, 3))
        car["speed"] = round(random.uniform(270.0, track_info.get("max_speed", 335.0)), 1)
        car["in_pit"] = False
        car["pit_request"] = False

    # Rank final classification by total distance
    ranked = sorted(state.items(), key=lambda kv: kv[1]["total_distance"], reverse=True)
    for pos, (cname, c) in enumerate(ranked, 1):
        c["position"] = pos

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
# ML & Deep Learning Pipeline (Modular 'ml' Package)
# -----------------------------------------------------------------------------
from ml.dataset import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
    FEATURE_LABELS,
    validate_feature_schema,
    build_leak_free_race_dataset,
    split_by_race_group
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


def build_training_dataset(num_races: int = 40, show_progress: bool = False, track_info: Dict[str, Any] = None) -> pd.DataFrame:
    """
    Constructs a leak-free multi-race training dataset tagged with unique race_id
    and matching live telemetry feature definitions.
    """
    base_lap = float(track_info.get("base_lap", 85.0)) if track_info else 85.0
    return build_leak_free_race_dataset(num_races=int(num_races), base_lap=base_lap, random_state=42)


def train_model_from_df(df_local: pd.DataFrame, random_state: int = 42):
    """
    Trains RandomForest with group-based race splitting and baseline evaluation.
    """
    pipeline, df_compare, metadata, (X_test, y_test) = train_f1_lap_predictor(
        df=df_local,
        random_state=random_state
    )
    metrics = {
        "r2": metadata["metrics"]["test_r2"],
        "mae": metadata["metrics"]["test_mae"],
        "rmse": metadata["metrics"]["test_rmse"],
        "baseline_mae": metadata["metrics"]["baseline_test_mae"],
        "lift": metadata["metrics"]["lift_mae_seconds"]
    }
    y_pred = pipeline.predict(X_test)
    return pipeline, metrics, (X_test, y_test.values, y_pred), metadata


def prepare_features_from_state(state_arg: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Extracts live telemetry features ensuring 100% schema and semantic
    consistency with model training features.
    """
    rows_p = []
    for name, car in state_arg.items():
        speed = float(car.get("speed", 220.0))
        throttle = float(car.get("throttle", 0.8) * 100.0)
        brake = float(car.get("brake", 0.0) * 100.0)
        tyre_wear = float(car.get("tyre_wear", 0.05) * 100.0)
        fuel = float(car.get("fuel", 80.0))
        brake_temp = float(car.get("brake_temp", BRAKE_MIN_TEMP))
        aggression = float(car.get("aggression", 1.0))
        pit_stops = int(car.get("pit_stops", 0))
        incident_count = int(car.get("incident_count", 0))
        tyre = car.get("tyre", "Medium")

        rows_p.append({
            "car": name,
            "tyre": tyre,
            "speed": round(speed, 1),
            "throttle": round(throttle, 1),
            "brake": round(brake, 1),
            "tyre_wear": round(tyre_wear, 1),
            "fuel": round(fuel, 1),
            "brake_temp": round(brake_temp, 1),
            "aggression": round(aggression, 2),
            "pit_stops": pit_stops,
            "incident_count": incident_count
        })

    df = pd.DataFrame(rows_p)
    return validate_feature_schema(df, expected_features=ALL_FEATURES, require_target=False)


def generate_race_commentary(state: Dict[str, Any], track_info: Dict[str, Any], safety_car: bool) -> List[str]:
    """Generates lively, authentic F1 broadcast pitwall commentary in plain English."""
    comments = []
    if safety_car:
        comments.append("🚨 **SAFETY CAR DEPLOYED!** The entire field is slowed down to 120 km/h. Gaps between cars are shrinking—this is the golden moment for a pit stop!")
        return comments

    sorted_cars = sorted(state.items(), key=lambda kv: kv[1]["position"])
    if not sorted_cars:
        return ["🏎️ Preparing race start..."]

    leader_name, leader_car = sorted_cars[0]
    p2_name, p2_car = sorted_cars[1] if len(sorted_cars) > 1 else (None, None)

    all_fin = all(c.get("finished", False) for c in state.values())
    if all_fin:
        comments.append(f"🏆 **CHEQUERED FLAG!** {DRIVER_LABELS.get(leader_name, leader_name)} takes VICTORY at the {track_info['name']} Grand Prix! What an incredible drive!")
        return comments

    lap = leader_car.get("lap", 1)
    tot_laps = track_info["total_laps"]
    comments.append(f"🏎️ **Lap {lap}/{tot_laps}**: {DRIVER_LABELS.get(leader_name, leader_name)} is leading the Grand Prix, flying at **{leader_car['speed']:.0f} km/h**!")

    if p2_car:
        gap_m = leader_car.get("total_distance", 0.0) - p2_car.get("total_distance", 0.0)
        gap_sec = gap_m / max(1.0, (p2_car.get("speed", 220.0) / 3.6))
        if gap_sec < 1.0:
            comments.append(f"⚔️ **WHEEL-TO-WHEEL BATTLE!** {DRIVER_LABELS.get(p2_name, p2_name)} is within 1 second (**+{gap_sec:.2f}s**) of the leader and hunting for P1!")
        elif gap_sec < 3.5:
            comments.append(f"⏱️ **Podium Fight**: {DRIVER_LABELS.get(p2_name, p2_name)} is trailing P1 by **+{gap_sec:.1f}s**.")

    # Tyre degradation warnings
    danger_tyres = [DRIVER_LABELS.get(n, n) for n, c in state.items() if c.get("tyre_wear", 0.0) > 0.60 and not c.get("in_pit") and not c.get("finished")]
    if danger_tyres:
        comments.append(f"⚠️ **TYRE WEAR ALERT**: {', '.join(danger_tyres[:2])} has worn tyres (>60% wear). Grip is dropping fast—time to call a pit stop!")

    # Pit lane activity
    in_pit = [DRIVER_LABELS.get(n, n) for n, c in state.items() if c.get("in_pit")]
    if in_pit:
        comments.append(f"🛑 **PIT STOP IN PROGRESS**: {', '.join(in_pit)} is in the pit lane right now getting fresh tyres!")

    # DRS active
    drs_cars = [DRIVER_LABELS.get(n, n) for n, c in state.items() if c.get("drs_active")]
    if drs_cars:
        comments.append(f"💨 **DRS WING OPEN**: {', '.join(drs_cars)} popped rear wing open for an extra top-speed boost!")

    return comments


# -----------------------------------------------------------------------------
# Streamlit App Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    st.set_page_config(page_title="F1 2025 Telemetry Simulator & Predictive Analytics", layout="wide")

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
    st.title("🏎️ Formula 1 2025 Telemetry Simulator & Predictive Analytics")
    st.caption("🔴 **Official 2025 Championship Season Grid** • Lewis Hamilton at Ferrari • Real-Time Telemetry & ML Ensemble Studio")

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
    # Experience Level Mode Selector
    # -------------------------------------------------------------------------
    c_mode1, c_mode2 = st.columns([2.3, 1.1])
    with c_mode1:
        exp_mode = st.radio(
            "🎮 Choose Your Experience Mode:",
            options=[
                "🟢 Easy Spectator Mode (Watch the race, see the 2D map & call pit stops — beginner friendly)",
                "🔬 Pro Telemetry & Data Science Studio (Detailed sensor graphs, PCA, Ensembles & Neural Networks)"
            ],
            index=0 if st.session_state.get("is_spectator", True) else 1,
            horizontal=True,
            key="radio_exp_mode"
        )
        is_spectator = "Easy" in exp_mode
        st.session_state["is_spectator"] = is_spectator
    with c_mode2:
        if is_spectator:
            st.success("🟢 **Spectator Mode Active**\n\nSit back, watch the 2D circuit map, and call pit stops!")
        else:
            st.info("🔬 **Pro Studio Active**\n\nFull sensor graphs, PCA, Ensembles & Neural Networks.")

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
    # Simplified Control Station (Beginner Friendly)
    # -------------------------------------------------------------------------
    col1, col2, col3, col4, col5 = st.columns([1.5, 1.2, 1.3, 1.0, 1.2])
    with col1:
        if not st.session_state.running:
            if st.button("🟢 START GRAND PRIX!", type="primary", use_container_width=True, key="btn_start_race"):
                st.session_state.running = True
                st.session_state.last_update = time.time()
                st.rerun()
        else:
            if st.button("⏸️ PAUSE RACE", type="secondary", use_container_width=True, key="btn_pause_race"):
                st.session_state.running = False
                st.rerun()

    with col2:
        if st.button("🔄 Restart Race", use_container_width=True, key="btn_reset_race"):
            st.session_state.sim_state = make_initial_state(current_track_info)
            st.session_state.running = False
            st.session_state.safety_car = False
            st.session_state.race_finished_banner = False
            st.session_state.last_update = time.time()
            for cname, c in st.session_state.sim_state.items():
                st.session_state[f"aggr_{cname}"] = c["aggression"]
            st.rerun()

    with col3:
        speed_choice = st.select_slider(
            "⚡ Sim Speed",
            options=[1, 2, 5, 8, 15, 30],
            value=int(st.session_state.sim_speed),
            format_func=lambda x: f"{x}x ({'Slow' if x<=2 else ('Ideal' if x<=8 else 'Fast')})",
            key="sim_speed_slider"
        )
        st.session_state.sim_speed = speed_choice

    with col4:
        sc = st.toggle("🚨 Safety Car", value=st.session_state.safety_car, key="safety_car_box")
        st.session_state.safety_car = sc

    with col5:
        if st.button("🏁 Skip to Finish", use_container_width=True, key="btn_fast_finish"):
            st.session_state.sim_state = fast_forward_race_to_finish(st.session_state.sim_state, current_track_info)
            st.session_state.running = False
            st.session_state.race_finished_banner = True
            st.rerun()

    st.write("---")

    # Sidebar controls
    with st.sidebar:
        st.header("🏁 Race Control Station")
        st.write(f"Circuit: **{current_track_info['name']} ({current_track_info['country']})**")
        st.write(f"Race Pace: **{st.session_state.sim_speed}x Real Time**")
        st.write("Race Status:", "🟢 Running" if st.session_state.running else "🔴 Paused / Stopped")

        st.write("---")
        st.subheader("📞 Pitwall Quick Radio")
        st.caption("Call any driver in for new tyres to fix worn rubber!")
        selected = st.selectbox("Choose Driver:", CAR_NAMES, format_func=lambda c: DRIVER_LABELS.get(c, c))
        pit_choice_key = f"pit_choice_{selected}"
        pit_tyre_choice = st.selectbox(
            "Tyre Compound:",
            options=["Soft", "Medium", "Hard", "Inter", "Wet"],
            format_func=lambda t: f"{'🔴' if t=='Soft' else ('🟡' if t=='Medium' else ('⚪' if t=='Hard' else '🟢'))} {t} ({'Fastest' if t=='Soft' else ('Balanced' if t=='Medium' else 'Longest lasting')})",
            index=0,
            key=pit_choice_key
        )

        if st.button("📞 BOX, BOX, BOX! (Call Pit Stop)", type="primary", use_container_width=True, key="btn_side_pit"):
            st.session_state.sim_state[selected]["pit_request"] = True
            st.session_state.sim_state[selected]["pit_tyre_choice"] = pit_tyre_choice
            st.toast(f"📞 Pit stop confirmed for {selected}! Entering pits for {pit_tyre_choice} tyres.", icon="🛑")
            st.success(f"Pit queued for {selected} ({pit_tyre_choice})")

        if not is_spectator:
            st.write("---")
            st.subheader("🔬 Advanced Telemetry Controls")
            st.session_state.refresh_ms = st.slider("Screen Refresh (ms)", 200, 2000, int(st.session_state.refresh_ms), step=100)
            if st.button("Force DRS Wing Open (Selected)"):
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
            new_aggr = st.slider("Driver Aggression", min_value=0.7, max_value=1.4, step=0.01, key=aggr_key)
            st.session_state.sim_state[selected]["aggression"] = float(new_aggr)

            def _cb_randomize_grid_aggression():
                applied_summary = {}
                for cname, c in st.session_state.sim_state.items():
                    val = round(random.uniform(0.85, 1.25), 2)
                    c["aggression"] = val
                    st.session_state[f"aggr_{cname}"] = val
                    applied_summary[cname] = val
                st.session_state["aggr_popup_data"] = applied_summary

            st.button(
                "🎲 Randomize Grid Aggression",
                use_container_width=True,
                key="btn_randomize_grid_aggr",
                on_click=_cb_randomize_grid_aggression,
            )

    # -------------------------------------------------------------------------
    # Leaderboard & Metrics
    # -------------------------------------------------------------------------
    track_len = current_track_info["length_m"]
    total_laps = current_track_info["total_laps"]
    total_race_dist = track_len * total_laps

    # Check if all cars have completed the scheduled laps
    all_finished = all(car.get("finished", False) for car in state.values())
    if all_finished:
        st.session_state.running = False

    # Grand Prix Completion Progress Bar (Leader pace)
    leader_car = min(state.values(), key=lambda c: c["position"]) if state else None
    if leader_car:
        l_lap = total_laps if (all_finished or leader_car.get("finished")) else min(total_laps, leader_car.get("lap", 1))
        l_pct = 100 if (all_finished or leader_car.get("finished")) else int(min(100.0, max(0.0, (leader_car.get("total_distance", 0.0) / total_race_dist) * 100.0)))
        st.progress(
            l_pct / 100.0,
            text=f"🏁 Grand Prix Race Progress: Lap {l_lap}/{total_laps} ({l_pct}% Completed)"
        )

    # Podium Celebration Card when finished or directly requested
    if all_finished or st.session_state.get("race_finished_banner"):
        sorted_final = sorted(state.items(), key=lambda kv: kv[1]["position"])
        w_name, w_car = sorted_final[0]
        st.success(f"### 🏆 GRAND PRIX COMPLETE! WINNER: {DRIVER_LABELS.get(w_name, w_name)} 🥇")
        pod1, pod2, pod3 = st.columns(3)
        with pod1:
            st.info(f"🥇 **P1 WINNER**: {DRIVER_LABELS.get(w_name, w_name)}\n\n* Best Lap: `{w_car.get('best_lap_time', 'N/A')}s`\n* Final Tyre: `{w_car['tyre']}` ({round(w_car['tyre_wear']*100, 1)}% wear)\n* Status: `Finished (Chequered Flag)`")
        if len(sorted_final) > 1:
            p2_n, p2_c = sorted_final[1]
            with pod2:
                st.info(f"🥈 **P2 RUNNER-UP**: {DRIVER_LABELS.get(p2_n, p2_n)}\n\n* Best Lap: `{p2_c.get('best_lap_time', 'N/A')}s`\n* Final Tyre: `{p2_c['tyre']}` ({round(p2_c['tyre_wear']*100, 1)}% wear)\n* Status: `Finished`")
        if len(sorted_final) > 2:
            p3_n, p3_c = sorted_final[2]
            with pod3:
                st.info(f"🥉 **P3 PODIUM**: {DRIVER_LABELS.get(p3_n, p3_n)}\n\n* Best Lap: `{p3_c.get('best_lap_time', 'N/A')}s`\n* Final Tyre: `{p3_c['tyre']}` ({round(p3_c['tyre_wear']*100, 1)}% wear)\n* Status: `Finished`")

    rows = []
    for name, car in sorted(state.items(), key=lambda kv: kv[1]["position"]):
        is_fin = car.get("finished", False) or (car.get("lap", 1) >= total_laps and car.get("lap_distance", 0.0) >= track_len)
        pit_status = "🏁 FIN" if is_fin else ("IN PIT" if car.get("in_pit") else ("PIT REQ" if car.get("pit_request") else "TRACK"))

        # Lap Progress: Locked to 100% when finished, else percentage of the active lap
        if is_fin or car.get("lap", 1) > total_laps:
            lap_progress_pct = 100
            race_progress_pct = 100
        else:
            lap_progress_pct = int(min(100.0, max(0.0, (car.get("lap_distance", 0.0) / track_len) * 100.0)))
            race_progress_pct = int(min(100.0, max(0.0, (car.get("total_distance", 0.0) / total_race_dist) * 100.0)))

        rows.append({
            "Position": f"P{car['position']}",
            "Driver": name,
            "Lap": f"{min(total_laps, car['lap'])}/{total_laps}",
            "Progress": lap_progress_pct,
            "Race Progress": race_progress_pct,
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
        | **📍 Lap Progress** | **Lap Completion %** | How far the car has traveled through the current lap (`0%` = start line, `100%` = finish line). |
        | **🏁 Race Progress** | **Grand Prix Completion %** | Total race distance completed out of the entire scheduled Grand Prix. |
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
            "Driver": st.column_config.TextColumn("🏎️ Driver", help="F1 2025 Driver code (VER, HAM, LEC, NOR, PIA, RUS, ALO, SAI)"),
            "Lap": st.column_config.TextColumn("🔄 Lap", help="Current Lap / Total Grand Prix Laps"),
            "Progress": st.column_config.ProgressColumn("📍 Lap Progress", help="Track completion percentage for current lap (0% start to 100% finish line)", min_value=0, max_value=100, format="%d%%"),
            "Race Progress": st.column_config.ProgressColumn("🏁 Race Progress", help="Overall Grand Prix completion percentage", min_value=0, max_value=100, format="%d%%"),
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

    # Prepare clean DataFrames with explicit headers and index names
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

    tab_spectator, tab_graphs, tab_tables, tab_dim_red, tab_ensemble, tab_neural_net, tab_beginner = st.tabs([
        "🏎️ Live Race Broadcast (Easy Mode)",
        "📈 Graphical Charts View",
        "📋 Telemetry Data Tables View",
        "📉 Dimensionality Reduction (PCA & t-SNE)",
        "🌳 Ensemble Methods & Diagnostics",
        "🧠 Neural Network Lab & Deployment",
        "📖 F1 Beginner's Rules & Strategy Guide"
    ])

    # =========================================================================
    # TAB 1: LIVE SPECTATOR BROADCAST (NON-TECH FRIENDLY EASY MODE)
    # =========================================================================
    with tab_spectator:
        st.caption("🏎️ *Live Grand Prix broadcast view designed for everyone. Sit back, watch the cars circle the circuit, follow the live commentary, and make pit stop calls!*")

        # Top 3-step beginner guide callout
        st.success("""
        👋 **Welcome to the Grand Prix Pit Wall! Here is how to play in 3 easy steps:**
        1. Click **🟢 START GRAND PRIX!** at the top to release the cars onto the circuit.
        2. Watch the **Live 2D Track Map** below to see cars cornering and overtaking in real time!
        3. Check tyre wear in the table below: when rubber turns **🔴 Red (>60% wear)**, use the **Pitstop Radio** to call the driver in for fresh tyres!
        """)

        # 2-Column: Live 2D Track Map & Live Commentary + Pitstop Radio
        col_spec_map, col_spec_pitwall = st.columns([1.4, 1.0])

        with col_spec_map:
            st.markdown("#### 🗺️ Live 2D Grand Prix Circuit Track Map")
            st.caption(f"Real-time car positions on **{current_track_info['name']} ({current_track_info['country']})**. Colored dots match official team liveries.")
            try:
                import f1_circuit_maps
                df_map_cars = f1_circuit_maps.get_live_car_coordinates(
                    track_name=current_track_info["name"],
                    state=state,
                    track_length_m=current_track_info["length_m"]
                )
                track_chart = f1_circuit_maps.render_circuit_map_altair(
                    track_name=current_track_info["name"],
                    df_cars=df_map_cars
                )
                st.altair_chart(track_chart, use_container_width=True)
            except Exception as e:
                st.info("Live circuit map rendering...")

        with col_spec_pitwall:
            st.markdown("#### 🎙️ Live Pitwall Radio & Commentary")
            commentary_lines = generate_race_commentary(state, current_track_info, st.session_state.safety_car)
            for comment in commentary_lines:
                st.info(comment)

            st.write("---")
            st.markdown("#### 📞 1-Click Pitstop Command Radio")
            st.caption("Take the pitwall headset and order a tyre change for your driver!")

            c_spec_p1, c_spec_p2 = st.columns(2)
            with c_spec_p1:
                spec_driver_choice = st.selectbox(
                    "Driver to Box:",
                    CAR_NAMES,
                    format_func=lambda c: DRIVER_LABELS.get(c, c),
                    key="spec_driver_select"
                )
            with c_spec_p2:
                spec_tyre_choice = st.selectbox(
                    "New Tyre Compound:",
                    options=["Soft", "Medium", "Hard"],
                    format_func=lambda t: f"{'🔴' if t=='Soft' else ('🟡' if t=='Medium' else '⚪')} {t} ({'Fast' if t=='Soft' else ('Balanced' if t=='Medium' else 'Durable')})",
                    key="spec_tyre_select"
                )

            if st.button("📞 BOX, BOX, BOX! (Call Driver to Pits)", type="primary", use_container_width=True, key="btn_spec_box_now"):
                st.session_state.sim_state[spec_driver_choice]["pit_request"] = True
                st.session_state.sim_state[spec_driver_choice]["pit_tyre_choice"] = spec_tyre_choice
                st.toast(f"🛑 Pit confirmed! {spec_driver_choice} will enter the pit lane on the next lap for {spec_tyre_choice} tyres.", icon="🏎️")
                st.success(f"Pit stop scheduled for {DRIVER_LABELS.get(spec_driver_choice, spec_driver_choice)}!")

        st.write("---")
        st.subheader("📺 TV Broadcast Tower: Live Race Leaderboard")
        st.caption("Simplified standings with driver positions, speeds, and easy-to-read colour-coded tyre health.")

        # Friendly TV Tower display
        spec_rows = []
        leader_dist = max(c.get("total_distance", 0.0) for c in state.values()) if state else 0.0
        for name, car in sorted(state.items(), key=lambda kv: kv[1]["position"]):
            gap_m = leader_dist - car.get("total_distance", 0.0)
            gap_str = "LEADER 🥇" if car["position"] == 1 else f"+{gap_m / max(1.0, (car['speed']/3.6)):.1f}s"
            wear_pct = round(car["tyre_wear"] * 100, 1)
            wear_badge = "🟢 Fresh (High Grip)" if wear_pct < 35 else ("🟡 Used (Normal)" if wear_pct < 65 else "🔴 Critical (BOX NOW!)")

            spec_rows.append({
                "Position": f"P{car['position']}",
                "Driver": DRIVER_LABELS.get(name, name),
                "Speed": f"{car['speed']:.0f} km/h",
                "Tyre": f"{car['tyre']}",
                "Grip Status": wear_badge,
                "Tyre Wear (%)": wear_pct,
                "Gap to P1": gap_str,
                "DRS Wing": "🟢 Open (+Speed)" if car.get("drs_active") else "⚪ Closed",
                "Status": "🛑 IN PIT" if car.get("in_pit") else ("🏁 FINISHED" if car.get("finished") else "🏎️ RACING")
            })

        st.dataframe(
            spec_rows,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Position": st.column_config.TextColumn("🏆 Pos"),
                "Driver": st.column_config.TextColumn("🏎️ Driver & Team"),
                "Speed": st.column_config.TextColumn("⚡ Speed"),
                "Tyre": st.column_config.TextColumn("🛞 Compound"),
                "Grip Status": st.column_config.TextColumn("🛡️ Tyre Condition"),
                "Tyre Wear (%)": st.column_config.ProgressColumn("📉 Tyre Wear %", min_value=0, max_value=100, format="%.1f%%"),
                "Gap to P1": st.column_config.TextColumn("⏱️ Gap to Leader"),
                "DRS Wing": st.column_config.TextColumn("💨 DRS Status"),
                "Status": st.column_config.TextColumn("🚦 Status")
            }
        )

    # =========================================================================
    # TAB 2: DETAILED GRAPHICAL TELEMETRY
    # =========================================================================
    with tab_graphs:
        st.info("💡 **In Plain English**: These graphs trace how fast cars accelerate on straights, how hard they brake into corners (heating carbon brakes up to 800°C), and how lap times change as tyres wear down.")
        st.caption("💡 *Live circuit tracking and real-time telemetry analytics. Hover over any car marker or chart for exact figures.*")

        # --- 2D Real-Time Circuit Map & Live Battle Visualizer ---
        try:
            import f1_circuit_maps
            df_map_cars = f1_circuit_maps.get_live_car_coordinates(
                track_name=current_track_info["name"],
                state=state,
                track_length_m=current_track_info["length_m"]
            )
            track_chart = f1_circuit_maps.render_circuit_map_altair(
                track_name=current_track_info["name"],
                df_cars=df_map_cars
            )
            st.altair_chart(track_chart, use_container_width=True)
        except Exception as e:
            pass

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
        st.info("💡 **In Plain English**: This is the complete race engineering sheet. It gives you the exact numbers for car speeds, lap times, sector splits, brake disc temperatures, and tyre wear percentages.")
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

    with tab_dim_red:
        st.info("💡 **In Plain English**: A modern F1 car has dozens of sensors measuring speed, throttle, brakes, fuel, and tyre wear every millisecond. Looking at 10+ numbers at the same time is overwhelming. PCA and t-SNE squash all 10 confusing numbers down into a simple 2D map so race engineers can spot tyre trends and driver styles at a single glance!")
        st.caption("🧭 *High-Dimensional Telemetry Compression & Manifold Learning: Uncovering driver habits, tyre regimes, and vehicle dynamics using PCA and t-SNE.*")

        if not check_ml_available():
            st.warning("Dimensionality reduction features require `pandas`, `numpy`, and `scikit-learn`. Please ensure they are installed in your Python environment.")
        else:
            import importlib
            import f1_dimensionality_reduction
            importlib.reload(f1_dimensionality_reduction)
            from f1_dimensionality_reduction import (
                extract_telemetry_samples,
                compute_pca,
                compute_tsne,
                FEATURE_COLUMNS,
                FEATURE_DISPLAY_NAMES
            )

            df_dim_samples = extract_telemetry_samples(state, include_history=True)

            sub_pca, sub_tsne, sub_compare, sub_guide = st.tabs([
                "📊 Principal Component Analysis (PCA)",
                "🌀 t-SNE Manifold & Clusters",
                "⚖️ Side-by-Side Face-Off",
                "📖 Beginner & Intuition Guide"
            ])

            pca_res = compute_pca(df_dim_samples, n_components=4)

            # --- SUB-TAB 1: PCA ---
            with sub_pca:
                if pca_res is not None:
                    # 4 Top KPI Cards
                    k1, k2, k3, k4 = st.columns(4)
                    var_pc1 = pca_res.explained_variance_ratio[0] * 100.0
                    var_pc2 = pca_res.explained_variance_ratio[1] * 100.0
                    k1.metric("🎯 PC1 Variance", f"{var_pc1:.1f}%", help="Variance explained by Principal Component 1")
                    k2.metric("🎯 PC2 Variance", f"{var_pc2:.1f}%", help="Variance explained by Principal Component 2")
                    k3.metric("📈 2D Information Preserved", f"{(var_pc1 + var_pc2):.1f}%", help="Total telemetry information retained in just 2 axes")
                    k4.metric("🗜️ Dimension Compression", f"{len(FEATURE_COLUMNS)} Sensors ➔ 2 PCs", help="Multi-sensor reduction into 2 orthogonal axes")

                    st.info(f"💡 **Information Barometer**: {pca_res.get_variance_summary_text()}")

                    col_p1, col_p2 = st.columns([1.3, 1.0])

                    with col_p1:
                        st.subheader("🎯 2D PCA Telemetry Projection")
                        pca_color_choice = st.selectbox(
                            "🎨 Color Data Points By:",
                            options=["🏎️ Driver", "🛞 Tyre Compound", "📍 Sample Type"],
                            index=0,
                            key="pca_color_selector"
                        )
                        color_field_map = {
                            "🏎️ Driver": "driver",
                            "🛞 Tyre Compound": "tyre_compound",
                            "📍 Sample Type": "sample_type"
                        }
                        target_color = color_field_map[pca_color_choice]

                        df_pca_plot = pca_res.transformed_df.copy()

                        if hasattr(st, "scatter_chart"):
                            st.scatter_chart(
                                df_pca_plot,
                                x="PC1",
                                y="PC2",
                                color=target_color,
                                use_container_width=True
                            )
                        else:
                            st.dataframe(df_pca_plot[["driver", "tyre_compound", "PC1", "PC2", "speed", "brake_temp", "tyre_wear"]])

                        st.markdown(f"""
                        **🔍 Axis Interpretations (Linear Orthogonal Loadings):**
                        * **Horizontal Axis (PC1)**: `{pca_res.interpret_component(0)}`
                        * **Vertical Axis (PC2)**: `{pca_res.interpret_component(1)}`
                        """)

                    with col_p2:
                        st.subheader("📊 Scree Plot (Explained Variance)")
                        scree_data = pd.DataFrame({
                            "Principal Component": [f"PC{i+1}" for i in range(pca_res.n_components)],
                            "Explained Variance (%)": pca_res.explained_variance_ratio * 100.0
                        }).set_index("Principal Component")
                        st.bar_chart(scree_data)

                        with st.expander("🔬 Sensor Loading Matrix (Biplot Weights)", expanded=False):
                            st.caption("Shows how strongly each original sensor impacts PC1 and PC2. Higher absolute numbers indicate dominant influence.")
                            formatted_loadings = pca_res.loadings_df.copy()
                            formatted_loadings.index = [FEATURE_DISPLAY_NAMES.get(idx, idx) for idx in formatted_loadings.index]
                            st.dataframe(formatted_loadings.round(3), use_container_width=True)
                else:
                    st.info("Collecting sufficient telemetry ticks to compute PCA...")

            # --- SUB-TAB 2: t-SNE ---
            with sub_tsne:
                st.subheader("🌀 Non-Linear Manifold Embedding (t-SNE)")
                st.caption("t-SNE preserves local neighborhoods, mapping multi-sensor car behavior into natural driving style and tyre wear clusters.")

                tsne_c1, tsne_c2, tsne_c3 = st.columns([1, 1, 1])
                with tsne_c1:
                    user_perp = st.slider("🔍 Perplexity (Effective Neighbors)", min_value=5, max_value=40, value=15, step=5, help="Controls balance between local details and global race clusters")
                with tsne_c2:
                    user_iter = st.slider("🔄 Optimization Steps", min_value=250, max_value=1000, value=500, step=50, help="Gradient descent iterations for embedding convergence")
                with tsne_c3:
                    tsne_color_choice = st.selectbox(
                        "🎨 Color Clusters By:",
                        options=["🏎️ Driver", "🛞 Tyre Compound"],
                        index=0,
                        key="tsne_color_selector"
                    )

                target_tsne_color = "driver" if "Driver" in tsne_color_choice else "tyre_compound"

                col_btn, col_info = st.columns([1.2, 2.8])
                with col_btn:
                    run_tsne_btn = st.button("🌀 Run / Update t-SNE Manifold", type="primary", key="btn_compute_tsne")
                with col_info:
                    st.caption("💡 *t-SNE runs an iterative optimization. Click the button to re-cluster the latest telemetry snapshots.*")

                if run_tsne_btn or "cached_tsne_res" not in st.session_state:
                    tsne_res = compute_tsne(df_dim_samples, perplexity=float(user_perp), n_iter=int(user_iter))
                    if tsne_res is not None:
                        st.session_state["cached_tsne_res"] = tsne_res
                else:
                    tsne_res = st.session_state.get("cached_tsne_res")

                if tsne_res is not None:
                    col_t1, col_t2 = st.columns([1.4, 1.0])
                    with col_t1:
                        if hasattr(st, "scatter_chart"):
                            st.scatter_chart(
                                tsne_res.embedded_df,
                                x="tSNE1",
                                y="tSNE2",
                                color=target_tsne_color,
                                use_container_width=True
                            )
                        else:
                            st.dataframe(tsne_res.embedded_df[["driver", "tyre_compound", "tSNE1", "tSNE2"]])

                    with col_t2:
                        st.markdown(f"""
                        #### 🏝️ Key Discovered Manifolds:
                        * **Driver Style Clustering**: Notice how high-aggression drivers (`VER`, `ALO`) form distinct local neighborhoods compared to smoother drivers.
                        * **Tyre Friction Islands**: Soft compounds separate from Hard compounds due to distinct traction and braking profiles.
                        * **KL Divergence**: `{tsne_res.kl_divergence:.3f}` *(Measures how accurately 2D probabilities match original high-D affinities)*.
                        """)
                        with st.expander("⚠️ Understanding The Crowding Problem", expanded=False):
                            st.markdown("""
                            In 8-dimensional sensor space, there is vast volume for data points to spread out. When forced into a flat 2D canvas, points tend to collapse on top of each other.

                            **The Solution**: t-SNE uses a heavy-tailed **Student-t distribution** in 2D. This pushes moderately distant clusters further apart while keeping close neighbors tightly bound!
                            """)
                else:
                    st.info("Collecting sufficient telemetry points to compute t-SNE embedding...")

            # --- SUB-TAB 3: COMPARISON ---
            with sub_compare:
                st.subheader("⚖️ Side-by-Side: PCA vs. t-SNE Face-Off")
                st.caption("Visual comparison of the exact same telemetry samples projected linearly (PCA) vs non-linearly (t-SNE).")

                cmp_c1, cmp_c2 = st.columns(2)
                with cmp_c1:
                    st.markdown("#### 📏 PCA: Linear Global Structure")
                    if pca_res is not None and hasattr(st, "scatter_chart"):
                        st.scatter_chart(pca_res.transformed_df, x="PC1", y="PC2", color="driver", use_container_width=True)
                    st.caption("Preserves largest variances. Distance between distant points is mathematically meaningful.")

                with cmp_c2:
                    st.markdown("#### 🏝️ t-SNE: Non-linear Local Clusters")
                    if tsne_res is not None and hasattr(st, "scatter_chart"):
                        st.scatter_chart(tsne_res.embedded_df, x="tSNE1", y="tSNE2", color="driver", use_container_width=True)
                    st.caption("Preserves neighbor similarities. Discovers clusters that linear methods miss.")

                st.markdown("---")
                st.markdown(r"""
                | Comparison Metric | 📏 Principal Component Analysis (PCA) | 🌀 t-SNE (Manifold Learning) |
                | :--- | :--- | :--- |
                | **Method Type** | Linear, orthogonal projection | Non-linear, probabilistic embedding |
                | **What it Preserves** | Global variance (extreme values & spread) | Local neighborhoods (who is similar to whom) |
                | **Axis Interpretability** | **High**: Exact sensor weights (loadings) for PC1 & PC2 | **None**: Coordinate axes have arbitrary units |
                | **Speed / Computation** | **Instant** ($O(D^2 N)$ via matrix decomposition) | **Iterative** ($O(N \log N)$ via Barnes-Hut) |
                | **Project New Cars?** | **Yes**: Instant multiplication $X_{new} \cdot W$ | **No**: Requires re-fitting the embedding |
                | **Primary F1 Use Case** | Compressing 10+ sensors into master energy/thermal dials | Discovering driver styles, tyre degradation phases & anomalies |
                """)

            # --- SUB-TAB 4: BEGINNER GUIDE ---
            with sub_guide:
                st.subheader("📖 Beginner's Intuition Guide: Dimensionality Reduction")

                with st.expander("💡 1. The Shadow Analogy (Understanding PCA)", expanded=True):
                    st.markdown("""
                    Imagine holding a 3D toy car in a dark room. If you shine a flashlight directly from the side, its 2D shadow on the wall clearly outlines the wheels, wing, and cockpit.

                    If you shined the light from the front, the shadow would just look like a small rectangle—losing most details.

                    **PCA is like finding the exact flashlight angle that casts the largest, most informative 2D shadow of your multi-dimensional telemetry data.**
                    """)

                with st.expander("🏝️ 2. The Cocktail Party Analogy (Understanding t-SNE)", expanded=False):
                    st.markdown("""
                    Imagine a large room with 100 people talking about Formula 1, aerodynamics, tyre chemistry, and engine tuning.

                    **t-SNE is like arranging everyone at small cocktail tables**: people who talk about tyre wear sit together at Table A, while engine engineers sit at Table B.

                    t-SNE makes sure that friends stay close together, without obsessing over the exact distance between Table A and Table B across the room.
                    """)

                with st.expander("🏎️ 3. Why Formula 1 Engineers Rely on Both", expanded=False):
                    st.markdown("""
                    * **Race Strategy**: PCA reduces dozens of engine, thermal, and tyre sensors into a single **"Degradation Index"** that can be tracked in real-time on the pit wall.
                    * **Driver Coaching**: t-SNE clusters corner-entry speeds, braking points, and throttle aggression to show drivers how their technique differs from their teammate.
                    """)

            # Telemetry Export Button
            if pca_res is not None and tsne_res is not None:
                st.write("---")
                df_export_dim = pca_res.transformed_df.copy()
                df_export_dim["tSNE1"] = tsne_res.embedded_df["tSNE1"]
                df_export_dim["tSNE2"] = tsne_res.embedded_df["tSNE2"]
                csv_dim_bytes = df_export_dim.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download Dimensionality Reduction Dataset (CSV)",
                    csv_dim_bytes,
                    file_name=f"f1_dim_reduction_{current_track_info['name'].lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    help="Exports raw telemetry sensors alongside PC1, PC2, tSNE1, and tSNE2 coordinates"
                )

    with tab_ensemble:
        st.info("💡 **In Plain English**: Instead of asking just one computer model for lap-time predictions, Ensemble Learning asks a council of 100 decision trees to vote together! This combines the wisdom of multiple models (like Random Forest and XGBoost) to give much more reliable race predictions.")
        st.caption("🏎️ *Ensemble Learning & Telemetry Diagnostics: Bagging (Random Forest), Boosting (XGBoost & LightGBM), Bias–Variance Decomposition, and Regularization (L1/L2).*")
        
        if not check_ml_available():
            st.warning("Ensemble analytics require `pandas`, `numpy`, and `scikit-learn`. Please ensure they are installed in your Python environment.")
        else:
            import importlib
            import f1_ensemble_analytics
            importlib.reload(f1_ensemble_analytics)
            from f1_ensemble_analytics import (
                generate_telemetry_regression_dataset,
                extract_live_telemetry_for_ensemble,
                benchmark_ensemble_models,
                compute_bias_variance_tradeoff,
                analyze_overfitting_underfitting,
                analyze_regularization_impact,
                extract_ensemble_feature_importances,
                predict_what_if_strategy,
                STUDENT_STUDY_GUIDE,
                TELEMETRY_FEATURES
            )

            # Data Source Toggle
            col_ens_ctrl1, col_ens_ctrl2 = st.columns([1.5, 1.0])
            with col_ens_ctrl1:
                ensemble_data_mode = st.radio(
                    "📊 Telemetry Dataset Source:",
                    options=["⚡ Multi-Lap Synthetic Telemetry (1,000 Samples)", "🔴 Live Race Session Telemetry"],
                    horizontal=True,
                    key="ensemble_data_mode_choice"
                )
            with col_ens_ctrl2:
                noise_slider = st.slider("🧪 Telemetry Sensor Noise Level (σ)", min_value=0.2, max_value=2.5, value=0.8, step=0.1, key="ens_noise_slider")

            if "Live" in ensemble_data_mode:
                df_ensemble_base = extract_live_telemetry_for_ensemble(state)
            else:
                df_ensemble_base = generate_telemetry_regression_dataset(n_samples=1000, noise_level=noise_slider)

            sub_arena, sub_importance, sub_whatif, sub_bv, sub_overfit, sub_guide = st.tabs([
                "🏆 Ensemble Arena (Bagging vs. Boosting)",
                "📊 Feature Importance & Explainability",
                "🔮 Pitwall 'What-If' Strategy Calculator",
                "⚖️ Bias–Variance Trade-Off Lab",
                "🛡️ Overfitting & Regularization Playground",
                "📚 Student Study Guide & Theory Reference"
            ])

            # --- SUB-TAB 1: ENSEMBLE ARENA ---
            with sub_arena:
                st.subheader("🏁 Head-to-Head Benchmark: Bagging vs. Boosting")
                st.write("Evaluating **Single Decision Tree** (Baseline) against **Bagging Regressor**, **Random Forest** (Bagging), **XGBoost** (Gradient Boosting), and **LightGBM** (Histogram Gradient Boosting) on F1 telemetry lap pace prediction.")

                if st.button("🚀 Run Live Ensemble Benchmark", key="btn_run_ensemble_bench"):
                    with st.spinner("Training and benchmarking ensemble models..."):
                        bench_df, fitted_models = benchmark_ensemble_models(df_ensemble_base)
                        st.session_state["ens_bench_df"] = bench_df
                        st.session_state["ens_fitted_models"] = fitted_models

                if "ens_bench_df" in st.session_state:
                    bench_df = st.session_state["ens_bench_df"]
                    
                    # Top Metric Summary Cards
                    top_model = bench_df.iloc[0]
                    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                    kpi1.metric("🥇 Champion Model", top_model["Model Name"].split(" (")[0], help="Highest R² on unseen telemetry holdout")
                    kpi2.metric("🎯 Top R² Score", f"{top_model['R² Score']:.4f}", help="Variance of lap time explained")
                    kpi3.metric("⏱️ Lowest MAE", f"{top_model['MAE (seconds)']:.3f} s", help="Mean Absolute Error in seconds per lap")
                    kpi4.metric("⚡ Fastest Training", f"{bench_df['Train Latency (ms)'].min():.1f} ms", help="Lowest fit time across models")

                    st.dataframe(
                        bench_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Model Name": st.column_config.TextColumn("🤖 Algorithm"),
                            "Paradigm": st.column_config.TextColumn("🏷️ Ensemble Class"),
                            "Mechanism": st.column_config.TextColumn("⚙️ Core Mechanism"),
                            "R² Score": st.column_config.NumberColumn("🎯 R² Score", format="%.4f"),
                            "MAE (seconds)": st.column_config.NumberColumn("⏱️ MAE (s)", format="%.3f"),
                            "RMSE (seconds)": st.column_config.NumberColumn("📉 RMSE (s)", format="%.3f"),
                            "Train Latency (ms)": st.column_config.NumberColumn("⏱️ Train Time (ms)", format="%.1f"),
                            "Predict Latency (ms)": st.column_config.NumberColumn("⚡ Predict Time (ms)", format="%.2f"),
                        }
                    )

                    c_ch1, c_ch2 = st.columns(2)
                    with c_ch1:
                        st.subheader("🎯 Model Accuracy Comparison (R² Score)")
                        chart_df_r2 = bench_df[["Model Name", "R² Score"]].set_index("Model Name")
                        st.bar_chart(chart_df_r2)

                    with c_ch2:
                        st.subheader("⏱️ Lap Error Comparison (MAE in Seconds)")
                        chart_df_mae = bench_df[["Model Name", "MAE (seconds)"]].set_index("Model Name")
                        st.bar_chart(chart_df_mae)

                    st.info("""
                    💡 **Engineering Insight**: 
                    * **Bagging (Random Forest)** trains independent trees in parallel on bootstrapped telemetry samples, drastically reducing model variance.
                    * **Boosting (XGBoost / LightGBM)** builds sequential shallow trees, with each round fitting the gradient residuals of the previous model, effectively driving bias down while regularization keeps variance in check.
                    """)
                else:
                    st.info("👆 Click **'Run Live Ensemble Benchmark'** to fit all models and compare their metrics.")

            # --- SUB-TAB 2: FEATURE IMPORTANCE & EXPLAINABILITY ---
            with sub_importance:
                st.subheader("📊 Ensemble Feature Importance & Model Explainability")
                st.markdown("Understand how **Bagging** (Mean Decrease in Impurity / Gini) and **Boosting** (Gain / Split Frequency) attribute predictive weight across telemetry sensors.")

                # Ensure models are fitted
                if "ens_fitted_models" not in st.session_state:
                    with st.spinner("Training baseline ensemble models for feature extraction..."):
                        bench_df, fitted_models = benchmark_ensemble_models(df_ensemble_base)
                        st.session_state["ens_bench_df"] = bench_df
                        st.session_state["ens_fitted_models"] = fitted_models

                df_importances = extract_ensemble_feature_importances(st.session_state["ens_fitted_models"])
                
                # Chart
                c_fi1, c_fi2 = st.columns([1.4, 1.0])
                with c_fi1:
                    st.subheader("🎯 Comparative Feature Importance (%)")
                    chart_fi = df_importances.set_index("Feature")
                    st.bar_chart(chart_fi)

                with c_fi2:
                    st.subheader("📋 Sensor Weight Breakdown")
                    st.dataframe(df_importances, use_container_width=True, hide_index=True)

                st.info("""
                💡 **Key Model Interpretability Takeaway**:
                * **Speed & Tyre Degradation** dominate lap time variance in Formula 1 vehicle dynamics, accounting for over **65%** of predictive splits across both Random Forest and XGBoost.
                * **Brake Disc Temperatures** activate non-linear penalties at high thermal thresholds (>750°C), which gradient boosters capture effectively through deep residual splits.
                """)

            # --- SUB-TAB 3: PITWALL 'WHAT-IF' STRATEGY CALCULATOR ---
            with sub_whatif:
                st.subheader("🔮 Pitwall 'What-If' Strategy Calculator")
                st.markdown("Simulate hypothetical race scenarios: Adjust tyre wear, fuel load, aggression, and compound to inspect instantaneous model predictions and strategic pit recommendations.")

                if "ens_fitted_models" not in st.session_state:
                    with st.spinner("Training baseline ensemble models for prediction engine..."):
                        bench_df, fitted_models = benchmark_ensemble_models(df_ensemble_base)
                        st.session_state["ens_bench_df"] = bench_df
                        st.session_state["ens_fitted_models"] = fitted_models

                w_c1, w_c2, w_c3 = st.columns(3)
                with w_c1:
                    what_compound = st.selectbox("🛞 Tyre Compound:", ["Soft", "Medium", "Hard", "Inter", "Wet"], index=1, key="what_comp")
                    what_wear = st.slider("📉 Current Tyre Wear (%):", min_value=5.0, max_value=95.0, value=35.0, step=1.0, key="what_wear")
                with w_c2:
                    what_fuel = st.slider("⛽ Remaining Fuel Tank (%):", min_value=5.0, max_value=100.0, value=50.0, step=1.0, key="what_fuel")
                    what_aggression = st.slider("🔥 Driver Aggression Multiplier:", min_value=0.80, max_value=1.30, value=1.05, step=0.01, key="what_aggr")
                with w_c3:
                    what_speed = st.slider("⚡ Cornering Speed (km/h):", min_value=160.0, max_value=340.0, value=255.0, step=5.0, key="what_speed")
                    what_brake_temp = st.slider("🌡️ Brake Disc Heat (°C):", min_value=300.0, max_value=900.0, value=520.0, step=10.0, key="what_brake")

                # Compute prediction
                current_track_base_lap = current_track_info.get("base_lap", 85.0)
                what_res = predict_what_if_strategy(
                    fitted_models=st.session_state["ens_fitted_models"],
                    speed=what_speed,
                    throttle=85.0,
                    brake=15.0,
                    tyre_wear=what_wear,
                    fuel=what_fuel,
                    brake_temp=what_brake_temp,
                    aggression=what_aggression,
                    compound=what_compound,
                    baseline_lap=current_track_base_lap
                )

                # Strategic Recommendation Alert
                if what_res["rec_color"] == "error":
                    st.error(f"### {what_res['recommendation']}")
                elif what_res["rec_color"] == "warning":
                    st.warning(f"### {what_res['recommendation']}")
                else:
                    st.success(f"### {what_res['recommendation']}")

                p_kpi1, p_kpi2, p_kpi3 = st.columns(3)
                p_kpi1.metric("⏱️ Ensemble Consensus Pace", f"{what_res['avg_predicted_pace']:.3f} s", help="Mean lap time prediction across fitted models")
                p_kpi2.metric("📊 Pace Delta vs Circuit Base", f"{what_res['pace_delta']:+.3f} s", delta=f"{what_res['pace_delta']:+.3f} s", delta_color="inverse")
                p_kpi3.metric("🏎️ Circuit Baseline", f"{current_track_base_lap:.1f} s", help=f"Optimal target pace at {current_track_info['name']}")

                st.write("#### 🤖 Predictions by Individual Model:")
                pred_rows = [{"Model": k, "Predicted Lap Time": f"{v:.3f} s", "Delta vs Base": f"{v - current_track_base_lap:+.3f} s"} for k, v in what_res["predictions"].items()]
                st.dataframe(pred_rows, use_container_width=True, hide_index=True)

            # --- SUB-TAB 4: BIAS-VARIANCE TRADEOFF LAB ---
            with sub_bv:
                st.subheader("⚖️ Bias–Variance Decomposition & Error Dynamics")
                st.markdown(r"Total Error decomposes as: $\text{Error} = \text{Bias}^2 + \text{Variance} + \text{Noise}$. See how tree depth shifts the balance from underfitting to overfitting.")

                col_bv1, col_bv2 = st.columns([1.5, 1.0])
                with col_bv1:
                    bv_model_family = st.selectbox(
                        "🔍 Select Model Architecture:",
                        options=["Bagging (Random Forest)", "Boosting (XGBoost)", "Boosting (LightGBM)", "Single Decision Tree"],
                        index=0,
                        key="bv_model_family_selector"
                    )
                with col_bv2:
                    bootstraps_count = st.slider("🔄 Bootstrap Iterations", min_value=5, max_value=20, value=10, step=1, key="bv_bootstraps")

                if st.button("📈 Compute Bias-Variance Decomposition", key="btn_run_bv"):
                    with st.spinner("Executing bootstrap re-sampling decomposition..."):
                        df_bv = compute_bias_variance_tradeoff(
                            df_ensemble_base,
                            model_family=bv_model_family,
                            depth_range=[1, 2, 3, 4, 6, 8, 11, 15],
                            n_bootstraps=bootstraps_count
                        )
                        st.session_state["df_bv"] = df_bv

                if "df_bv" in st.session_state:
                    df_bv = st.session_state["df_bv"]
                    
                    st.subheader("📊 Bias² vs. Variance vs. Test MSE Across Tree Depth")
                    bv_chart_data = df_bv.set_index("Tree Depth")[["Test MSE", "Train MSE", "Bias² (Underfitting Error)", "Variance (Overfitting Error)"]]
                    st.line_chart(bv_chart_data)

                    best_depth_row = df_bv.loc[df_bv["Test MSE"].idxmin()]
                    b1, b2, b3, b4 = st.columns(4)
                    b1.metric("🎯 Optimal Depth", f"Depth {int(best_depth_row['Tree Depth'])}")
                    b2.metric("📉 Min Test MSE", f"{best_depth_row['Test MSE']:.3f}")
                    b3.metric("⚖️ Bias² at Optimum", f"{best_depth_row['Bias² (Underfitting Error)']:.3f}")
                    b4.metric("🛡️ Variance at Optimum", f"{best_depth_row['Variance (Overfitting Error)']:.3f}")

                    st.dataframe(df_bv, use_container_width=True, hide_index=True)

                    st.success(f"""
                    **Key Observation for {bv_model_family}**:
                    * At **low depths (1–3)**: $\\text{{Bias}}^2$ dominates — the model is too rigid (**Underfitting / High Bias**).
                    * At **optimal depth ({int(best_depth_row['Tree Depth'])})**: $\\text{{Bias}}^2 + \\text{{Variance}}$ is minimized, delivering peak generalization on unseen laps.
                    * At **high depths (10+)**: $\\text{{Variance}}$ climbs while training MSE approaches zero — the model memorizes sensor noise (**Overfitting / High Variance**).
                    """)
                else:
                    st.info("👆 Click **'Compute Bias-Variance Decomposition'** to analyze the tradeoff across depth levels.")

            # --- SUB-TAB 3: OVERFITTING & REGULARIZATION PLAYGROUND ---
            with sub_overfit:
                st.subheader("🛡️ Overfitting, Underfitting & Regularization Simulator")
                st.markdown(r"Interactively adjust tree depth, estimators, learning rate, and L1 / L2 penalties to inspect the **Generalization Gap** ($\Delta R^2$).")

                c_ctrl1, c_ctrl2, c_ctrl3 = st.columns(3)
                with c_ctrl1:
                    sel_algo = st.selectbox("🤖 Model Algorithm:", options=["XGBoost", "LightGBM", "Random Forest", "Decision Tree"], key="diag_algo")
                    sel_depth = st.slider("🌳 Max Tree Depth:", min_value=1, max_value=20, value=8, step=1, key="diag_depth")
                with c_ctrl2:
                    sel_est = st.slider("🌲 Number of Estimators:", min_value=10, max_value=120, value=50, step=10, key="diag_est") if sel_algo != "Decision Tree" else 1
                    sel_lr = st.slider("⚡ Learning Rate (η):", min_value=0.01, max_value=0.40, value=0.10, step=0.01, key="diag_lr") if "Decision" not in sel_algo and "Forest" not in sel_algo else 0.1
                with c_ctrl3:
                    sel_l1 = st.slider("🎯 L1 Penalty (reg_alpha / Lasso):", min_value=0.0, max_value=10.0, value=0.1, step=0.1, key="diag_l1") if "Decision" not in sel_algo and "Forest" not in sel_algo else 0.0
                    sel_l2 = st.slider("🛡️ L2 Penalty (reg_lambda / Ridge):", min_value=0.0, max_value=20.0, value=1.0, step=0.5, key="diag_l2") if "Decision" not in sel_algo and "Forest" not in sel_algo else 0.0

                diag_res = analyze_overfitting_underfitting(
                    df_ensemble_base,
                    max_depth=sel_depth,
                    n_estimators=sel_est,
                    learning_rate=sel_lr,
                    reg_alpha=sel_l1,
                    reg_lambda=sel_l2,
                    model_type=sel_algo
                )

                # Diagnostic Banner
                if diag_res["diagnosis_color"] == "success":
                    st.success(f"### {diag_res['diagnosis']}\n{diag_res['explanation']}")
                elif diag_res["diagnosis_color"] == "warning":
                    st.warning(f"### {diag_res['diagnosis']}\n{diag_res['explanation']}")
                else:
                    st.error(f"### {diag_res['diagnosis']}\n{diag_res['explanation']}")

                m_c1, m_c2, m_c3, m_c4 = st.columns(4)
                m_c1.metric("🎯 Train R²", f"{diag_res['train_r2']:.4f}")
                m_c2.metric("🧪 Test R²", f"{diag_res['test_r2']:.4f}")
                m_c3.metric("📏 Generalization Gap (Δ R²)", f"{diag_res['gen_gap_r2']:.4f}", help="Train R² minus Test R² (Large gap = Overfitting)")
                m_c4.metric("📉 Test MSE", f"{diag_res['test_mse']:.3f} s²")

                st.write("---")
                st.subheader("🔬 Regularization Sweep: Taming High Variance")
                st.caption("Inspect how sweeping L1 (Lasso) or L2 (Ridge) penalty dampens the overfitting gap on deep trees.")

                sweep_col1, sweep_col2 = st.columns([1.5, 1.0])
                with sweep_col1:
                    reg_type_sweep = st.radio("Select Regularization Type to Sweep:", ["L2 (Ridge / reg_lambda)", "L1 (Lasso / reg_alpha)"], horizontal=True, key="reg_sweep_type")
                with sweep_col2:
                    run_sweep_btn = st.button("🧪 Run Regularization Sweep", key="btn_run_reg_sweep")

                if run_sweep_btn:
                    with st.spinner("Sweeping regularization penalties..."):
                        df_reg_sweep = analyze_regularization_impact(
                            df_ensemble_base,
                            reg_type="L1 (Lasso / reg_alpha)" if "L1" in reg_type_sweep else "L2 (Ridge)",
                            penalty_values=[0.0, 0.05, 0.2, 1.0, 5.0, 15.0, 40.0, 80.0]
                        )
                        st.session_state["df_reg_sweep"] = df_reg_sweep

                if "df_reg_sweep" in st.session_state:
                    df_reg_sweep = st.session_state["df_reg_sweep"]
                    st.line_chart(df_reg_sweep.set_index("Penalty Strength (λ / α)")[["Train R²", "Test R²", "Generalization Gap (Δ R²)"]])
                    st.dataframe(df_reg_sweep, use_container_width=True, hide_index=True)
                    st.info("💡 **Takeaway**: As penalty strength increases from 0.0 upwards, the Generalization Gap narrows, preventing the tree booster from fitting noise.")

            # --- SUB-TAB 4: STUDENT STUDY GUIDE & NOTES ---
            with sub_guide:
                st.subheader("📚 F1 Telemetry Machine Learning Student Master Guide")
                st.markdown("Curated reference notes, formulas, and interview-ready concepts covering Ensemble Methods, Bias-Variance, Overfitting, and Regularization.")

                with st.expander("🏁 1. Ensemble Methods: Bagging vs. Boosting (Architecture & XGBoost / LightGBM)", expanded=True):
                    st.markdown(STUDENT_STUDY_GUIDE["bagging_vs_boosting"])

                with st.expander("⚖️ 2. Bias–Variance Decomposition & Error Trade-off", expanded=False):
                    st.markdown(STUDENT_STUDY_GUIDE["bias_variance"])

                with st.expander("🛡️ 3. Overfitting vs. Underfitting: Symptoms, Root Causes & Prevention", expanded=False):
                    st.markdown(STUDENT_STUDY_GUIDE["overfitting_remedies"])

                with st.expander("🎯 4. Regularization (L1 Lasso vs. L2 Ridge) in Tree Boosters", expanded=False):
                    st.markdown(STUDENT_STUDY_GUIDE["regularization_l1_l2"])

                # Downloadable Study Notes & Diagnostics Export
                st.write("---")
                col_exp1, col_exp2 = st.columns(2)
                with col_exp1:
                    md_notes = f"""# F1 Telemetry Machine Learning Study Guide
{STUDENT_STUDY_GUIDE['bagging_vs_boosting']}

{STUDENT_STUDY_GUIDE['bias_variance']}

{STUDENT_STUDY_GUIDE['overfitting_remedies']}

{STUDENT_STUDY_GUIDE['regularization_l1_l2']}
"""
                    st.download_button(
                        "📥 Download Student Study Notes (.md)",
                        md_notes.encode("utf-8"),
                        file_name=f"f1_ensemble_study_guide_{datetime.now().strftime('%Y%m%d')}.md",
                        mime="text/markdown",
                        help="Export the complete theory guide and formulas as Markdown"
                    )
                with col_exp2:
                    csv_ens_data = df_ensemble_base.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "📥 Download Telemetry Regression Dataset (CSV)",
                        csv_ens_data,
                        file_name=f"f1_ensemble_telemetry_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        help="Download the multi-channel F1 telemetry dataset used for ensemble benchmarking"
                    )

    with tab_neural_net:
        if not check_ml_available():
            st.warning("Neural network analytics require `pandas`, `numpy`, and `joblib`. Please ensure they are installed in your Python environment.")
        else:
            import f1_neural_network_lab
            import importlib
            importlib.reload(f1_neural_network_lab)
            f1_neural_network_lab.render_neural_network_lab(
                state=state,
                current_track_info=current_track_info,
                driver_labels=DRIVER_LABELS
            )

    # TAB 7: F1 BEGINNER'S RULES & STRATEGY GUIDE (NON-TECH FRIENDLY)
    # =========================================================================
    with tab_beginner:
        st.subheader("📖 Formula 1 Beginner's Rules, Physics & Pit Strategy Guide")
        st.markdown("Everything you need to understand Grand Prix racing, pit strategy, tyres, and physics in simple, everyday language!")

        b_c1, b_c2 = st.columns(2)
        with b_c1:
            st.markdown("""
            ### 🛞 1. The 5 Tyre Compounds (Why Rubber Matters)
            Formula 1 cars do not just have one tyre. Pirelli supplies 5 distinct tyre compounds:
            * 🔴 **Soft Tyres (Red)**: Very soft, sticky rubber. Gives **maximum grip** and the fastest lap times, but burns away quickly (lasts ~10–15 laps).
            * 🟡 **Medium Tyres (Yellow)**: The balanced tyre. Fast and durable (lasts ~20–25 laps). Most drivers start the race on Mediums.
            * ⚪ **Hard Tyres (White)**: Tough, dense rubber. Slower to heat up, but extremely durable (can last 35+ laps).
            * 🟢 **Intermediates (Green)**: Used in light rain or on a damp, drying track to channel water away.
            * 🔵 **Full Wet (Blue)**: Heavy monsoon rain tyre that pumps **85 litres of water per second** to prevent hydroplaning!

            ---

            ### 📉 2. What is the 'Tyre Cliff'?
            When an F1 car corners at 250+ km/h, friction strips away microscopic layers of rubber:
            * **0% – 35% Wear**: 🟢 *Fresh Grip* — car handles crisply and corners at top speed.
            * **35% – 60% Wear**: 🟡 *Used Grip* — lap times drop by ~0.5 to 1.0 second as tyre slips slightly.
            * **> 60% Wear**: 🔴 *The Tyre Cliff* — grip falls off a cliff! The car starts sliding violently, brakes lock up, and lap times drop by 2–4 seconds. **This is when the team radio yells: 'BOX, BOX, BOX!'**
            """)

        with b_c2:
            st.markdown("""
            ### 💨 3. What is DRS (Drag Reduction System)?
            Formula 1 cars produce huge wings that push the car down onto the road, but also create massive air resistance (drag):
            * On designated straights, if a car is within **1.0 second** of the car ahead, the driver can press a button.
            * A hydraulic flap pops open the **rear wing**.
            * This dumps aerodynamic drag and grants an instant **+10 to +15 km/h speed boost** to pass the car ahead!

            ---

            ### ⛽ 4. Why Does Fuel Burn Make Cars Faster?
            * At the race start, cars are loaded with ~110 kg of petrol.
            * Carrying that extra weight makes the car sluggish into corners (costing ~0.3 seconds per 10 kg).
            * As fuel burns lap after lap, the car gets lighter and nimbler, which is why **the fastest laps of the race often happen on the very last laps!**

            ---

            ### 🛑 5. Pit Stop Tactics: The 'Undercut'
            * **The Undercut**: If you are stuck behind a rival and can't pass them on track, you pit **one lap earlier** for fresh Soft tyres.
            * Your fresh tyres give you so much grip on your out-lap that when your rival pits on the next lap, **you fly past them as they exit the pit lane!**
            """)

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
