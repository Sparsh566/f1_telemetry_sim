"""
f1_circuit_maps.py
==================
Real-time 2D Circuit Track Map & Live Battle Visualizer for Formula 1 circuits.
Generates authentic closed-loop 2D racing lines and maps real-time car coordinates
for Monza, Silverstone, Spa, Monaco, Suzuka, Bahrain, and Interlagos.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import altair as alt

# -----------------------------------------------------------------------------
# 1. Authentic 2D Circuit Track Coordinates (Normalized 0 - 100)
# -----------------------------------------------------------------------------
def _generate_monza_track() -> np.ndarray:
    """Monza: High speed 'Temple of Speed' layout with Rettifilo, Lesmos, Ascari & Parabolica."""
    pts = [
        [15, 85], [45, 85], [52, 83], [54, 86],  # Start/Finish -> Rettifilo Chicane
        [62, 86], [75, 78],                      # Curva Grande
        [80, 70], [78, 68], [82, 65],            # Roggia Chicane
        [84, 55], [82, 45],                      # Lesmo 1 & 2
        [70, 32], [55, 25],                      # Serraglio & Curva del Vialone
        [48, 22], [52, 19], [46, 17],            # Variante Ascari Chicane
        [30, 20], [18, 30],                      # Back straight to Parabolica
        [10, 48], [10, 68], [15, 85]             # Curva Parabolica back to start
    ]
    return np.array(pts, dtype=float)

def _generate_spa_track() -> np.ndarray:
    """Spa-Francorchamps: Long Ardennes loop with La Source, Eau Rouge, Kemmel, Pouhon, Bus Stop."""
    pts = [
        [20, 25], [16, 28], [18, 35],            # La Source hairpin
        [24, 40], [28, 48], [32, 54],            # Eau Rouge & Raidillon climb
        [45, 68], [58, 80],                      # Kemmel Straight
        [63, 84], [67, 82], [65, 76],            # Les Combes & Malmedy
        [62, 65], [60, 56],                      # Bruxelles hairpin
        [50, 48], [42, 42],                      # Rivage & Speakers corner
        [36, 32], [32, 28],                      # Double-left Pouhon
        [38, 20], [48, 16],                      # Fagnes & Stavelot
        [62, 18], [75, 24],                      # Courbe Paul Frere & Blanchimont
        [70, 28], [72, 25], [20, 25]             # Bus Stop Chicane to start line
    ]
    return np.array(pts, dtype=float)

def _generate_monaco_track() -> np.ndarray:
    """Monaco: Harbor street loop with Sainte Dévote, Casino, Hairpin, Tunnel, Swimming Pool."""
    pts = [
        [28, 20], [38, 20], [42, 25],            # Pit straight -> Sainte Dévote
        [50, 42], [58, 55], [66, 68],            # Beau Rivage uphill & Massenet
        [72, 70], [68, 62],                      # Casino Square & Mirabeau Haute
        [62, 60], [58, 64], [55, 60],            # Fairmont Hairpin
        [52, 52], [58, 45],                      # Mirabeau Bas & Portier
        [72, 38], [82, 32],                      # Iconic Seafront Tunnel
        [80, 28], [76, 29],                      # Nouvelle Chicane
        [68, 26], [58, 24],                      # Tabac corner
        [52, 28], [46, 25],                      # Swimming Pool Chicane
        [36, 24], [32, 21], [28, 20]             # La Rascasse & Anthony Noghes
    ]
    return np.array(pts, dtype=float)

def _generate_silverstone_track() -> np.ndarray:
    """Silverstone: High-G sweeping complex with Abbey, Arena, Brooklands, Copse, Maggotts-Becketts."""
    pts = [
        [30, 20], [45, 20], [52, 24],            # Hamilton Straight -> Abbey
        [58, 30], [52, 38], [42, 40],            # Farm Curve, Village & The Loop
        [35, 45], [42, 55], [52, 65],            # Wellington Straight & Brooklands
        [60, 68], [68, 62],                      # Luffield & Woodcote
        [75, 52], [84, 45],                      # Copse corner
        [86, 38], [80, 32], [76, 35], [72, 30],  # Maggotts, Becketts & Chapel
        [62, 25], [52, 20],                      # Hangar Straight
        [42, 14], [35, 15], [30, 20]             # Stowe, Vale & Club corner
    ]
    return np.array(pts, dtype=float)

def _generate_suzuka_track() -> np.ndarray:
    """Suzuka: Legendary figure-8 layout with S-Curves, Degners, Hairpin, Spoon & 130R."""
    pts = [
        [35, 25], [50, 25], [60, 30],            # Main straight -> Turns 1 & 2
        [65, 40], [58, 48], [66, 56], [58, 64],  # Esses (S-Curves) & Dunlop
        [52, 70], [56, 74],                      # Degner 1 & 2
        [48, 70], [42, 62],                      # Under the crossover bridge
        [36, 54], [32, 56], [35, 62],            # Suzuka Hairpin
        [42, 72], [48, 82], [55, 86],            # 200R curve
        [65, 88], [72, 82], [68, 76],            # Spoon curve
        [55, 66], [45, 52],                      # Over the crossover bridge
        [36, 38], [28, 32],                      # Ultra-fast 130R
        [26, 27], [30, 26], [35, 25]             # Casio Triangle Chicane to start line
    ]
    return np.array(pts, dtype=float)

def _generate_bahrain_track() -> np.ndarray:
    """Bahrain: Sakhir desert grand prix with heavy braking zones and 3 long DRS straights."""
    pts = [
        [20, 20], [60, 20], [70, 24], [66, 28],  # Main straight -> Turn 1 & 2 hairpin
        [58, 34], [62, 42], [74, 52],            # Turns 3 & 4 DRS straight
        [78, 62], [72, 68],                      # Turn 5, 6, 7 fast chicane
        [60, 72], [52, 66],                      # Turns 8 hairpin
        [44, 60], [38, 54],                      # Turns 9 & 10 tricky lockup zone
        [48, 42], [55, 36],                      # Back straight to Turn 11 & 12
        [52, 28], [42, 26], [30, 23], [20, 20]   # Turns 13, 14 & final Turn 15
    ]
    return np.array(pts, dtype=float)

def _generate_interlagos_track() -> np.ndarray:
    """Interlagos: Undulating Brazilian bowl with Senna S, Curva do Sol, Junção."""
    pts = [
        [30, 25], [55, 25], [62, 22], [58, 16],  # Start straight -> Senna 'S' (Turn 1 & 2)
        [50, 14], [42, 18],                      # Curva do Sol
        [32, 24], [22, 34],                      # Reta Oposta straight
        [18, 44], [24, 52],                      # Descida do Lago
        [35, 58], [48, 62], [58, 65],            # Ferradura high-speed sweep
        [62, 58], [55, 52],                      # Curva do Laranjinha & Pinheirinho
        [48, 48], [44, 54],                      # Bico de Pato slow hairpin
        [40, 46], [32, 38],                      # Mergulho
        [26, 32], [28, 28], [30, 25]             # Junção & uphill climb to start line
    ]
    return np.array(pts, dtype=float)


TRACK_GENERATORS = {
    "monza": _generate_monza_track,
    "spa": _generate_spa_track,
    "monaco": _generate_monaco_track,
    "silverstone": _generate_silverstone_track,
    "suzuka": _generate_suzuka_track,
    "bahrain": _generate_bahrain_track,
    "interlagos": _generate_interlagos_track
}

def get_track_spline(track_name: str, num_samples: int = 350) -> np.ndarray:
    """
    Interpolates a smooth closed polygon spline for the requested circuit.
    """
    key = track_name.lower().split(" ")[0].split("-")[0]
    gen_func = TRACK_GENERATORS.get(key, _generate_monza_track)
    base_points = gen_func()
    
    # Parametric closed interpolation
    t_base = np.linspace(0, 1, len(base_points))
    t_smooth = np.linspace(0, 1, num_samples)
    
    x_smooth = np.interp(t_smooth, t_base, base_points[:, 0])
    y_smooth = np.interp(t_smooth, t_base, base_points[:, 1])
    
    # Smooth with rolling window
    w = 5
    x_smooth = np.convolve(np.pad(x_smooth, (w, w), mode='wrap'), np.ones(w)/w, mode='valid')[:num_samples]
    y_smooth = np.convolve(np.pad(y_smooth, (w, w), mode='wrap'), np.ones(w)/w, mode='valid')[:num_samples]
    
    return np.column_stack([x_smooth, y_smooth])


# Driver distinct 2025 F1 team broadcast colors
DRIVER_TEAM_COLORS = {
    "VER": "#3671C6",  # Red Bull Navy Blue
    "HAM": "#F91536",  # Ferrari Rosso Corsa (Hamilton 2025 Transfer)
    "LEC": "#F91536",  # Ferrari Rosso Corsa
    "NOR": "#FF8000",  # McLaren Papaya Orange
    "PIA": "#FF8000",  # McLaren Papaya Orange
    "RUS": "#6CD3BF",  # Mercedes Petronas Teal
    "ALO": "#229971",  # Aston Martin Racing Green
    "SAI": "#00A0DE"   # Williams Racing Heritage Blue (Sainz 2025 Transfer)
}


# -----------------------------------------------------------------------------
# 2. Real-Time Track Coordinate Interpolation
# -----------------------------------------------------------------------------
def get_live_car_coordinates(
    track_name: str,
    state: Dict[str, Dict[str, Any]],
    track_length_m: float
) -> pd.DataFrame:
    """
    Maps each car's instantaneous lap progress percentage to 2D (x, y) coordinates
    along the authentic circuit perimeter.
    """
    track_points = get_track_spline(track_name, num_samples=500)
    n_pts = len(track_points)
    
    rows = []
    for name, car in sorted(state.items(), key=lambda kv: kv[1]["position"]):
        lap_dist = float(car.get("lap_distance", 0.0))
        progress = (lap_dist / track_length_m) if track_length_m > 0 else 0.0
        progress = float(np.clip(progress, 0.0, 0.9999))
        
        idx = int(progress * n_pts) % n_pts
        x_pos, y_pos = track_points[idx]
        
        team_color = DRIVER_TEAM_COLORS.get(name, "#E10600")
        
        pos = car.get("position", 1)
        speed = round(float(car.get("speed", 250.0)), 1)
        tyre = car.get("tyre", "Medium")
        wear = round(float(car.get("tyre_wear", 0.2)) * 100.0, 1)
        
        rows.append({
            "Driver": name,
            "Label": f"{name} (P{pos})",
            "x": float(x_pos),
            "y": float(y_pos),
            "Position": f"P{pos}",
            "Pos_Num": pos,
            "Speed (km/h)": speed,
            "Tyre": tyre,
            "Wear (%)": wear,
            "Sector": f"S{car.get('sector', 1)}",
            "Color": team_color
        })
        
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# 3. Altair 2D Circuit Map Visualizer
# -----------------------------------------------------------------------------
def render_circuit_map_altair(
    track_name: str,
    df_cars: pd.DataFrame
) -> alt.Chart:
    """
    Builds an authentic 2D Formula 1 circuit map visualization with:
      - Circuit asphalt track boundary
      - Live driver position markers with official F1 team colors
      - Text labels with real-time race positions
    """
    track_spline = get_track_spline(track_name, num_samples=350)
    df_track = pd.DataFrame(track_spline, columns=["x", "y"])
    df_track["order"] = np.arange(len(df_track))
    
    # 1. Base Circuit Asphalt (dark sleek racing line)
    track_line = alt.Chart(df_track).mark_line(
        color="#8E9AA7",
        strokeWidth=5.5,
        opacity=0.85
    ).encode(
        x=alt.X("x:Q", scale=alt.Scale(domain=[0, 95]), axis=None),
        y=alt.Y("y:Q", scale=alt.Scale(domain=[5, 95]), axis=None),
        order="order:Q"
    )
    
    # 2. Track inner kerb guideline
    track_kerb = alt.Chart(df_track).mark_line(
        color="#E10600",
        strokeWidth=1.5,
        strokeDash=[4, 4],
        opacity=0.5
    ).encode(
        x="x:Q",
        y="y:Q",
        order="order:Q"
    )
    
    # 3. Start / Finish Line marker
    start_line_pt = df_track.iloc[[0]]
    start_marker = alt.Chart(start_line_pt).mark_point(
        shape="square",
        size=180,
        color="#FFFFFF",
        fill="#E10600",
        strokeWidth=2
    ).encode(
        x="x:Q",
        y="y:Q",
        tooltip=alt.value("🏁 Start / Finish Line")
    )
    
    # 4. Live Car Position Dots
    car_dots = alt.Chart(df_cars).mark_circle(
        size=260,
        stroke="#FFFFFF",
        strokeWidth=2.0
    ).encode(
        x="x:Q",
        y="y:Q",
        color=alt.Color("Color:N", scale=None),
        tooltip=[
            alt.Tooltip("Label:N", title="🏎️ Driver"),
            alt.Tooltip("Position:N", title="🏆 Position"),
            alt.Tooltip("Speed (km/h):Q", title="⚡ Velocity"),
            alt.Tooltip("Sector:N", title="⏱️ Sector"),
            alt.Tooltip("Tyre:N", title="🛞 Compound"),
            alt.Tooltip("Wear (%):Q", title="📉 Tyre Wear")
        ]
    )
    
    # 5. Live Car Driver Text Labels (VER, HAM, LEC)
    car_text = alt.Chart(df_cars).mark_text(
        dy=-14,
        fontSize=12,
        fontWeight="bold",
        color="#FFFFFF"
    ).encode(
        x="x:Q",
        y="y:Q",
        text="Label:N"
    )
    
    combined_chart = (track_line + track_kerb + start_marker + car_dots + car_text).properties(
        width="container",
        height=380,
        title=f"🏁 2D Real-Time Circuit Map — {track_name} (Live GPS Tracking)"
    ).configure_view(
        strokeOpacity=0
    )
    
    return combined_chart
