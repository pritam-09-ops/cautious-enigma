"""
duck_curve_simulation.py
========================
Full duck-curve simulation for the cautious-enigma repo.

Builds on the existing analyze_duck_curve() and predict_curtailment_strategy()
functions and adds:
  • Realistic Mumbai grid-demand model
  • GHI → PV power conversion (MW)
  • Multi-day & seasonal simulation
  • Ramp-rate and curtailment analysis
  • 6-panel publication-quality plot
  • Detailed terminal report

Usage
-----
    # from repo root
    python src/duck_curve_simulation.py

    # custom CSV (must have timestamp + ghi columns)
    python src/duck_curve_simulation.py --data data/sample_solar_data.csv

    # choose a specific date
    python src/duck_curve_simulation.py --date 2025-06-15

    # save plots to a file
    python src/duck_curve_simulation.py --save plots/duck_curve.png

Dependencies
------------
    pip install numpy pandas matplotlib scikit-learn
    (torch NOT required — this file is standalone)
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Allow running from repo root OR from src/
# ─────────────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent if _HERE.name == "src" else _HERE
sys.path.insert(0, str(_REPO / "src"))

try:
    from duck_curve_analysis import (
        analyze_duck_curve,
        predict_curtailment_strategy,
        print_duck_curve_summary,
        HIGH_RAMP_THRESHOLD,
        CURTAILMENT_RAMP_THRESHOLD,
    )
    from feature_engineering import (
        compute_extraterrestrial_radiation,
        compute_solar_zenith_angle,
        compute_clearness_index,
    )
    print("✓  Imported modules from repo/src/")
except ImportError:
    # ── Fallback: inline the minimal logic so the script is truly standalone ──
    print("⚠  src/ modules not found — using built-in fallback implementations.")
    HIGH_RAMP_THRESHOLD        = 150.0
    CURTAILMENT_RAMP_THRESHOLD = 100.0
    SOLAR_CONSTANT             = 1361.0

    def compute_extraterrestrial_radiation(doy):
        B  = 2 * np.pi * (doy - 1) / 365.0
        E0 = (1.000110 + 0.034221*np.cos(B) + 0.001280*np.sin(B)
              + 0.000719*np.cos(2*B) + 0.000077*np.sin(2*B))
        return SOLAR_CONSTANT * E0

    def compute_solar_zenith_angle(timestamps, latitude=19.076):
        lat_r = np.radians(latitude)
        ts    = pd.DatetimeIndex(timestamps)
        ha    = np.radians(15.0 * (ts.hour + ts.minute/60.0 - 12.0))
        decl  = np.radians(23.45 * np.sin(np.radians(360/365*(ts.dayofyear-81))))
        cz    = np.clip(np.sin(lat_r)*np.sin(decl)
                        + np.cos(lat_r)*np.cos(decl)*np.cos(ha), -1, 1)
        return np.degrees(np.arccos(cz))

    def compute_clearness_index(ghi, doy, zenith):
        et  = compute_extraterrestrial_radiation(doy)
        cz  = np.cos(np.radians(zenith))
        eth = et * np.maximum(cz, 0.0)
        return np.clip(np.where(eth > 10, ghi/eth, 0.0), 0, 1)

    def analyze_duck_curve(predictions, timestamps=None):
        arr        = np.asarray(predictions, float)
        ramp_rates = np.diff(arr, prepend=arr[0])
        mid        = len(arr) // 2
        stability  = max(0.0, 100.0 - (np.max(np.abs(ramp_rates))/HIGH_RAMP_THRESHOLD)*30)
        stress     = "LOW" if stability>=80 else ("MODERATE" if stability>=60 else "HIGH")
        return dict(
            morning_ramp_rate = float(ramp_rates[:mid].max()),
            evening_ramp_rate = float(ramp_rates[mid:].min()),
            peak_generation   = float(arr.max()),
            peak_hour         = int(arr.argmax()),
            high_ramp_events  = np.where(np.abs(ramp_rates)>HIGH_RAMP_THRESHOLD)[0].tolist(),
            stability_score   = round(stability,1),
            stress_level      = stress,
            ramp_rates        = ramp_rates.tolist(),
        )

    def predict_curtailment_strategy(analysis, predictions):
        rr       = np.asarray(analysis["ramp_rates"])
        schedule = []
        for i, (p, r) in enumerate(zip(predictions, rr)):
            ar = abs(r)
            if ar > HIGH_RAMP_THRESHOLD:
                action = "CURTAIL"; pct = min(40.0,(ar/HIGH_RAMP_THRESHOLD-1)*20+20)
            elif ar > CURTAILMENT_RAMP_THRESHOLD:
                action = "REDUCE";  pct = min(20.0,(ar/CURTAILMENT_RAMP_THRESHOLD-1)*10+5)
            else:
                action = "NORMAL";  pct = 0.0
            schedule.append(dict(hour=i, ghi_forecast=round(float(p),2),
                                 ramp_rate=round(float(r),2),
                                 action=action, curtailment_pct=round(pct,1)))
        s = analysis["stability_score"]
        rec = ("Full dispatch recommended — grid is stable." if s>=80
               else "Moderate curtailment advised during high-ramp windows." if s>=60
               else "Significant curtailment required. Coordinate with grid operator.")
        return dict(curtailment_schedule=schedule, dispatch_recommendation=rec,
                    high_ramp_hours=analysis["high_ramp_events"],
                    stability_score=s, stress_level=analysis["stress_level"])

    def print_duck_curve_summary(analysis, curtailment):
        W = 60
        print("\n" + "=" * W)
        print("  DUCK CURVE & GRID STABILITY SUMMARY")
        print("=" * W)
        print(f"  Morning ramp rate  : {analysis['morning_ramp_rate']:+.1f} MW/h")
        print(f"  Evening ramp rate  : {analysis['evening_ramp_rate']:+.1f} MW/h")
        print(f"  Peak generation    : {analysis['peak_generation']:.1f} W/m²"
              f"  at {analysis['peak_hour']:02d}:00")
        print(f"  Stability score    : {analysis['stability_score']}/100"
              f"  [{analysis['stress_level']} STRESS]")
        if analysis["high_ramp_events"]:
            hrs = ", ".join(f"{h:02d}:00" for h in analysis["high_ramp_events"])
            print(f"  High-ramp hours    : {hrs}")
        else:
            print("  High-ramp hours    : None detected ✓")
        print(f"\n  Dispatch advisory  : {curtailment['dispatch_recommendation']}")
        curtailed = [s for s in curtailment["curtailment_schedule"]
                     if s["action"] != "NORMAL"]
        if curtailed:
            print(f"  Curtailment slots  : {len(curtailed)} h requiring REDUCE/CURTAIL action")
        else:
            print("  Curtailment slots  : None required ✓")
        print("=" * W)

# ═════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════
LATITUDE         = 19.076          # IIT Bombay, Mumbai
SOLAR_FARM_MW    = 100.0           # Installed PV capacity (MW)
PANEL_EFFICIENCY = 0.20            # 20 % panel efficiency
INVERTER_EFF     = 0.97            # Inverter losses
TEMP_COEFF       = -0.0045         # -0.45 %/°C above 25°C
PANEL_AREA_M2    = (SOLAR_FARM_MW * 1e6) / (1000 * PANEL_EFFICIENCY)  # m²

# Dark-theme colour palette
BG     = "#0D1B2A"
PANEL  = "#112236"
BLUE   = "#1E88E5"
CYAN   = "#00BCD4"
ORANGE = "#FFA726"
GREEN  = "#66BB6A"
RED    = "#EF5350"
PURPLE = "#AB47BC"
WHITE  = "#FFFFFF"
LGRAY  = "#B0BEC5"
YELLOW = "#FFD54F"
PINK   = "#F48FB1"


# ═════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING & GENERATION
# ═════════════════════════════════════════════════════════════════════════════

def load_or_generate_data(csv_path: str, n_days: int = 90) -> pd.DataFrame:
    """
    Load GHI data from CSV, or synthesise n_days of realistic Mumbai data.

    Returns a DataFrame with columns: timestamp, ghi
    """
    default_path = _REPO / "data" / "sample_solar_data.csv"
    path = Path(csv_path) if csv_path else default_path

    if path.exists():
        df = pd.read_csv(path, parse_dates=["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        df["ghi"] = df["ghi"].clip(lower=0.0).interpolate(method="linear").fillna(0.0)
        print(f"  ✓  Loaded {len(df):,} rows from {path.name}")
        # Extend to at least 90 days so seasonal plots are meaningful
        if len(df) < 90 * 24:
            df = _extend_data(df, target_days=90)
    else:
        print(f"  ⚠  {path.name} not found — generating {n_days}-day synthetic dataset.")
        df = _synthesise_data(n_days=n_days)

    return df


def _synthesise_data(n_days: int = 90, lat: float = LATITUDE) -> pd.DataFrame:
    """
    Generate synthetic hourly GHI using the Spencer/Iqbal clear-sky model
    with stochastic cloud cover, calibrated to Mumbai weather patterns.
    """
    rng  = np.random.default_rng(42)
    base = pd.Timestamp("2025-01-01")
    ts   = pd.date_range(base, periods=n_days * 24, freq="h")

    hour   = ts.hour
    doy    = ts.dayofyear
    lat_r  = np.radians(lat)
    ha     = np.radians(15.0 * (hour - 12.0))
    decl   = np.radians(23.45 * np.sin(np.radians(360/365 * (doy - 81))))
    cz     = np.maximum(np.sin(lat_r)*np.sin(decl) + np.cos(lat_r)*np.cos(decl)*np.cos(ha), 0.0)
    et_rad = compute_extraterrestrial_radiation(doy)
    clear  = et_rad * cz * 0.78          # clear-sky GHI ≈ 78% of ET

    # Seasonal cloud factor: monsoon (Jun–Sep) is heavily cloudy
    month     = ts.month
    cloud_mu  = np.where((month >= 6) & (month <= 9), 0.45, 0.72)  # Kt mean
    cloud_sig = np.where((month >= 6) & (month <= 9), 0.18, 0.12)
    kt        = np.clip(rng.normal(cloud_mu, cloud_sig), 0.05, 1.0)

    # Zero out night-time
    kt   = np.where(cz < 0.05, 0.0, kt)
    ghi  = np.round(np.maximum(clear * kt, 0.0), 2)

    return pd.DataFrame({"timestamp": ts, "ghi": ghi})


def _extend_data(df: pd.DataFrame, target_days: int = 90) -> pd.DataFrame:
    """Tile existing data to reach target_days."""
    have_days = (df["timestamp"].max() - df["timestamp"].min()).days + 1
    if have_days >= target_days:
        return df
    copies = [df]
    offset = have_days
    while offset < target_days:
        extra = df.copy()
        extra["timestamp"] = extra["timestamp"] + pd.Timedelta(days=offset)
        copies.append(extra)
        offset += have_days
    combined = pd.concat(copies, ignore_index=True)
    start = combined["timestamp"].min()
    end   = start + pd.Timedelta(days=target_days)
    return combined[combined["timestamp"] < end].reset_index(drop=True)


# ═════════════════════════════════════════════════════════════════════════════
# 2. GHI → PV POWER CONVERSION
# ═════════════════════════════════════════════════════════════════════════════

def ghi_to_pv_power(ghi_wm2: np.ndarray,
                    ambient_temp_c: np.ndarray,
                    capacity_mw: float = SOLAR_FARM_MW) -> np.ndarray:
    """
    Convert GHI (W/m²) to AC power output (MW) for a utility-scale farm.

    Model:
        DC power = GHI × panel_area × η_panel
        Temperature derating: η_temp = 1 + α × (T_cell – 25)
            T_cell ≈ T_ambient + 25 × (GHI / 800)   [NOCT approximation]
        AC power = DC power × η_inverter

    Args:
        ghi_wm2:       irradiance array (W/m²)
        ambient_temp_c: ambient temperature (°C)
        capacity_mw:   installed DC capacity (MW)

    Returns:
        AC power output (MW), clipped to [0, capacity_mw]
    """
    ghi = np.asarray(ghi_wm2, float)
    T   = np.asarray(ambient_temp_c, float)

    # NOCT cell temperature
    T_cell = T + 25.0 * (ghi / 800.0)

    # Temperature derating factor
    eta_temp = 1.0 + TEMP_COEFF * (T_cell - 25.0)
    eta_temp = np.clip(eta_temp, 0.5, 1.05)

    # Power (W), then convert to MW
    dc_power_w  = ghi * PANEL_AREA_M2 * PANEL_EFFICIENCY * eta_temp
    ac_power_mw = dc_power_w * INVERTER_EFF / 1e6

    return np.clip(ac_power_mw, 0.0, capacity_mw)


def synthetic_temperature(timestamps: pd.DatetimeIndex) -> np.ndarray:
    """
    Realistic hourly temperature profile for Mumbai (°C).
    Annual range: 18–40°C; daily cycle ±4°C; monsoon cooler.
    """
    doy   = timestamps.dayofyear
    hour  = timestamps.hour
    month = timestamps.month

    # Seasonal baseline (peaks in May, lowest in Jan)
    seasonal = 29.0 + 11.0 * np.sin(np.radians(360/365*(doy - 120)))
    # Monsoon cooling
    monsoon_cool = np.where((month >= 6) & (month <= 9), -5.0, 0.0)
    # Diurnal cycle: coolest at 5am, hottest at 2pm
    diurnal = 4.0 * np.sin(np.radians(360/24*(hour - 5)))

    return seasonal + monsoon_cool + diurnal


# ═════════════════════════════════════════════════════════════════════════════
# 3. GRID DEMAND MODEL
# ═════════════════════════════════════════════════════════════════════════════

def simulate_grid_demand(timestamps: pd.DatetimeIndex,
                         base_load_mw: float = 800.0) -> np.ndarray:
    """
    Realistic Mumbai-style grid demand (MW).

    Profile shape:
      • Night trough (0–5h): ~70% of base
      • Morning ramp (5–9h): rises sharply as city wakes
      • Midday plateau (9–18h): high industrial + commercial demand
      • Evening peak (18–22h): residential surge — this is the duck's neck
      • Night decline (22–24h)

    Seasonal modulation:
      • Summer: +15% (AC loads)
      • Monsoon: –8% (mild weather)
      • Winter: base
    """
    h     = timestamps.hour
    month = timestamps.month

    # Normalised hourly demand shape (0–1)
    shape = np.array([
        0.70, 0.68, 0.67, 0.67, 0.68, 0.72,   # 00–05
        0.78, 0.88, 0.95, 0.98, 0.99, 1.00,   # 06–11
        0.99, 0.98, 0.96, 0.95, 0.96, 0.98,   # 12–17
        1.05, 1.10, 1.08, 1.02, 0.92, 0.80,   # 18–23
    ])

    demand = shape[h] * base_load_mw

    # Seasonal factor
    summer  = ((month >= 3) & (month <= 5)).astype(float)
    monsoon = ((month >= 6) & (month <= 9)).astype(float)
    demand  = demand * (1 + 0.15*summer - 0.08*monsoon)

    # Small random noise ±2%
    rng    = np.random.default_rng(0)
    noise  = rng.uniform(-0.02, 0.02, len(demand))
    demand = demand * (1 + noise)

    return demand


# ═════════════════════════════════════════════════════════════════════════════
# 4. SINGLE-DAY SIMULATION
# ═════════════════════════════════════════════════════════════════════════════

def simulate_day(day_df: pd.DataFrame) -> dict:
    """
    Run the full duck-curve simulation for a single 24-hour day.

    Args:
        day_df: DataFrame with 24 rows, columns: timestamp, ghi

    Returns:
        dict with all simulation outputs
    """
    assert len(day_df) == 24, "Expects exactly 24 hourly rows."

    ts    = pd.DatetimeIndex(day_df["timestamp"])
    hours = np.arange(24)
    ghi   = day_df["ghi"].values

    # Temperature + PV power
    temp      = synthetic_temperature(ts)
    pv_power  = ghi_to_pv_power(ghi, temp)

    # Grid demand
    demand    = simulate_grid_demand(ts)

    # Net demand = demand – solar  (the "duck")
    net_demand = demand - pv_power

    # Curtailed power: clamp negative net to 0, record excess
    curtailed  = np.maximum(-net_demand, 0.0)
    net_demand = np.maximum(net_demand, 0.0)

    # Duck-curve analysis (uses GHI values as proxy for generation profile)
    analysis     = analyze_duck_curve(pv_power.tolist())
    curtailment  = predict_curtailment_strategy(analysis, pv_power.tolist())

    # Ramp rates on demand and net-demand
    demand_ramp     = np.diff(demand,     prepend=demand[0])
    net_demand_ramp = np.diff(net_demand, prepend=net_demand[0])
    pv_ramp         = np.diff(pv_power,   prepend=pv_power[0])

    # Clearness index
    zenith = compute_solar_zenith_angle(ts, latitude=LATITUDE)
    kt     = compute_clearness_index(ghi, ts.dayofyear, zenith)

    # Energy totals (MWh = MW × 1 hour)
    daily_solar_kwh  = float(pv_power.sum())
    daily_demand_kwh = float(demand.sum())
    solar_fraction   = float(pv_power.sum() / demand.sum() * 100)

    return dict(
        date             = str(ts[0].date()),
        hours            = hours,
        timestamps       = ts,
        ghi              = ghi,
        pv_power         = pv_power,
        demand           = demand,
        net_demand       = net_demand,
        curtailed        = curtailed,
        temp             = temp,
        zenith           = zenith,
        kt               = kt,
        demand_ramp      = demand_ramp,
        net_demand_ramp  = net_demand_ramp,
        pv_ramp          = pv_ramp,
        analysis         = analysis,
        curtailment      = curtailment,
        daily_solar_mwh  = daily_solar_kwh,
        daily_demand_mwh = daily_demand_kwh,
        solar_fraction   = solar_fraction,
    )


# ═════════════════════════════════════════════════════════════════════════════
# 5. MULTI-DAY / SEASONAL SIMULATION
# ═════════════════════════════════════════════════════════════════════════════

def simulate_all_days(df: pd.DataFrame) -> list[dict]:
    """Run simulate_day() for every complete 24-hour day in df."""
    results = []
    grouped = df.groupby(df["timestamp"].dt.date)
    for date, group in grouped:
        if len(group) < 24:
            continue
        group_24 = group.iloc[:24].copy().reset_index(drop=True)
        if group_24["ghi"].sum() == 0:
            continue   # skip fully-null days
        try:
            results.append(simulate_day(group_24))
        except Exception:
            continue
    print(f"  ✓  Simulated {len(results)} days")
    return results


def pick_representative_days(results: list[dict]) -> dict:
    """
    Choose one representative day per calendar season.
    Returns dict: {season_name: day_result}
    """
    seasons = {
        "Winter (Jan)"   : [1],
        "Spring (Mar)"   : [3],
        "Summer (May)"   : [5],
        "Monsoon (Jul)"  : [7],
    }
    picked = {}
    for name, months in seasons.items():
        candidates = [
            r for r in results
            if pd.Timestamp(r["date"]).month in months
        ]
        if candidates:
            # Pick the day closest to median solar fraction
            fracs = [c["solar_fraction"] for c in candidates]
            med   = float(np.median(fracs))
            best  = min(candidates, key=lambda c: abs(c["solar_fraction"]-med))
            picked[name] = best
    return picked


# ═════════════════════════════════════════════════════════════════════════════
# 6. PLOTTING
# ═════════════════════════════════════════════════════════════════════════════

def _style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_edgecolor("#1E3A5F")
    ax.tick_params(colors=LGRAY, labelsize=8)
    ax.xaxis.label.set_color(LGRAY)
    ax.yaxis.label.set_color(LGRAY)
    if title:
        ax.set_title(title, color=WHITE, fontsize=10, fontweight="bold", pad=6)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(color="#1E3A5F", linestyle="--", linewidth=0.6, alpha=0.5)


def _hour_ticks(ax, step=3):
    ax.set_xticks(range(0, 24, step))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, step)],
                       rotation=30, ha="right", fontsize=7.5)


def plot_simulation(day: dict,
                    all_results: list[dict],
                    seasonal: dict,
                    save_path: str = None):
    """
    6-panel figure:
      [0] Duck curve main  — GHI, PV, Demand, Net demand, Curtailment
      [1] Ramp rates       — PV ramp & net-demand ramp per hour
      [2] Curtailment heat — action per hour (NORMAL / REDUCE / CURTAIL)
      [3] Stability trend  — stability score across all simulated days
      [4] Seasonal overlay — duck curves for 4 seasons
      [5] Energy breakdown — daily bar chart (solar / other / curtailed)
    """
    fig = plt.figure(figsize=(18, 14), facecolor=BG)
    gs  = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38,
                   top=0.92, bottom=0.06, left=0.06, right=0.97)

    ax0 = fig.add_subplot(gs[0, :2])   # main duck curve — wide
    ax1 = fig.add_subplot(gs[0, 2])    # ramp rates
    ax2 = fig.add_subplot(gs[1, :2])   # curtailment heatmap
    ax3 = fig.add_subplot(gs[1, 2])    # stability trend
    ax4 = fig.add_subplot(gs[2, :2])   # seasonal overlay
    ax5 = fig.add_subplot(gs[2, 2])    # energy breakdown

    h  = day["hours"]
    c  = day["curtailment"]["curtailment_schedule"]

    # ── [0] MAIN DUCK CURVE ──────────────────────────────────────────────────
    _style_ax(ax0, title=f"Duck Curve — {day['date']} | "
              f"Solar Fraction: {day['solar_fraction']:.1f}%  |  "
              f"Stability: {day['analysis']['stability_score']}/100 "
              f"[{day['analysis']['stress_level']}]",
              xlabel="Hour", ylabel="Power (MW)")

    ax0.fill_between(h, day["pv_power"],  alpha=0.20, color=ORANGE)
    ax0.fill_between(h, day["net_demand"],alpha=0.12, color=BLUE)
    ax0.fill_between(h, day["curtailed"], alpha=0.25, color=RED,
                     label="Curtailed Generation")

    ax0.plot(h, day["demand"],    color=BLUE,   lw=2.2, label="Grid Demand")
    ax0.plot(h, day["pv_power"],  color=ORANGE, lw=2.2, label="PV Generation")
    ax0.plot(h, day["net_demand"],color=CYAN,   lw=2.2,
             linestyle="--", label="Net Demand (Duck)")

    # Annotate peak PV
    pk = int(np.argmax(day["pv_power"]))
    ax0.annotate(f"Peak: {day['pv_power'][pk]:.0f} MW",
                 xy=(pk, day["pv_power"][pk]),
                 xytext=(pk+1.5, day["pv_power"][pk]+25),
                 arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.5),
                 color=ORANGE, fontsize=8, fontweight="bold")

    # Zone shading
    ax0.axvspan(6,  10, alpha=0.05, color=GREEN,  label="_nolegend_")
    ax0.axvspan(11, 14, alpha=0.05, color=YELLOW, label="_nolegend_")
    ax0.axvspan(17, 20, alpha=0.05, color=RED,    label="_nolegend_")

    ax0.text(8,   ax0.get_ylim()[0] + 10, "Morning\nRamp ↑",
             ha="center", color=GREEN,  fontsize=7, fontweight="bold")
    ax0.text(12.5, ax0.get_ylim()[0] + 10, "Peak\nGeneration",
             ha="center", color=YELLOW, fontsize=7, fontweight="bold")
    ax0.text(18.5, ax0.get_ylim()[0] + 10, "Evening\nRamp ↓",
             ha="center", color=RED,    fontsize=7, fontweight="bold")

    ax0.legend(facecolor=PANEL, edgecolor="#1E3A5F", labelcolor=WHITE,
               fontsize=8, loc="upper left")
    _hour_ticks(ax0)

    # ── [1] RAMP RATES ───────────────────────────────────────────────────────
    _style_ax(ax1, title="Hourly Ramp Rates", xlabel="Hour", ylabel="MW/h")

    ax1.bar(h, day["pv_ramp"],         color=ORANGE, alpha=0.75,
            label="PV Ramp",          width=0.45, align="center")
    ax1.bar(h + 0.45, day["net_demand_ramp"], color=CYAN, alpha=0.75,
            label="Net Demand Ramp",  width=0.45, align="center")

    ax1.axhline(0, color=LGRAY, lw=0.8, linestyle="-")
    ax1.axhline( HIGH_RAMP_THRESHOLD,        color=RED,    lw=1.2,
                linestyle=":", alpha=0.8, label="High ramp threshold")
    ax1.axhline(-HIGH_RAMP_THRESHOLD,        color=RED,    lw=1.2,
                linestyle=":",  alpha=0.8)
    ax1.legend(facecolor=PANEL, edgecolor="#1E3A5F", labelcolor=WHITE,
               fontsize=7, loc="upper right")
    _hour_ticks(ax1)

    # ── [2] CURTAILMENT HEATMAP ──────────────────────────────────────────────
    _style_ax(ax2, title="Curtailment Action by Hour",
              xlabel="Hour", ylabel="Curtailment (%)")

    action_colors = {"NORMAL": GREEN, "REDUCE": ORANGE, "CURTAIL": RED}
    labels_seen   = set()
    for slot in c:
        col   = action_colors[slot["action"]]
        label = slot["action"] if slot["action"] not in labels_seen else "_nolegend_"
        labels_seen.add(slot["action"])
        ax2.bar(slot["hour"], slot["curtailment_pct"] if slot["curtailment_pct"]>0 else 1,
                color=col, alpha=0.85, width=0.85, label=label)
        if slot["curtailment_pct"] > 0:
            ax2.text(slot["hour"], slot["curtailment_pct"]+0.3,
                     f"{slot['curtailment_pct']:.0f}%",
                     ha="center", va="bottom", color=WHITE, fontsize=7)

    ax2.set_ylim(0, 45)
    ax2.legend(facecolor=PANEL, edgecolor="#1E3A5F", labelcolor=WHITE,
               fontsize=8, loc="upper right")
    _hour_ticks(ax2)

    # Second y-axis: GHI
    ax2b = ax2.twinx()
    ax2b.plot(h, day["ghi"], color=YELLOW, lw=1.5, alpha=0.6, linestyle=":")
    ax2b.set_ylabel("GHI (W/m²)", color=YELLOW, fontsize=8)
    ax2b.tick_params(axis="y", colors=YELLOW, labelsize=8)
    ax2b.spines["right"].set_edgecolor("#1E3A5F")
    ax2b.set_facecolor(PANEL)

    # ── [3] STABILITY TREND ──────────────────────────────────────────────────
    _style_ax(ax3, title="Stability Score (All Days)",
              xlabel="Day", ylabel="Score / 100")

    dates_all = range(len(all_results))
    scores    = [r["analysis"]["stability_score"] for r in all_results]
    colors_s  = [GREEN if s>=80 else (ORANGE if s>=60 else RED) for s in scores]

    ax3.scatter(dates_all, scores, c=colors_s, s=12, alpha=0.7, zorder=3)
    # 7-day rolling mean
    if len(scores) >= 7:
        roll = pd.Series(scores).rolling(7, center=True).mean()
        ax3.plot(dates_all, roll, color=WHITE, lw=1.8, alpha=0.8, label="7-day avg")
    ax3.axhline(80, color=GREEN,  lw=1, linestyle=":", alpha=0.7)
    ax3.axhline(60, color=ORANGE, lw=1, linestyle=":", alpha=0.7)
    ax3.set_ylim(0, 105)
    ax3.legend(facecolor=PANEL, edgecolor="#1E3A5F", labelcolor=WHITE, fontsize=7)

    patches = [mpatches.Patch(color=GREEN,  label="LOW stress (≥80)"),
               mpatches.Patch(color=ORANGE, label="MOD stress (60–79)"),
               mpatches.Patch(color=RED,    label="HIGH stress (<60)")]
    ax3.legend(handles=patches, facecolor=PANEL, edgecolor="#1E3A5F",
               labelcolor=WHITE, fontsize=6.5, loc="lower right")

    # ── [4] SEASONAL OVERLAY ─────────────────────────────────────────────────
    _style_ax(ax4, title="Seasonal Duck Curve Comparison",
              xlabel="Hour", ylabel="Power (MW)")

    season_colors = [CYAN, GREEN, ORANGE, BLUE]
    for (sname, sday), col in zip(seasonal.items(), season_colors):
        ax4.plot(sday["hours"], sday["net_demand"], color=col, lw=2.0,
                 label=f"{sname}  (SF: {sday['solar_fraction']:.0f}%)")
        ax4.fill_between(sday["hours"], sday["pv_power"], alpha=0.08, color=col)

    ax4.legend(facecolor=PANEL, edgecolor="#1E3A5F", labelcolor=WHITE,
               fontsize=8, loc="upper left")
    _hour_ticks(ax4)

    # ── [5] ENERGY BREAKDOWN ─────────────────────────────────────────────────
    _style_ax(ax5, title="Daily Energy Mix",
              xlabel="Day index", ylabel="MWh")

    n_bars = min(len(all_results), 60)
    subset = all_results[-n_bars:]
    xs     = np.arange(n_bars)

    solar_mwh   = [r["daily_solar_mwh"]   for r in subset]
    demand_mwh  = [r["daily_demand_mwh"]  for r in subset]
    other_mwh   = [max(0, d - s) for d, s in zip(demand_mwh, solar_mwh)]
    curtail_mwh = [float(r["curtailed"].sum()) for r in subset]

    ax5.bar(xs, solar_mwh, color=ORANGE, alpha=0.85, label="Solar (MWh)", width=0.7)
    ax5.bar(xs, other_mwh, bottom=solar_mwh, color=BLUE, alpha=0.7,
            label="Other sources", width=0.7)
    ax5.bar(xs, curtail_mwh, bottom=[s+o for s,o in zip(solar_mwh, other_mwh)],
            color=RED, alpha=0.65, label="Curtailed", width=0.7)

    ax5.legend(facecolor=PANEL, edgecolor="#1E3A5F", labelcolor=WHITE,
               fontsize=7, loc="upper left")

    # ── Super-title ──────────────────────────────────────────────────────────
    fig.text(0.5, 0.965,
             "☀️  Duck Curve Simulation — Solar Irradiance & PV Power Prediction",
             ha="center", color=WHITE, fontsize=15, fontweight="bold")
    fig.text(0.5, 0.945,
             f"IIT Bombay  |  {SOLAR_FARM_MW:.0f} MW Farm  |  "
             f"Lat {LATITUDE}° (Mumbai)  |  "
             f"{len(all_results)} days simulated",
             ha="center", color=LGRAY, fontsize=9)

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=BG, edgecolor="none")
        print(f"\n  ✓  Plot saved → {save_path}")
    else:
        out = str(_REPO / "duck_curve_simulation.png")
        fig.savefig(out, dpi=150, bbox_inches="tight",
                    facecolor=BG, edgecolor="none")
        print(f"\n  ✓  Plot saved → {out}")

    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# 7. TERMINAL REPORT
# ═════════════════════════════════════════════════════════════════════════════

def print_report(day: dict, all_results: list[dict]):
    """Full terminal report for the chosen simulation day."""
    a = day["analysis"]
    ct = day["curtailment"]
    c  = ct["curtailment_schedule"]

    W = 68
    print("\n" + "=" * W)
    print("  DUCK CURVE SIMULATION REPORT")
    print(f"  Date : {day['date']}  |  Farm: {SOLAR_FARM_MW:.0f} MW  |  "
          f"Location: Mumbai (lat {LATITUDE}°)")
    print("=" * W)

    print("\n── SOLAR GENERATION ──────────────────────────────────────────")
    print(f"  Peak GHI          : {day['ghi'].max():.1f} W/m²"
          f"  at {int(day['ghi'].argmax()):02d}:00")
    print(f"  Peak PV Power     : {day['pv_power'].max():.1f} MW"
          f"  at {int(day['pv_power'].argmax()):02d}:00")
    print(f"  Daily Solar Energy: {day['daily_solar_mwh']:.0f} MWh")
    print(f"  Solar Fraction    : {day['solar_fraction']:.1f}%"
          "  (solar / total demand)")

    print("\n── DUCK CURVE ANALYSIS ────────────────────────────────────────")
    print(f"  Morning Ramp Rate : {a['morning_ramp_rate']:+.1f} MW/h")
    print(f"  Evening Ramp Rate : {a['evening_ramp_rate']:+.1f} MW/h")
    print(f"  Peak Hour         : {a['peak_hour']:02d}:00")
    print(f"  Stability Score   : {a['stability_score']}/100"
          f"  [{a['stress_level']} STRESS]")
    if a["high_ramp_events"]:
        hrs = ", ".join(f"{h:02d}:00" for h in a["high_ramp_events"])
        print(f"  High Ramp Events  : {hrs}")
    else:
        print("  High Ramp Events  : None ✓")

    print("\n── CURTAILMENT SCHEDULE ───────────────────────────────────────")
    print(f"  {ct['dispatch_recommendation']}")
    curtailed_slots = [s for s in c if s["action"] != "NORMAL"]
    if curtailed_slots:
        print(f"\n  {'Hour':>5} | {'GHI (W/m²)':>11} | {'Ramp':>8} | "
              f"{'Action':>8} | {'Curtail':>8}")
        print("  " + "-" * 54)
        for s in curtailed_slots:
            print(f"  {s['hour']:02d}:00  | {s['ghi_forecast']:>9.1f}   | "
                  f"{s['ramp_rate']:>+7.1f}  | {s['action']:>8} | "
                  f"{s['curtailment_pct']:>6.1f}%")
    else:
        print("  No curtailment required. ✓")

    # Multi-day summary
    if len(all_results) > 1:
        scores  = [r["analysis"]["stability_score"] for r in all_results]
        fracs   = [r["solar_fraction"] for r in all_results]
        n_high  = sum(1 for r in all_results if r["analysis"]["stress_level"]=="HIGH")
        n_mod   = sum(1 for r in all_results if r["analysis"]["stress_level"]=="MODERATE")
        n_low   = sum(1 for r in all_results if r["analysis"]["stress_level"]=="LOW")

        print("\n── MULTI-DAY SUMMARY ──────────────────────────────────────────")
        print(f"  Days simulated        : {len(all_results)}")
        print(f"  Avg stability score   : {np.mean(scores):.1f}/100")
        print(f"  Min stability score   : {min(scores):.1f}/100")
        print(f"  Days LOW stress       : {n_low}  ({n_low/len(all_results)*100:.0f}%)")
        print(f"  Days MODERATE stress  : {n_mod}  ({n_mod/len(all_results)*100:.0f}%)")
        print(f"  Days HIGH stress      : {n_high}  ({n_high/len(all_results)*100:.0f}%)")
        print(f"  Avg solar fraction    : {np.mean(fracs):.1f}%")
        print(f"  Max solar fraction    : {max(fracs):.1f}%")

    print("\n" + "=" * W)
    print("  SIMULATION COMPLETE")
    print("=" * W + "\n")


# ═════════════════════════════════════════════════════════════════════════════
# 8. MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Duck Curve Simulation — cautious-enigma repo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--data", default=None,
        help="Path to GHI CSV (timestamp + ghi columns). "
             "Defaults to data/sample_solar_data.csv"
    )
    parser.add_argument(
        "--date", default=None,
        help="ISO date to analyse in detail, e.g. 2025-06-15. "
             "Defaults to the day with median solar fraction."
    )
    parser.add_argument(
        "--capacity", type=float, default=SOLAR_FARM_MW,
        help=f"Solar farm capacity in MW (default: {SOLAR_FARM_MW})"
    )
    parser.add_argument(
        "--save", default=None,
        help="File path to save the plot, e.g. plots/duck_curve.png"
    )
    args = parser.parse_args()

    # Update module-level capacity constants
    import sys as _sys
    _mod = _sys.modules[__name__]
    _mod.SOLAR_FARM_MW = args.capacity
    _mod.PANEL_AREA_M2 = (args.capacity * 1e6) / (1000 * PANEL_EFFICIENCY)

    print("\n" + "=" * 68)
    print("  Duck Curve Simulation — Solar Irradiance & PV Power Prediction")
    print("  IIT Bombay | cautious-enigma repo")
    print("=" * 68)

    # ── Load data ─────────────────────────────────────────────────────────
    print("\n[1/4] Loading data …")
    df = load_or_generate_data(args.data)

    # ── Simulate all days ─────────────────────────────────────────────────
    print("\n[2/4] Running simulation across all days …")
    all_results = simulate_all_days(df)

    if not all_results:
        print("ERROR: No valid days found in data.")
        sys.exit(1)

    # ── Pick focus day ────────────────────────────────────────────────────
    print("\n[3/4] Selecting focus day …")
    if args.date:
        matches = [r for r in all_results if r["date"] == args.date]
        if not matches:
            print(f"  ⚠  {args.date} not found — using median-solar-fraction day.")
            fracs = [r["solar_fraction"] for r in all_results]
            focus = all_results[int(np.argsort(fracs)[len(fracs)//2])]
        else:
            focus = matches[0]
    else:
        fracs = [r["solar_fraction"] for r in all_results]
        focus = all_results[int(np.argsort(fracs)[len(fracs)//2])]

    print(f"  ✓  Focus day: {focus['date']}  "
          f"(solar fraction {focus['solar_fraction']:.1f}%)")

    # ── Seasonal representatives ──────────────────────────────────────────
    seasonal = pick_representative_days(all_results)
    if not seasonal:
        seasonal = {"Simulated": focus}

    # ── Plot ──────────────────────────────────────────────────────────────
    print("\n[4/4] Generating plots …")
    plot_simulation(focus, all_results, seasonal, save_path=args.save)

    # ── Terminal report ───────────────────────────────────────────────────
    print_report(focus, all_results)


if __name__ == "__main__":
    main()
